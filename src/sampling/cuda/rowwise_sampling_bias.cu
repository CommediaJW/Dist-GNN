#include <curand_kernel.h>

#include "../../common/cuda/cub_function.h"
#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "ops.h"
#include "warpselect/WarpSelect.cuh"

#define BLOCK_SIZE 128

namespace dgs {
namespace sampling {
namespace cuda {

template <typename NType, typename EType>
inline std::pair<torch::Tensor, torch::Tensor> _GetSubAndTempIndptr(
    torch::Tensor seeds, torch::Tensor indptr, NType num_pick, bool replace) {
  NType num_items = seeds.numel();
  torch::Tensor sub_indptr = torch::empty(
      (num_items + 1),
      torch::TensorOptions().dtype(indptr.dtype()).device(torch::kCUDA));
  torch::Tensor temp_indptr = torch::empty(
      (num_items + 1),
      torch::TensorOptions().dtype(indptr.dtype()).device(torch::kCUDA));
  thrust::device_ptr<EType> sub_prefix(
      static_cast<EType *>(sub_indptr.data_ptr<EType>()));
  thrust::device_ptr<EType> temp_prefix(
      static_cast<EType *>(temp_indptr.data_ptr<EType>()));

  using it = thrust::counting_iterator<NType>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [in = seeds.data_ptr<NType>(), in_indptr = indptr.data_ptr<EType>(),
       sub_ptr = thrust::raw_pointer_cast(sub_prefix),
       tmp_ptr = thrust::raw_pointer_cast(temp_prefix), replace, num_pick,
       num_items] __device__(int i) mutable {
        NType row = in[i];
        EType begin = in_indptr[row];
        EType end = in_indptr[row + 1];
        EType deg = end - begin;
        if (replace) {
          sub_ptr[i] = deg == 0 ? 0 : num_pick;
          tmp_ptr[i] = deg;
        } else {
          sub_ptr[i] = MIN(deg, num_pick);
          tmp_ptr[i] = deg > num_pick ? deg : 0;
        }
        if (i == num_items - 1) {
          sub_ptr[num_items] = 0;
          tmp_ptr[num_items] = 0;
        }
      });

  common::cuda::cub_exclusiveSum<EType>(thrust::raw_pointer_cast(sub_prefix),
                                        num_items + 1);
  common::cuda::cub_exclusiveSum<EType>(thrust::raw_pointer_cast(temp_prefix),
                                        num_items + 1);
  return {sub_indptr, temp_indptr};
}

template <typename NType, typename EType, typename FloatType, int TILE_SIZE,
          int BLOCK_WARPS, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void _CSRRowWiseSampleBiasKernel(
    const uint64_t rand_seed, const NType num_picks, const NType num_rows,
    const NType *__restrict__ const in_rows,
    const EType *__restrict__ const in_ptr,
    const NType *__restrict__ const in_cols,
    const FloatType *__restrict__ const prob,
    const EType *__restrict__ const out_ptr, NType *__restrict__ const out_rows,
    NType *const out_cols) {
  // we assign one warp per row
  assert(num_picks <= 32);
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  __shared__ EType warpselect_out_index[WARP_SIZE * BLOCK_WARPS];

  // init warpselect
  warpselect::WarpSelect<FloatType, EType,
                         true,  // produce largest values
                         warpselect::Comparator<FloatType>, NumWarpQ,
                         NumThreadQ, WARP_SIZE * BLOCK_WARPS>
      heap(warpselect::_Limits<FloatType>::getMin(), -1, num_picks);

  NType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const NType last_row =
      MIN(static_cast<NType>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

  int laneid = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.y;
  EType *warpselect_out_index_per_warp =
      warpselect_out_index + warp_id * WARP_SIZE;

  while (out_row < last_row) {
    const NType row = in_rows[out_row];
    const EType in_row_start = in_ptr[row];
    const EType deg = in_ptr[row + 1] - in_row_start;
    const EType out_row_start = out_ptr[out_row];
    // A-Res value needs to be calculated only if deg is greater than num_picks
    // in weighted rowwise sampling without replacement
    if (deg > num_picks) {
      heap.reset();
      int limit = warpselect::roundDown(deg, WARP_SIZE);
      EType i = laneid;

      for (; i < limit; i += WARP_SIZE) {
        FloatType item_prob = prob[in_row_start + i];
        FloatType ares_prob = __powf(curand_uniform(&rng), 1.0f / item_prob);
        heap.add(ares_prob, i);
      }

      if (i < deg) {
        FloatType item_prob = prob[in_row_start + i];
        FloatType ares_prob = __powf(curand_uniform(&rng), 1.0f / item_prob);
        heap.addThreadQ(ares_prob, i);
        i += WARP_SIZE;
      }

      heap.reduce();
      heap.writeOutV(warpselect_out_index_per_warp, num_picks);

      for (int idx = laneid; idx < num_picks; idx += WARP_SIZE) {
        const EType out_idx = out_row_start + idx;
        const EType in_idx = warpselect_out_index_per_warp[idx] + in_row_start;
        out_rows[out_idx] = static_cast<NType>(row);
        out_cols[out_idx] = in_cols[in_idx];
      }
    } else {
      for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        // get in and out index
        const EType out_idx = out_row_start + idx;
        const EType in_idx = in_row_start + idx;
        // copy permutation over
        out_rows[out_idx] = static_cast<NType>(row);
        out_cols[out_idx] = in_cols[in_idx];
      }
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename NType, typename EType, typename FloatType, int TILE_SIZE,
          int BLOCK_WARPS, int WARP_SIZE>
__global__ void _CSRRowWiseSampleBiasReplaceKernel(
    const uint64_t rand_seed, const NType num_picks, const NType num_rows,
    const NType *__restrict__ const in_rows,
    const EType *__restrict__ const in_ptr,
    const NType *__restrict__ const in_cols,
    const FloatType *__restrict__ const prob,
    const EType *__restrict__ const out_ptr,
    const EType *__restrict__ const cdf_ptr, FloatType *__restrict__ const cdf,
    NType *__restrict__ const out_rows, NType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  NType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const NType last_row =
      MIN(static_cast<NType>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * BLOCK_WARPS + threadIdx.x, 0, &rng);

  typedef cub::WarpScan<FloatType> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[BLOCK_WARPS];
  int warp_id = threadIdx.y;
  int laneid = threadIdx.x;

  while (out_row < last_row) {
    const NType row = in_rows[out_row];
    const EType in_row_start = in_ptr[row];
    const EType out_row_start = out_ptr[out_row];
    const EType cdf_row_start = cdf_ptr[out_row];
    const EType deg = in_ptr[row + 1] - in_row_start;
    const FloatType MIN_THREAD_DATA = static_cast<FloatType>(0.0f);

    if (deg > 0) {
      EType max_iter = (1 + (deg - 1) / WARP_SIZE) * WARP_SIZE;
      // Have the block iterate over segments of items

      FloatType warp_aggregate = static_cast<FloatType>(0.0f);
      for (int idx = laneid; idx < max_iter; idx += WARP_SIZE) {
        FloatType thread_data =
            idx < deg ? prob[in_row_start + idx] : MIN_THREAD_DATA;
        if (laneid == 0) thread_data += warp_aggregate;
        thread_data = max(thread_data, MIN_THREAD_DATA);

        WarpScan(temp_storage[warp_id])
            .InclusiveSum(thread_data, thread_data, warp_aggregate);
        __syncwarp();
        // Store scanned items to cdf array
        if (idx < deg) {
          cdf[cdf_row_start + idx] = thread_data;
        }
      }
      __syncwarp();

      for (int idx = laneid; idx < num_picks; idx += WARP_SIZE) {
        // get random value
        FloatType sum = cdf[cdf_row_start + deg - 1];
        FloatType rand = static_cast<FloatType>(curand_uniform(&rng) * sum);
        // get the offset of the first value within cdf array which is greater
        // than random value.
        EType item = cub::UpperBound<FloatType *, EType, FloatType>(
            &cdf[cdf_row_start], deg, rand);
        item = MIN(item, deg - 1);
        // get in and out index
        const EType in_idx = in_row_start + item;
        const EType out_idx = out_row_start + idx;
        // copy permutation over
        out_rows[out_idx] = static_cast<NType>(row);
        out_cols[out_idx] = in_cols[in_idx];
      }
    }
    out_row += BLOCK_WARPS;
  }
}

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingBiasCUDA(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    torch::Tensor probs, int64_t num_picks, bool replace) {
  DGS_ID_TYPE_SWITCH(indptr.dtype(), EType, {
    DGS_ID_TYPE_SWITCH(indices.dtype(), NType, {
      DGS_VALUE_TYPE_SWITCH(probs.dtype(), FloatType, {
        NType num_rows = seeds.numel();
        torch::Tensor sub_indptr, temp_indptr;
        std::tie(sub_indptr, temp_indptr) = _GetSubAndTempIndptr<NType, EType>(
            seeds, indptr, num_picks, replace);
        thrust::device_ptr<EType> sub_prefix(
            static_cast<EType *>(sub_indptr.data_ptr<EType>()));
        thrust::device_ptr<EType> temp_prefix(
            static_cast<EType *>(temp_indptr.data_ptr<EType>()));
        int nnz = sub_prefix[num_rows];
        int temp_size = temp_prefix[num_rows];

        torch::Tensor coo_row = torch::empty(
            nnz,
            torch::TensorOptions().dtype(seeds.dtype()).device(torch::kCUDA));
        torch::Tensor coo_col = torch::empty(
            nnz,
            torch::TensorOptions().dtype(indices.dtype()).device(torch::kCUDA));

        const uint64_t random_seed = 7777;
        constexpr int WARP_SIZE = 32;
        constexpr int BLOCK_WARPS = BLOCK_SIZE / WARP_SIZE;
        // constexpr int TILE_SIZE = BLOCK_WARPS * 16;
        constexpr int TILE_SIZE = 16;
        // constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
        if (replace) {
          torch::Tensor temp = torch::empty(
              temp_size,
              torch::TensorOptions().dtype(probs.dtype()).device(torch::kCUDA));
          const dim3 block(WARP_SIZE, BLOCK_WARPS);
          const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
          _CSRRowWiseSampleBiasReplaceKernel<NType, EType, FloatType, TILE_SIZE,
                                             BLOCK_WARPS, WARP_SIZE>
              <<<grid, block>>>(
                  random_seed, num_picks, num_rows, seeds.data_ptr<NType>(),
                  indptr.data_ptr<EType>(), indices.data_ptr<NType>(),
                  probs.data_ptr<FloatType>(), sub_indptr.data_ptr<EType>(),
                  temp_indptr.data_ptr<EType>(), temp.data_ptr<FloatType>(),
                  coo_row.data_ptr<NType>(), coo_col.data_ptr<NType>());
        } else {
          const dim3 block(WARP_SIZE, BLOCK_WARPS);
          const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
          _CSRRowWiseSampleBiasKernel<NType, EType, FloatType, TILE_SIZE,
                                      BLOCK_WARPS, WARP_SIZE, 32, 2>
              <<<grid, block>>>(
                  random_seed, num_picks, num_rows, seeds.data_ptr<NType>(),
                  indptr.data_ptr<EType>(), indices.data_ptr<NType>(),
                  probs.data_ptr<FloatType>(), sub_indptr.data_ptr<EType>(),
                  coo_row.data_ptr<NType>(), coo_col.data_ptr<NType>());
        }

        return std::make_tuple(coo_row, coo_col);
      });
    });
  });

  return std::make_tuple(torch::Tensor(), torch::Tensor());
}

}  // namespace cuda
}  // namespace sampling
}  // namespace dgs