#include <curand_kernel.h>
#include <torch/script.h>

#include "../../cache/tensor_p2p_cache.h"
#include "../../common/cuda/cub_function.h"
#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "../../hashmap/cuda/hashmap.h"
#include "ops.h"
#include "warpselect/WarpSelect.cuh"

#define BLOCK_SIZE 128

namespace dgs {
namespace sampling {
namespace cuda {

template <typename NType, typename EType, typename FloatType, int TILE_SIZE,
          int BLOCK_WARPS, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void _CSRRowWiseSampleBiasWithP2PCachingKernel(
    const uint64_t rand_seed, const NType num_picks, const NType num_rows,
    const NType gpu_hash_size, const NType *__restrict__ const in_rows,
    const EType *__restrict__ const row_begin,
    const EType *__restrict__ const row_end,
    cache::tensor_p2p_server_wrapper<NType> *__restrict__ in_gpu_index,
    const NType *__restrict__ const in_cpu_index,
    cache::tensor_p2p_server_wrapper<FloatType> *__restrict__ in_gpu_probs,
    const FloatType *__restrict__ const in_cpu_probs,
    const EType *__restrict__ const out_ptr,
    NType *__restrict__ const gpu_hashmap_key,
    NType *__restrict__ const gpu_hashmap_devid,
    NType *__restrict__ const out_rows, NType *__restrict__ const out_cols) {
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

  hashmap::cuda::Hashmap<NType, NType> table(gpu_hashmap_key, gpu_hashmap_devid,
                                             gpu_hash_size);

  while (out_row < last_row) {
    const NType row = in_rows[out_row];
    const EType in_row_start = row_begin[out_row];
    const EType deg = row_end[out_row] - in_row_start;
    const EType out_row_start = out_ptr[out_row];

    const NType pos = table.SearchForPos(row);

    // A-Res value needs to be calculated only if deg is greater than num_picks
    // in weighted rowwise sampling without replacement
    if (deg > num_picks) {
      heap.reset();
      int limit = warpselect::roundDown(deg, WARP_SIZE);
      EType i = laneid;

      for (; i < limit; i += WARP_SIZE) {
        FloatType item_prob = 0;
        if (pos != -1) {
          item_prob =
              in_gpu_probs->At(gpu_hashmap_devid[pos], in_row_start + i);
        } else {
          item_prob = in_cpu_probs[in_row_start + i];
        }
        FloatType ares_prob = __powf(curand_uniform(&rng), 1.0f / item_prob);
        heap.add(ares_prob, i);
      }

      if (i < deg) {
        FloatType item_prob = 0;
        if (pos != -1) {
          item_prob =
              in_gpu_probs->At(gpu_hashmap_devid[pos], in_row_start + i);
        } else {
          item_prob = in_cpu_probs[in_row_start + i];
        }
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
        if (pos != -1) {
          out_cols[out_idx] = in_gpu_index->At(gpu_hashmap_devid[pos], in_idx);
        } else {
          out_cols[out_idx] = in_cpu_index[in_idx];
        }
      }
    } else {
      for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        // get in and out index
        const EType out_idx = out_row_start + idx;
        const EType in_idx = in_row_start + idx;
        // copy permutation over
        out_rows[out_idx] = static_cast<NType>(row);
        if (pos != -1) {
          out_cols[out_idx] = in_gpu_index->At(gpu_hashmap_devid[pos], in_idx);
        } else {
          out_cols[out_idx] = in_cpu_index[in_idx];
        }
      }
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename NType, typename EType, typename FloatType, int TILE_SIZE,
          int BLOCK_WARPS, int WARP_SIZE>
__global__ void _CSRRowWiseSampleBiasReplaceWithP2PCachingKernel(
    const uint64_t rand_seed, const NType num_picks, const NType num_rows,
    const NType gpu_hash_size, const NType *__restrict__ const in_rows,
    const EType *__restrict__ const row_begin,
    const EType *__restrict__ const row_end,
    cache::tensor_p2p_server_wrapper<NType> *__restrict__ in_gpu_index,
    const NType *__restrict__ const in_cpu_index,
    cache::tensor_p2p_server_wrapper<FloatType> *__restrict__ in_gpu_probs,
    const FloatType *__restrict__ const in_cpu_probs,
    const EType *__restrict__ const out_ptr,
    const EType *__restrict__ const cdf_ptr, FloatType *__restrict__ const cdf,
    NType *__restrict__ const gpu_hashmap_key,
    NType *__restrict__ const gpu_hashmap_devid,
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

  hashmap::cuda::Hashmap<NType, NType> table(gpu_hashmap_key, gpu_hashmap_devid,
                                             gpu_hash_size);

  while (out_row < last_row) {
    const NType row = in_rows[out_row];
    const EType in_row_start = row_begin[out_row];
    const EType out_row_start = out_ptr[out_row];
    const EType cdf_row_start = cdf_ptr[out_row];
    const EType deg = row_end[out_row] - in_row_start;
    const FloatType MIN_THREAD_DATA = static_cast<FloatType>(0.0f);

    const NType pos = table.SearchForPos(row);

    if (deg > 0) {
      EType max_iter = (1 + (deg - 1) / WARP_SIZE) * WARP_SIZE;
      // Have the block iterate over segments of items

      FloatType warp_aggregate = static_cast<FloatType>(0.0f);
      for (int idx = laneid; idx < max_iter; idx += WARP_SIZE) {
        FloatType thread_data = 0;
        if (pos != -1) {
          thread_data =
              in_gpu_probs->At(gpu_hashmap_devid[pos], in_row_start + idx);
        } else {
          thread_data = in_cpu_probs[in_row_start + idx];
        }
        thread_data = idx < deg ? thread_data : MIN_THREAD_DATA;
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
        if (pos != -1) {
          out_cols[out_idx] = in_gpu_index->At(gpu_hashmap_devid[pos], in_idx);
        } else {
          out_cols[out_idx] = in_cpu_index[in_idx];
        }
      }
    }
    out_row += BLOCK_WARPS;
  }
}

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingBiasWithP2PCachingCUDA(
    torch::Tensor seeds, cache::TensorP2PServer *gpu_indptr,
    cache::TensorP2PServer *gpu_indices, cache::TensorP2PServer *gpu_probs,
    torch::Tensor gpu_hashmap_key, torch::Tensor gpu_hashmap_devid,
    torch::Tensor gpu_hashmap_idx, torch::Tensor cpu_indptr,
    torch::Tensor cpu_indices, torch::Tensor cpu_probs,
    torch::Tensor cpu_hashmap_key, torch::Tensor cpu_hashmap_idx,
    int64_t num_picks, bool replace) {
  DGS_ID_TYPE_SWITCH(cpu_indptr.dtype(), EType, {
    DGS_ID_TYPE_SWITCH(cpu_indices.dtype(), NType, {
      DGS_VALUE_TYPE_SWITCH(cpu_probs.dtype(), FloatType, {
        NType num_items = seeds.numel();
        NType gpu_hashmap_size = gpu_hashmap_key.numel();
        NType cpu_hashmap_size = cpu_hashmap_key.numel();
        cache::tensor_p2p_server_wrapper<EType> *gpu_indptr_wrapper_ptr =
            reinterpret_cast<cache::tensor_p2p_server_wrapper<EType> *>(
                gpu_indptr->wrapper_p2p_server_ptr_);
        cache::tensor_p2p_server_wrapper<NType> *gpu_indices_wrapper_ptr =
            reinterpret_cast<cache::tensor_p2p_server_wrapper<NType> *>(
                gpu_indices->wrapper_p2p_server_ptr_);
        cache::tensor_p2p_server_wrapper<FloatType> *gpu_probs_wrapper_ptr =
            reinterpret_cast<cache::tensor_p2p_server_wrapper<FloatType> *>(
                gpu_probs->wrapper_p2p_server_ptr_);

        torch::Tensor row_begin =
            torch::empty({num_items}, torch::TensorOptions()
                                          .dtype(cpu_indptr.dtype())
                                          .device(torch::kCUDA));
        torch::Tensor row_end =
            torch::empty({num_items}, torch::TensorOptions()
                                          .dtype(cpu_indptr.dtype())
                                          .device(torch::kCUDA));
        torch::Tensor sub_indptr =
            torch::empty({num_items + 1}, torch::TensorOptions()
                                              .dtype(cpu_indptr.dtype())
                                              .device(torch::kCUDA));
        torch::Tensor temp_indptr =
            torch::empty({num_items + 1}, torch::TensorOptions()
                                              .dtype(cpu_indptr.dtype())
                                              .device(torch::kCUDA));

        // get sub indptr
        using it = thrust::counting_iterator<NType>;
        thrust::for_each(
            thrust::device, it(0), it(num_items),
            [seeds = seeds.data_ptr<NType>(),
             gpu_indptr = gpu_indptr_wrapper_ptr,
             gpu_hashmap_key = gpu_hashmap_key.data_ptr<NType>(),
             gpu_hashmap_devid = gpu_hashmap_devid.data_ptr<NType>(),
             gpu_hashmap_idx = gpu_hashmap_idx.data_ptr<NType>(),
             cpu_indptr = cpu_indptr.data_ptr<EType>(),
             cpu_hashmap_key = cpu_hashmap_key.data_ptr<NType>(),
             cpu_hashmap_idx = cpu_hashmap_idx.data_ptr<NType>(),
             sub_indptr = sub_indptr.data_ptr<EType>(),
             temp_indptr = temp_indptr.data_ptr<EType>(),
             row_begin = row_begin.data_ptr<EType>(),
             row_end = row_end.data_ptr<EType>(), replace, num_picks,
             gpu_hashmap_size, cpu_hashmap_size,
             num_items] __device__(int64_t i) mutable {
              NType nid = seeds[i];
              hashmap::cuda::Hashmap<NType, NType> gpu_table(
                  gpu_hashmap_key, gpu_hashmap_idx, gpu_hashmap_size);
              hashmap::cuda::Hashmap<NType, NType> cpu_table(
                  cpu_hashmap_key, cpu_hashmap_idx, cpu_hashmap_size);
              const NType pos = gpu_table.SearchForPos(nid);
              if (pos != -1) {
                row_begin[i] = gpu_indptr->At(gpu_hashmap_devid[pos],
                                              gpu_hashmap_idx[pos]);
                row_end[i] = gpu_indptr->At(gpu_hashmap_devid[pos],
                                            gpu_hashmap_idx[pos] + 1);
              } else {
                const NType cpu_pos = cpu_table.SearchForPos(nid);
                const NType cpu_idx = cpu_hashmap_idx[cpu_pos];
                row_begin[i] = cpu_indptr[cpu_idx];
                row_end[i] = cpu_indptr[cpu_idx + 1];
              }
              EType deg = row_end[i] - row_begin[i];
              if (replace) {
                sub_indptr[i] = deg == 0 ? 0 : num_picks;
                temp_indptr[i] = deg;
              } else {
                sub_indptr[i] = MIN(deg, num_picks);
                temp_indptr[i] = deg > num_picks ? deg : 0;
              }
              if (i == num_items - 1) {
                sub_indptr[num_items] = 0;
                temp_indptr[num_items] = 0;
              }
            });
        common::cuda::cub_exclusiveSum<EType>(sub_indptr.data_ptr<EType>(),
                                              num_items + 1);
        common::cuda::cub_exclusiveSum<EType>(temp_indptr.data_ptr<EType>(),
                                              num_items + 1);
        thrust::device_ptr<EType> item_prefix(
            static_cast<EType *>(sub_indptr.data_ptr<EType>()));
        thrust::device_ptr<EType> temp_prefix(
            static_cast<EType *>(temp_indptr.data_ptr<EType>()));
        int nnz = item_prefix[num_items];
        int temp_size = temp_prefix[num_items];

        torch::Tensor coo_row = torch::empty(
            {nnz},
            torch::TensorOptions().dtype(seeds.dtype()).device(torch::kCUDA));
        torch::Tensor coo_col =
            torch::empty({nnz}, torch::TensorOptions()
                                    .dtype(cpu_indices.dtype())
                                    .device(torch::kCUDA));

        const uint64_t random_seed = 7777;
        constexpr int WARP_SIZE = 32;
        constexpr int BLOCK_WARPS = BLOCK_SIZE / WARP_SIZE;
        constexpr int TILE_SIZE = 16;
        if (replace) {
          torch::Tensor temp =
              torch::empty({temp_size}, torch::TensorOptions()
                                            .dtype(cpu_probs.dtype())
                                            .device(torch::kCUDA));
          const dim3 block(WARP_SIZE, BLOCK_WARPS);
          const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
          _CSRRowWiseSampleBiasReplaceWithP2PCachingKernel<
              NType, EType, FloatType, TILE_SIZE, BLOCK_WARPS, WARP_SIZE>
              <<<grid, block>>>(
                  random_seed, num_picks, num_items, gpu_hashmap_size,
                  seeds.data_ptr<NType>(), row_begin.data_ptr<EType>(),
                  row_end.data_ptr<EType>(), gpu_indices_wrapper_ptr,
                  cpu_indices.data_ptr<NType>(), gpu_probs_wrapper_ptr,
                  cpu_probs.data_ptr<FloatType>(), sub_indptr.data_ptr<EType>(),
                  temp_indptr.data_ptr<EType>(), temp.data_ptr<FloatType>(),
                  gpu_hashmap_key.data_ptr<NType>(),
                  gpu_hashmap_devid.data_ptr<NType>(),
                  coo_row.data_ptr<NType>(), coo_col.data_ptr<NType>());
        } else {
          const dim3 block(WARP_SIZE, BLOCK_WARPS);
          const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
          _CSRRowWiseSampleBiasWithP2PCachingKernel<
              NType, EType, FloatType, TILE_SIZE, BLOCK_WARPS, WARP_SIZE, 32, 2>
              <<<grid, block>>>(
                  random_seed, num_picks, num_items, gpu_hashmap_size,
                  seeds.data_ptr<NType>(), row_begin.data_ptr<EType>(),
                  row_end.data_ptr<EType>(), gpu_indices_wrapper_ptr,
                  cpu_indices.data_ptr<NType>(), gpu_probs_wrapper_ptr,
                  cpu_probs.data_ptr<FloatType>(), sub_indptr.data_ptr<EType>(),
                  gpu_hashmap_key.data_ptr<NType>(),
                  gpu_hashmap_devid.data_ptr<NType>(),
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