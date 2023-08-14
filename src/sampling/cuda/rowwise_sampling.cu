#include <curand_kernel.h>

#include "../../common/cuda/atomic.h"
#include "../../common/cuda/cub_function.h"
#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "../../context/context.h"
#include "ops.h"

#define BLOCK_SIZE 128

namespace dgs {
namespace sampling {
namespace cuda {

template <typename NType, typename EType>
inline torch::Tensor _GetSubIndptr(torch::Tensor seeds, torch::Tensor indptr,
                                   NType num_pick, bool replace) {
  NType num_items = seeds.numel();
  torch::Tensor sub_indptr = torch::empty(
      (num_items + 1),
      torch::TensorOptions().dtype(indptr.dtype()).device(torch::kCUDA));
  thrust::device_ptr<EType> item_prefix(
      static_cast<EType *>(sub_indptr.data_ptr<EType>()));

  using it = thrust::counting_iterator<NType>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [in = seeds.data_ptr<NType>(), in_indptr = indptr.data_ptr<EType>(),
       out = thrust::raw_pointer_cast(item_prefix), replace,
       num_pick] __device__(int i) mutable {
        NType row = in[i];
        EType begin = in_indptr[row];
        EType end = in_indptr[row + 1];
        if (replace) {
          out[i] = (end - begin) == 0 ? 0 : num_pick;
        } else {
          out[i] = MIN(end - begin, num_pick);
        }
      });

  common::cuda::cub_exclusiveSum<EType>(thrust::raw_pointer_cast(item_prefix),
                                        num_items + 1);
  return sub_indptr;
}

template <typename NType, typename EType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
    const uint64_t rand_seed, const NType num_picks, const NType num_rows,
    const NType *__restrict__ const in_rows,
    const EType *__restrict__ const in_ptr,
    const NType *__restrict__ const in_index,
    const EType *__restrict__ const out_ptr, NType *__restrict__ const out_rows,
    NType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  NType out_row = blockIdx.x * TILE_SIZE;
  const NType last_row =
      min(static_cast<NType>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const NType row = in_rows[out_row];
    const EType in_row_start = in_ptr[row];
    const EType deg = in_ptr[row + 1] - in_row_start;
    const EType out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const EType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_cols[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          common::cuda::AtomicMax(out_cols + out_row_start + num, NType(idx));
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const EType perm_idx = out_cols[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
      }
    }
    out_row += 1;
  }
}

template <typename NType, typename EType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
    const uint64_t rand_seed, const NType num_picks, const NType num_rows,
    const NType *__restrict__ const in_rows,
    const EType *__restrict__ const in_ptr,
    const NType *__restrict__ const in_index,
    const EType *__restrict__ const out_ptr, NType *__restrict__ const out_rows,
    NType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  NType out_row = blockIdx.x * TILE_SIZE;
  const NType last_row =
      min(static_cast<NType>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const NType row = in_rows[out_row];
    const EType in_row_start = in_ptr[row];
    const EType out_row_start = out_ptr[out_row];
    const EType deg = in_ptr[row + 1] - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const EType edge = curand(&rng) % deg;
        const EType out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start + edge];
      }
    }
    out_row += 1;
  }
}

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformCUDA(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    int64_t num_picks, bool replace) {
  DGS_ID_TYPE_SWITCH(indptr.dtype(), EType, {
    DGS_ID_TYPE_SWITCH(indices.dtype(), NType, {
      int num_rows = seeds.numel();
      torch::Tensor sub_indptr =
          _GetSubIndptr<NType, EType>(seeds, indptr, num_picks, replace);
      thrust::device_ptr<EType> item_prefix(
          static_cast<EType *>(sub_indptr.data_ptr<EType>()));
      int nnz = item_prefix[num_rows];

      torch::Tensor coo_row = torch::empty(
          nnz,
          torch::TensorOptions().dtype(seeds.dtype()).device(torch::kCUDA));
      torch::Tensor coo_col = torch::empty(
          nnz,
          torch::TensorOptions().dtype(indices.dtype()).device(torch::kCUDA));

      uint64_t random_seed = dgs::ctx::randn_uint64();

      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      if (replace) {
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
        _CSRRowWiseSampleUniformReplaceKernel<NType, EType, TILE_SIZE>
            <<<grid, block>>>(
                random_seed, num_picks, num_rows, seeds.data_ptr<NType>(),
                indptr.data_ptr<EType>(), indices.data_ptr<NType>(),
                sub_indptr.data_ptr<EType>(), coo_row.data_ptr<NType>(),
                coo_col.data_ptr<NType>());
      } else {
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
        _CSRRowWiseSampleUniformKernel<NType, EType, TILE_SIZE>
            <<<grid, block>>>(
                random_seed, num_picks, num_rows, seeds.data_ptr<NType>(),
                indptr.data_ptr<EType>(), indices.data_ptr<NType>(),
                sub_indptr.data_ptr<EType>(), coo_row.data_ptr<NType>(),
                coo_col.data_ptr<NType>());
      }

      return std::make_tuple(coo_row, coo_col);
    });
  });
  return std::make_tuple(torch::Tensor(), torch::Tensor());
}

}  // namespace cuda
}  // namespace sampling
}  // namespace dgs