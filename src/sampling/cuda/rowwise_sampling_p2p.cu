#include <curand_kernel.h>

#include "../../cache/tensor_p2p_cache.h"
#include "../../common/cuda/atomic.h"
#include "../../common/cuda/cub_function.h"
#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "../../context/context.h"
#include "../../hashmap/cuda/hashmap.h"
#include "../../nccl/nccl_context.h"
#include "ops.h"

#define BLOCK_SIZE 128

namespace dgs {
namespace sampling {
namespace cuda {

template <typename NType, typename EType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformWithP2PCachingKernel(
    const uint64_t rand_seed, const NType num_picks, const NType num_rows,
    const NType gpu_hash_size, const NType *__restrict__ const in_rows,
    const EType *__restrict__ const row_begin,
    const EType *__restrict__ const row_end,
    cache::tensor_p2p_server_wrapper<NType> *__restrict__ in_gpu_index,
    const NType *__restrict__ const in_cpu_index,
    const EType *__restrict__ const out_ptr,
    NType *__restrict__ const gpu_hashmap_key,
    NType *__restrict__ const gpu_hashmap_devid,
    NType *__restrict__ const out_rows, NType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  NType out_row = blockIdx.x * TILE_SIZE;
  const NType last_row =
      min(static_cast<NType>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  hashmap::cuda::Hashmap<NType, NType> table(gpu_hashmap_key, gpu_hashmap_devid,
                                             gpu_hash_size);

  while (out_row < last_row) {
    const NType row = in_rows[out_row];
    const EType in_row_start = row_begin[out_row];
    const EType deg = row_end[out_row] - in_row_start;
    const EType out_row_start = out_ptr[out_row];

    const NType pos = table.SearchForPos(row);

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const EType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        if (pos != -1) {
          out_cols[out_row_start + idx] =
              in_gpu_index->At(gpu_hashmap_devid[pos], in_idx);
        } else {
          out_cols[out_row_start + idx] = in_cpu_index[in_idx];
        }
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
        if (pos != -1) {
          out_cols[out_row_start + idx] =
              in_gpu_index->At(gpu_hashmap_devid[pos], perm_idx);
        } else {
          out_cols[out_row_start + idx] = in_cpu_index[perm_idx];
        }
      }
    }
    out_row += 1;
  }
}

template <typename NType, typename EType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceWithP2PCachingKernel(
    const uint64_t rand_seed, const NType num_picks, const NType num_rows,
    const NType gpu_hash_size, const NType *__restrict__ const in_rows,
    const EType *__restrict__ const row_begin,
    const EType *__restrict__ const row_end,
    cache::tensor_p2p_server_wrapper<NType> *__restrict__ in_gpu_index,
    const NType *__restrict__ const in_cpu_index,
    const EType *__restrict__ const out_ptr,
    NType *__restrict__ const gpu_hashmap_key,
    NType *__restrict__ const gpu_hashmap_devid,
    NType *__restrict__ const out_rows, NType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  NType out_row = blockIdx.x * TILE_SIZE;
  const NType last_row =
      min(static_cast<NType>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  hashmap::cuda::Hashmap<NType, NType> table(gpu_hashmap_key, gpu_hashmap_devid,
                                             gpu_hash_size);

  while (out_row < last_row) {
    const NType row = in_rows[out_row];
    const EType in_row_start = row_begin[out_row];
    const EType deg = row_end[out_row] - in_row_start;
    const EType out_row_start = out_ptr[out_row];

    const NType pos = table.SearchForPos(row);

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const EType edge = curand(&rng) % deg;
        const EType out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        if (pos != -1) {
          out_cols[out_idx] =
              in_gpu_index->At(gpu_hashmap_devid[pos], in_row_start + edge);
        } else {
          out_cols[out_idx] = in_cpu_index[in_row_start + edge];
        }
      }
    }
    out_row += 1;
  }
}

std::tuple<torch::Tensor, torch::Tensor>
RowWiseSamplingUniformWithP2PCachingCUDA(
    torch::Tensor seeds, cache::TensorP2PServer *gpu_indptr,
    cache::TensorP2PServer *gpu_indices, torch::Tensor gpu_hashmap_key,
    torch::Tensor gpu_hashmap_devid, torch::Tensor gpu_hashmap_idx,
    torch::Tensor cpu_indptr, torch::Tensor cpu_indices, int64_t num_picks,
    bool replace) {
  DGS_ID_TYPE_SWITCH(cpu_indptr.dtype(), EType, {
    DGS_ID_TYPE_SWITCH(cpu_indices.dtype(), NType, {
      NType num_items = seeds.numel();
      NType gpu_hashmap_size = gpu_hashmap_key.numel();

      cache::tensor_p2p_server_wrapper<EType> *gpu_indptr_wrapper_ptr =
          reinterpret_cast<cache::tensor_p2p_server_wrapper<EType> *>(
              gpu_indptr->wrapper_p2p_server_ptr_);
      cache::tensor_p2p_server_wrapper<NType> *gpu_indices_wrapper_ptr =
          reinterpret_cast<cache::tensor_p2p_server_wrapper<NType> *>(
              gpu_indices->wrapper_p2p_server_ptr_);

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

      // get sub indptr
      using it = thrust::counting_iterator<NType>;
      thrust::for_each(
          thrust::device, it(0), it(num_items),
          [seeds = seeds.data_ptr<NType>(),
           cpu_indptr = cpu_indptr.data_ptr<EType>(),
           gpu_indptr = gpu_indptr_wrapper_ptr,
           gpu_hashmap_key = gpu_hashmap_key.data_ptr<NType>(),
           gpu_hashmap_devid = gpu_hashmap_devid.data_ptr<NType>(),
           gpu_hashmap_idx = gpu_hashmap_idx.data_ptr<NType>(),
           sub_indptr = sub_indptr.data_ptr<EType>(),
           row_begin = row_begin.data_ptr<EType>(),
           row_end = row_end.data_ptr<EType>(), gpu_hashmap_size, replace,
           num_picks] __device__(int64_t i) mutable {
            NType nid = seeds[i];
            hashmap::cuda::Hashmap<NType, NType> gpu_table(
                gpu_hashmap_key, gpu_hashmap_idx, gpu_hashmap_size);

            const NType pos = gpu_table.SearchForPos(nid);
            if (pos != -1) {
              row_begin[i] =
                  gpu_indptr->At(gpu_hashmap_devid[pos], gpu_hashmap_idx[pos]);
              row_end[i] = gpu_indptr->At(gpu_hashmap_devid[pos],
                                          gpu_hashmap_idx[pos] + 1);
            } else {
              row_begin[i] = cpu_indptr[nid];
              row_end[i] = cpu_indptr[nid + 1];
            }
            if (replace) {
              sub_indptr[i] = (row_end[i] - row_begin[i]) == 0 ? 0 : num_picks;
            } else {
              sub_indptr[i] = MIN(row_end[i] - row_begin[i], num_picks);
            }
          });
      common::cuda::cub_exclusiveSum<EType>(sub_indptr.data_ptr<EType>(),
                                            num_items + 1);
      thrust::device_ptr<EType> item_prefix(
          static_cast<EType *>(sub_indptr.data_ptr<EType>()));
      int nnz = item_prefix[num_items];

      torch::Tensor coo_row = torch::empty(
          {nnz},
          torch::TensorOptions().dtype(seeds.dtype()).device(torch::kCUDA));
      torch::Tensor coo_col =
          torch::empty({nnz}, torch::TensorOptions()
                                  .dtype(cpu_indices.dtype())
                                  .device(torch::kCUDA));

      uint64_t random_seed = dgs::ctx::randn_uint64();
      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      if (replace) {
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
        _CSRRowWiseSampleUniformReplaceWithP2PCachingKernel<NType, EType,
                                                            TILE_SIZE>
            <<<grid, block>>>(
                random_seed, num_picks, num_items, gpu_hashmap_size,
                seeds.data_ptr<NType>(), row_begin.data_ptr<EType>(),
                row_end.data_ptr<EType>(), gpu_indices_wrapper_ptr,
                cpu_indices.data_ptr<NType>(), sub_indptr.data_ptr<EType>(),
                gpu_hashmap_key.data_ptr<NType>(),
                gpu_hashmap_devid.data_ptr<NType>(), coo_row.data_ptr<NType>(),
                coo_col.data_ptr<NType>());

      } else {
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
        _CSRRowWiseSampleUniformWithP2PCachingKernel<NType, EType, TILE_SIZE>
            <<<grid, block>>>(
                random_seed, num_picks, num_items, gpu_hashmap_size,
                seeds.data_ptr<NType>(), row_begin.data_ptr<EType>(),
                row_end.data_ptr<EType>(), gpu_indices_wrapper_ptr,
                cpu_indices.data_ptr<NType>(), sub_indptr.data_ptr<EType>(),
                gpu_hashmap_key.data_ptr<NType>(),
                gpu_hashmap_devid.data_ptr<NType>(), coo_row.data_ptr<NType>(),
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