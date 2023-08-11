#include <torch/script.h>

#include "../../common/cuda/atomic.h"
#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "ops.h"

#define BLOCK_SIZE 128

namespace dgs {
namespace cache {
namespace cuda {

template <typename NType, typename EType, typename ValueType>
__global__ void _ComputeFrotierHeat(int64_t num_items, int64_t num_picks,
                                    int64_t indptr_diff,
                                    const NType *__restrict__ seeds,
                                    const EType *__restrict__ indptr,
                                    const NType *__restrict__ indices,
                                    const ValueType *__restrict__ seeds_heat,
                                    ValueType *__restrict__ frontier_heat) {
  NType curr_item = blockIdx.x * blockDim.x + threadIdx.x;
  if (curr_item < num_items) {
    const NType curr_row = seeds[curr_item];
    const EType row_start = indptr[curr_row] - indptr_diff;
    const EType row_end = indptr[curr_row + 1] - indptr_diff;
    const EType degree = row_end - row_start;
    for (EType i = 0; i < degree; i += 1) {
      ValueType edge_msg = MIN(1, seeds_heat[curr_row] * num_picks / degree);
      common::cuda::AtomicAdd(frontier_heat + indices[row_start + i], edge_msg);
    }
  }
}

torch::Tensor ComputeFrontierHeat(torch::Tensor seeds, torch::Tensor indptr,
                                  torch::Tensor indices,
                                  torch::Tensor seeds_heat, int64_t num_picks,
                                  int64_t indptr_diff) {
  DGS_ID_TYPE_SWITCH(indices.dtype(), NType, {
    DGS_ID_TYPE_SWITCH(indptr.dtype(), EType, {
      DGS_VALUE_TYPE_SWITCH(seeds_heat.dtype(), ValueType, {
        NType num_items = seeds.numel();
        torch::Tensor frontier_heat = torch::zeros_like(seeds_heat);
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + BLOCK_SIZE - 1) / BLOCK_SIZE);
        _ComputeFrotierHeat<NType, EType, ValueType><<<grid, block>>>(
            num_items, num_picks, indptr_diff, seeds.data_ptr<NType>(),
            indptr.data_ptr<EType>(), indices.data_ptr<NType>(),
            seeds_heat.data_ptr<ValueType>(),
            frontier_heat.data_ptr<ValueType>());
        return frontier_heat;
      });
    });
  });
  return torch::Tensor();
}

template <typename NType, typename EType, typename ValueType>
__global__ void _ComputeFrotierHeatWithBias(
    int64_t num_items, int64_t num_picks, int64_t indptr_diff,
    const NType *__restrict__ seeds, const EType *__restrict__ indptr,
    const NType *__restrict__ indices, const ValueType *__restrict__ probs,
    const ValueType *__restrict__ seeds_heat,
    ValueType *__restrict__ frontier_heat) {
  constexpr int CACHE_PER_THREAD = 32;
  __shared__ ValueType probs_cache[BLOCK_SIZE * CACHE_PER_THREAD];
  ValueType *local_cache = probs_cache + (CACHE_PER_THREAD * threadIdx.x);

  NType curr_item = blockIdx.x * blockDim.x + threadIdx.x;
  if (curr_item < num_items) {
    const NType curr_row = seeds[curr_item];
    const EType row_start = indptr[curr_row] - indptr_diff;
    const EType row_end = indptr[curr_row + 1] - indptr_diff;
    const EType degree = row_end - row_start;

    ValueType prob_sum = 0;
    for (EType i = 0; i < degree; i += 1) {
      if (i < CACHE_PER_THREAD) {
        local_cache[i] = probs[row_start + i];
        prob_sum += local_cache[i];
      } else {
        prob_sum += probs[row_start + i];
      }
    }
    // todo use shared memory to accelerate
    for (EType i = 0; i < degree; i += 1) {
      ValueType edge_msg = 0;
      if (i < CACHE_PER_THREAD) {
        edge_msg = MIN(
            1, seeds_heat[curr_row] * num_picks * (local_cache[i] / prob_sum));
      } else {
        edge_msg = MIN(1, seeds_heat[curr_row] * num_picks *
                              (probs[row_start + i] / prob_sum));
      }
      common::cuda::AtomicAdd(frontier_heat + indices[row_start + i], edge_msg);
    }
  }
}

torch::Tensor ComputeFrontierHeatWithBias(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    torch::Tensor probs, torch::Tensor seeds_heat, int64_t num_picks,
    int64_t indptr_diff) {
  DGS_ID_TYPE_SWITCH(indices.dtype(), NType, {
    DGS_ID_TYPE_SWITCH(indptr.dtype(), EType, {
      DGS_VALUE_TYPE_SWITCH(seeds_heat.dtype(), ValueType, {
        NType num_items = seeds.numel() - 1;
        torch::Tensor frontier_heat = torch::zeros_like(seeds_heat);
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + BLOCK_SIZE - 1) / BLOCK_SIZE);
        _ComputeFrotierHeatWithBias<NType, EType, ValueType><<<grid, block>>>(
            num_items, num_picks, indptr_diff, seeds.data_ptr<NType>(),
            indptr.data_ptr<EType>(), indices.data_ptr<NType>(),
            probs.data_ptr<ValueType>(), seeds_heat.data_ptr<ValueType>(),
            frontier_heat.data_ptr<ValueType>());
        return frontier_heat;
      });
    });
  });
  return torch::Tensor();
}

}  // namespace cuda
}  // namespace cache
}  // namespace dgs
