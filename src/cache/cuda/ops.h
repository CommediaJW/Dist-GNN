#ifndef DGS_CACHE_CUDA_OPS_H_
#define DGS_CACHE_CUDA_OPS_H_

namespace dgs {
namespace cache {
namespace cuda {

torch::Tensor ComputeFrontierHeat(torch::Tensor seeds, torch::Tensor indptr,
                                  torch::Tensor indices,
                                  torch::Tensor seeds_heat, int64_t num_picks,
                                  int64_t indptr_diff);
torch::Tensor ComputeFrontierHeatWithBias(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    torch::Tensor probs, torch::Tensor seeds_heat, int64_t num_picks,
    int64_t indptr_diff);

}  // namespace cuda
}  // namespace cache
}  // namespace dgs

#endif