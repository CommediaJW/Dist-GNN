#ifndef DGS_SAMPLING_CUDA_OPS_H_
#define DGS_SAMPLING_CUDA_OPS_H_

#include <torch/script.h>
#include <vector>
#include "../../cache/tensor_p2p_cache.h"

namespace dgs {
namespace sampling {
namespace cuda {

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformCUDA(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    int64_t num_picks, bool replace);

std::tuple<torch::Tensor, torch::Tensor>
RowWiseSamplingUniformWithP2PCachingCUDA(
    torch::Tensor seeds, cache::TensorP2PServer *gpu_indptr,
    cache::TensorP2PServer *gpu_indices, torch::Tensor gpu_hashmap_key,
    torch::Tensor gpu_hashmap_devid, torch::Tensor gpu_hashmap_idx,
    torch::Tensor cpu_indptr, torch::Tensor cpu_indices, int64_t num_picks,
    bool replace);

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingBiasCUDA(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    torch::Tensor probs, int64_t num_picks, bool replace);

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingBiasWithP2PCachingCUDA(
    torch::Tensor seeds, cache::TensorP2PServer *gpu_indptr,
    cache::TensorP2PServer *gpu_indices, cache::TensorP2PServer *gpu_probs,
    torch::Tensor gpu_hashmap_key, torch::Tensor gpu_hashmap_devid,
    torch::Tensor gpu_hashmap_idx, torch::Tensor cpu_indptr,
    torch::Tensor cpu_indices, torch::Tensor cpu_probs,
    torch::Tensor cpu_hashmap_key, torch::Tensor cpu_hashmap_idx,
    int64_t num_picks, bool replace);

std::tuple<torch::Tensor, std::vector<torch::Tensor>> TensorRelabelCUDA(
    std::vector<torch::Tensor> mapping_tensors,
    std::vector<torch::Tensor> requiring_relabel_tensors);

torch::Tensor ExtractIndptr(torch::Tensor nids, torch::Tensor indptr);

torch::Tensor ExtractEdgeData(torch::Tensor nids, torch::Tensor indptr,
                              torch::Tensor sub_indptr,
                              torch::Tensor edge_data);

}  // namespace cuda
}  // namespace sampling
}  // namespace dgs
#endif