#ifndef DGS_FEATURE_CUDA_OPS_H_
#define DGS_FEATURE_CUDA_OPS_H_

#include <torch/script.h>
#include "../../cache/tensor_p2p_cache.h"

namespace dgs {
namespace feature {
namespace cuda {

torch::Tensor GetFeaturesP2PCacheCUDA(torch::Tensor nids,
                                      torch::Tensor cpu_data,
                                      cache::TensorP2PServer *gpu_data,
                                      torch::Tensor gpu_hashmap_key,
                                      torch::Tensor gpu_hashmap_devid,
                                      torch::Tensor gpu_hashmap_idx);

}  // namespace cuda
}  // namespace feature
}  // namespace dgs
#endif