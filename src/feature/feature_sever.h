#ifndef DGS_FEATURE_SERVER_H_
#define DGS_FEATURE_SERVER_H_

#include <torch/script.h>
#include "../cache/tensor_p2p_cache.h"

namespace dgs {
namespace feature {

class P2PCacheFeatureServer {
 private:
  torch::Tensor cpu_features_;

  cache::TensorP2PServer* gpu_features_ = nullptr;

  torch::Tensor gpu_hashmap_key_;
  torch::Tensor gpu_hashmap_devid_;
  torch::Tensor gpu_hashmap_idx_;

  int64_t device_id_;
  bool gpu_cache_flag_ = false;

 public:
  P2PCacheFeatureServer(torch::Tensor data, torch::Tensor cache_nids,
                        int64_t device_id);
  ~P2PCacheFeatureServer();

  torch::Tensor GetCPUFeature();

  torch::Tensor GetGPUFeature();

  torch::Tensor GetFeatures(torch::Tensor nids);
};

}  // namespace feature

}  // namespace dgs

#endif