#ifndef DGS_SAMPLER_H_
#define DGS_SAMPLER_H_

#include <torch/script.h>

#include "../cache/tensor_p2p_cache.h"

namespace dgs {
namespace sampling {

// seeds, frontiers, coo_row, coo_col
typedef std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    NodeClassifictionSampledTensors;
struct NodeClassifictionSampledResult {
  std::vector<NodeClassifictionSampledTensors> structures_;

  NodeClassifictionSampledResult(
      std::vector<NodeClassifictionSampledTensors> structures)
      : structures_(structures) {}
  NodeClassifictionSampledResult() {}

  void to_cuda(int64_t device_id) {
    for (auto &structure : structures_) {
      std::get<0>(structure) = std::get<0>(structure).to(
          torch::TensorOptions().device(torch::kCUDA, device_id));
      std::get<1>(structure) = std::get<1>(structure).to(
          torch::TensorOptions().device(torch::kCUDA, device_id));
      std::get<2>(structure) = std::get<2>(structure).to(
          torch::TensorOptions().device(torch::kCUDA, device_id));
      std::get<3>(structure) = std::get<3>(structure).to(
          torch::TensorOptions().device(torch::kCUDA, device_id));
    }
  }

  std::vector<NodeClassifictionSampledTensors> to_py() { return structures_; }
};

class P2PCacheSampler {
 private:
  torch::Tensor cpu_indptr_;
  torch::Tensor cpu_indices_;
  torch::optional<torch::Tensor> cpu_probs_;
  int cpu_indptr_shmid_ = -1;
  int cpu_indices_shmid_ = -1;
  int cpu_probs_shmid_ = -1;

  torch::Tensor cpu_hashmap_key_;
  torch::Tensor cpu_hashmap_idx_;
  int cpu_hashmap_key_shmid_ = -1;
  int cpu_hashmap_idx_shmid_ = -1;

  cache::TensorP2PServer *gpu_indptr_;
  cache::TensorP2PServer *gpu_indices_;
  cache::TensorP2PServer *gpu_probs_;

  torch::Tensor gpu_hashmap_key_;
  torch::Tensor gpu_hashmap_devid_;
  torch::Tensor gpu_hashmap_idx_;

  int64_t device_id_;
  bool bias_ = false;
  bool cpu_cache_flag_ = false;
  bool gpu_cache_flag_ = false;

 public:
  P2PCacheSampler() {}
  P2PCacheSampler(torch::Tensor indptr, torch::Tensor indices,
                  torch::Tensor probs, torch::Tensor cache_nids,
                  torch::Tensor cpu_nids, int64_t device_id);
  ~P2PCacheSampler();
  std::vector<NodeClassifictionSampledTensors> NodeClassifictionSample(
      torch::Tensor seeds, std::vector<int64_t> fan_out, bool replace = false);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  GetCPUStructureTensors();
  std::tuple<torch::Tensor, torch::Tensor> GetCPUHashTensors();
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  GetLocalCachedStructureTensors();
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  GetLocalCachedHashTensors();
};

}  // namespace sampling
}  // namespace dgs

#endif