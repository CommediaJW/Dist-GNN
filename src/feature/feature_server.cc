#include "../common/cuda_context.h"
#include "../hashmap/cuda/ops.h"
#include "../nccl/nccl_context.h"
#include "cuda/ops.h"
#include "feature_sever.h"

namespace dgs {
namespace feature {

P2PCacheFeatureServer::P2PCacheFeatureServer(torch::Tensor data,
                                             torch::Tensor cache_nids,
                                             int64_t device_id) {
  CHECK_CPU(data);
  CHECK(device_id == nccl::nccl_ctx.local_rank_);
  this->device_id_ = device_id;
  int64_t world_size = nccl::nccl_ctx.world_size_;
  int64_t num_nodes = data.numel();

  // cpu data
  this->cpu_features_ = data;

  // gpu data
  CHECK(cache_nids.numel() > 0);
  if (cache_nids.numel() > 0) {
    gpu_cache_flag_ = true;
    auto cpu_cache_nids = cache_nids.cpu();

    if (!cache_nids.device().is_cuda()) {
      cache_nids = cache_nids.to(
          torch::TensorOptions().device(torch::kCUDA, this->device_id_));
    }

    auto sub_features =
        data.index_select(0, cpu_cache_nids)
            .to(torch::TensorOptions().device(torch::kCUDA, this->device_id_));
    this->gpu_features_ = new cache::TensorP2PServer(sub_features);

    // create hashmap
    std::vector<torch::Tensor> devices_cache_nids;
    int64_t all_cache_nids_num = 0;
    if (world_size > 1) {
      devices_cache_nids = nccl::nccl_ctx.NCCLTensorAllGather_(cache_nids);
      torch::Tensor cached_mask =
          torch::zeros(num_nodes, torch::TensorOptions()
                                      .dtype(torch::kBool)
                                      .device(torch::kCUDA, this->device_id_));
      for (int i = 0; i < world_size; i += 1) {
        cached_mask.index_put_({devices_cache_nids[i]}, true);
      }
      all_cache_nids_num = cached_mask.nonzero().numel();
    } else {
      devices_cache_nids.emplace_back(cache_nids);
      all_cache_nids_num = cache_nids.numel();
    }

    std::tie(this->gpu_hashmap_key_, this->gpu_hashmap_idx_,
             this->gpu_hashmap_devid_) =
        hashmap::cuda::CreateNidsP2PCacheHashMapCUDA(
            devices_cache_nids, all_cache_nids_num, this->device_id_);
  }
}

P2PCacheFeatureServer::~P2PCacheFeatureServer() {
  if (this->gpu_features_ != nullptr) {
    delete this->gpu_features_;
  }
}

torch::Tensor P2PCacheFeatureServer::GetFeatures(torch::Tensor nids) {
  CHECK_CUDA(nids);
  return cuda::GetFeaturesP2PCacheCUDA(
      nids, this->cpu_features_, this->gpu_features_, this->gpu_hashmap_key_,
      this->gpu_hashmap_idx_, this->gpu_hashmap_devid_);
}

torch::Tensor P2PCacheFeatureServer::GetCPUFeature() {
  return this->cpu_features_;
}

torch::Tensor P2PCacheFeatureServer::GetGPUFeature() {
  if (this->gpu_features_ != nullptr) {
    return this->gpu_features_->GetLocalDeviceTensor();
  } else {
    return torch::Tensor();
  }
}

}  // namespace feature
}  // namespace dgs