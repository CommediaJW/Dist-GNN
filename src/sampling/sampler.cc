#include <sys/ipc.h>
#include <sys/shm.h>

#include "../common/cuda_context.h"
#include "../common/pin_memory.h"
#include "../hashmap/cuda/ops.h"
#include "../nccl/nccl_context.h"
#include "cuda/ops.h"
#include "sampler.h"

namespace dgs {
namespace sampling {

NodeClassifictionSampledResult P2PCacheNodeClassificationSampleUniform(
    torch::Tensor seeds, torch::Tensor cpu_indptr, torch::Tensor cpu_indices,
    cache::TensorP2PServer *gpu_indptr, cache::TensorP2PServer *gpu_indices,
    torch::Tensor gpu_hashmap_key, torch::Tensor gpu_hashmap_devid,
    torch::Tensor gpu_hashmap_idx, std::vector<int64_t> fan_out, bool replace) {
  std::vector<NodeClassifictionSampledTensors> results;
  for (int i = fan_out.size() - 1; i >= 0; i -= 1) {
    torch::Tensor coo_row, coo_col;
    std::tie(coo_row, coo_col) = cuda::RowWiseSamplingUniformWithP2PCachingCUDA(
        seeds, gpu_indptr, gpu_indices, gpu_hashmap_key, gpu_hashmap_devid,
        gpu_hashmap_idx, cpu_indptr, cpu_indices, fan_out[i], replace);

    auto relabeled =
        cuda::TensorRelabelCUDA({seeds, coo_col}, {coo_row, coo_col});
    auto frontier = std::get<0>(relabeled);
    auto relabeled_coo_row = std::get<1>(relabeled)[0];
    auto relabeled_coo_col = std::get<1>(relabeled)[1];
    results.emplace_back(
        std::make_tuple(seeds, frontier, relabeled_coo_row, relabeled_coo_col));
    seeds = frontier;
  }
  return NodeClassifictionSampledResult(results);
}

NodeClassifictionSampledResult P2PCacheNodeClassificationSampleBias(
    torch::Tensor seeds, torch::Tensor cpu_indptr, torch::Tensor cpu_indices,
    torch::Tensor cpu_probs, cache::TensorP2PServer *gpu_indptr,
    cache::TensorP2PServer *gpu_indices, cache::TensorP2PServer *gpu_probs,
    torch::Tensor gpu_hashmap_key, torch::Tensor gpu_hashmap_devid,
    torch::Tensor gpu_hashmap_idx, std::vector<int64_t> fan_out, bool replace) {
  std::vector<NodeClassifictionSampledTensors> results;
  for (int i = fan_out.size() - 1; i >= 0; i -= 1) {
    torch::Tensor coo_row, coo_col;
    std::tie(coo_row, coo_col) = cuda::RowWiseSamplingBiasWithP2PCachingCUDA(
        seeds, gpu_indptr, gpu_indices, gpu_probs, gpu_hashmap_key,
        gpu_hashmap_devid, gpu_hashmap_idx, cpu_indptr, cpu_indices, cpu_probs,
        fan_out[i], replace);

    auto relabeled =
        cuda::TensorRelabelCUDA({seeds, coo_col}, {coo_row, coo_col});
    auto frontier = std::get<0>(relabeled);
    auto relabeled_coo_row = std::get<1>(relabeled)[0];
    auto relabeled_coo_col = std::get<1>(relabeled)[1];
    results.emplace_back(
        std::make_tuple(seeds, frontier, relabeled_coo_row, relabeled_coo_col));
    seeds = frontier;
  }
  return NodeClassifictionSampledResult(results);
}

P2PCacheSampler::P2PCacheSampler(torch::Tensor indptr, torch::Tensor indices,
                                 torch::Tensor probs,
                                 torch::Tensor local_gpu_cache_nids,
                                 int64_t device_id) {
  CHECK_CPU(indptr);
  CHECK_CPU(indices);
  CHECK_CPU(probs);

  CHECK(device_id == nccl::nccl_ctx.local_rank_);
  this->device_id_ = device_id;
  int64_t world_size = nccl::nccl_ctx.world_size_;
  int64_t num_nodes = indptr.numel() - 1;

  if (probs.numel() > 0) {
    this->bias_ = true;
  }

  // cpu data
  this->cpu_indptr_ = indptr;
  this->cpu_indices_ = indices;
  if (this->bias_) {
    this->cpu_probs_ = probs;
  }

  // gpu cache
  CHECK(local_gpu_cache_nids.numel() > 0);
  if (local_gpu_cache_nids.numel() > 0) {
    this->gpu_cache_flag_ = true;

    if (!local_gpu_cache_nids.device().is_cuda()) {
      local_gpu_cache_nids = local_gpu_cache_nids.to(
          torch::TensorOptions().device(torch::kCUDA, this->device_id_));
    }

    // extract data and cache
    auto sub_indptr = cuda::ExtractIndptr(local_gpu_cache_nids, indptr);
    this->gpu_indptr_ = new cache::TensorP2PServer(sub_indptr);

    auto sub_indices = cuda::ExtractEdgeData(local_gpu_cache_nids, indptr,
                                             sub_indptr, indices);
    this->gpu_indices_ = new cache::TensorP2PServer(sub_indices);

    if (this->bias_) {
      auto sub_probs = cuda::ExtractEdgeData(local_gpu_cache_nids, indptr,
                                             sub_indptr, probs);
      this->gpu_probs_ = new cache::TensorP2PServer(sub_probs);
    }

    // create hashmap
    std::vector<torch::Tensor> devices_cache_nids;
    int64_t all_cache_nids_num = 0;
    if (world_size > 1) {
      devices_cache_nids =
          nccl::nccl_ctx.NCCLTensorAllGather_(local_gpu_cache_nids);
      torch::Tensor cached_mask =
          torch::zeros(num_nodes, torch::TensorOptions()
                                      .dtype(torch::kBool)
                                      .device(torch::kCUDA, this->device_id_));
      for (int i = 0; i < world_size; i += 1) {
        cached_mask.index_put_({devices_cache_nids[i]}, true);
      }
      all_cache_nids_num = cached_mask.nonzero().numel();
    } else {
      devices_cache_nids.emplace_back(local_gpu_cache_nids);
      all_cache_nids_num = local_gpu_cache_nids.numel();
    }

    std::tie(this->gpu_hashmap_key_, this->gpu_hashmap_idx_,
             this->gpu_hashmap_devid_) =
        hashmap::cuda::CreateNidsP2PCacheHashMapCUDA(
            devices_cache_nids, all_cache_nids_num, this->device_id_);
  }
}

P2PCacheSampler::~P2PCacheSampler() {
  delete this->gpu_indptr_;
  delete this->gpu_indices_;
  if (this->bias_) {
    delete this->gpu_probs_;
  }
}

std::vector<NodeClassifictionSampledTensors>
P2PCacheSampler::NodeClassifictionSample(torch::Tensor seeds,
                                         std::vector<int64_t> fan_out,
                                         bool replace) {
  if (this->bias_) {
    return P2PCacheNodeClassificationSampleBias(
               seeds, this->cpu_indptr_, this->cpu_indices_,
               this->cpu_probs_.value(), this->gpu_indptr_, this->gpu_indices_,
               this->gpu_probs_, this->gpu_hashmap_key_,
               this->gpu_hashmap_devid_, this->gpu_hashmap_idx_, fan_out,
               replace)
        .to_py();
  } else {
    return P2PCacheNodeClassificationSampleUniform(
               seeds, this->cpu_indptr_, this->cpu_indices_, this->gpu_indptr_,
               this->gpu_indices_, this->gpu_hashmap_key_,
               this->gpu_hashmap_devid_, this->gpu_hashmap_idx_, fan_out,
               replace)
        .to_py();
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
P2PCacheSampler::GetCPUStructureTensors() {
  if (this->bias_) {
    return std::make_tuple(this->cpu_indptr_, this->cpu_indices_,
                           this->cpu_probs_.value());
  } else {
    return std::make_tuple(this->cpu_indptr_, this->cpu_indices_,
                           torch::Tensor());
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
P2PCacheSampler::GetLocalCachedStructureTensors() {
  if (this->bias_) {
    return std::make_tuple(this->gpu_indptr_->GetLocalDeviceTensor(),
                           this->gpu_indices_->GetLocalDeviceTensor(),
                           this->gpu_probs_->GetLocalDeviceTensor());
  } else {
    return std::make_tuple(this->gpu_indptr_->GetLocalDeviceTensor(),
                           this->gpu_indices_->GetLocalDeviceTensor(),
                           torch::Tensor());
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
P2PCacheSampler::GetLocalCachedHashTensors() {
  return std::make_tuple(this->gpu_hashmap_key_, this->gpu_hashmap_idx_,
                         this->gpu_hashmap_devid_);
}

}  // namespace sampling
}  // namespace dgs