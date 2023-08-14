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
    torch::Tensor gpu_hashmap_idx, torch::Tensor cpu_hashmap_key,
    torch::Tensor cpu_hashmap_idx, std::vector<int64_t> fan_out, bool replace) {
  std::vector<NodeClassifictionSampledTensors> results;
  for (int i = fan_out.size() - 1; i >= 0; i -= 1) {
    torch::Tensor coo_row, coo_col;
    std::tie(coo_row, coo_col) = cuda::RowWiseSamplingUniformWithP2PCachingCUDA(
        seeds, gpu_indptr, gpu_indices, gpu_hashmap_key, gpu_hashmap_devid,
        gpu_hashmap_idx, cpu_indptr, cpu_indices, cpu_hashmap_key,
        cpu_hashmap_idx, fan_out[i], replace);
    std::vector<torch::Tensor> mapping_tensors;
    std::vector<torch::Tensor> requiring_relabel_tensors;
    mapping_tensors.emplace_back(seeds);
    mapping_tensors.emplace_back(coo_col);
    requiring_relabel_tensors.emplace_back(coo_row);
    requiring_relabel_tensors.emplace_back(coo_col);
    auto relabeled =
        cuda::TensorRelabelCUDA(mapping_tensors, requiring_relabel_tensors);
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
    torch::Tensor gpu_hashmap_idx, torch::Tensor cpu_hashmap_key,
    torch::Tensor cpu_hashmap_idx, std::vector<int64_t> fan_out, bool replace) {
  std::vector<NodeClassifictionSampledTensors> results;
  for (int i = fan_out.size() - 1; i >= 0; i -= 1) {
    torch::Tensor coo_row, coo_col;
    std::tie(coo_row, coo_col) = cuda::RowWiseSamplingBiasWithP2PCachingCUDA(
        seeds, gpu_indptr, gpu_indices, gpu_probs, gpu_hashmap_key,
        gpu_hashmap_devid, gpu_hashmap_idx, cpu_indptr, cpu_indices, cpu_probs,
        cpu_hashmap_key, cpu_hashmap_idx, fan_out[i], replace);
    std::vector<torch::Tensor> mapping_tensors;
    std::vector<torch::Tensor> requiring_relabel_tensors;
    mapping_tensors.emplace_back(seeds);
    mapping_tensors.emplace_back(coo_col);
    requiring_relabel_tensors.emplace_back(coo_row);
    requiring_relabel_tensors.emplace_back(coo_col);
    auto relabeled =
        cuda::TensorRelabelCUDA(mapping_tensors, requiring_relabel_tensors);
    auto frontier = std::get<0>(relabeled);
    auto relabeled_coo_row = std::get<1>(relabeled)[0];
    auto relabeled_coo_col = std::get<1>(relabeled)[1];
    results.emplace_back(
        std::make_tuple(seeds, frontier, relabeled_coo_row, relabeled_coo_col));
    seeds = frontier;
  }
  return NodeClassifictionSampledResult(results);
}

std::tuple<torch::Tensor, int> RegisterSharedMemPinnedTensor(
    torch::Tensor tensor, int64_t rank) {
  void *buff = nullptr;
  int64_t size = tensor.element_size() * tensor.numel();
  CHECK(size > 0);

  int shmid;
  if (rank == 0) {
    shmid =
        shmget((key_t)0x6277 + std::rand(), size, IPC_CREAT | IPC_EXCL | 0666);
    SHM_CHECK(shmid);
  }

  CUDA_CALL(cudaHostRegister(&shmid, sizeof(int), cudaHostRegisterDefault));
  NCCL_CALL(ncclBroadcast(&shmid, &shmid, 1, ncclInt, 0,
                          nccl::nccl_ctx.global_comm_,
                          nccl::nccl_ctx.nccl_stream_));
  nccl::nccl_ctx.Barrier_();
  CUDA_CALL(cudaHostUnregister(&shmid));

  buff = (void *)shmat(shmid, nullptr, 0);
  SHM_CHECK(reinterpret_cast<int64_t>(buff));
  CUDA_CALL(cudaHostRegister(buff, size, cudaHostRegisterDefault));

  if (rank == 0) {
    DGS_VALUE_TYPE_SWITCH(tensor.dtype(), ValueType, {
      void *mem_ptr = reinterpret_cast<void *>(tensor.data_ptr<ValueType>());
      CUDA_CALL(cudaMemcpy(buff, mem_ptr, size, cudaMemcpyDefault));
    });
  }

  return std::make_tuple(
      torch::from_blob(buff, tensor.sizes(),
                       torch::TensorOptions().dtype(tensor.dtype())),
      shmid);
}

void FreeSharedMemPinnedTensor(torch::Tensor tensor, int shmid, int64_t rank) {
  void *mem_ptr;
  DGS_VALUE_TYPE_SWITCH(tensor.dtype(), ValueType, {
    mem_ptr = reinterpret_cast<void *>(tensor.data_ptr<ValueType>());
  });
  CUDA_CALL(cudaHostUnregister(mem_ptr));
  int err = shmdt(mem_ptr);
  SHM_CHECK(err);
  nccl::nccl_ctx.Barrier_();
  if (rank == 0) {
    int err = shmctl(shmid, IPC_RMID, nullptr);
    SHM_CHECK(err);
  }
}

P2PCacheSampler::P2PCacheSampler(torch::Tensor indptr, torch::Tensor indices,
                                 torch::Tensor probs, torch::Tensor cache_nids,
                                 torch::Tensor cpu_nids, int64_t device_id) {
  CHECK(device_id == nccl::nccl_ctx.local_rank_);
  this->device_id_ = device_id;
  int64_t world_size = nccl::nccl_ctx.world_size_;
  int64_t num_nodes = indptr.numel() - 1;

  TensorPinMemory(indptr);
  TensorPinMemory(indices);
  if (probs.numel() > 0) {
    TensorPinMemory(probs);
    this->bias_ = true;
  }

  // cpu data
  if (cpu_nids.numel() > 0) {
    this->cpu_cache_flag_ = true;
    if (!cpu_nids.device().is_cuda()) {
      TensorPinMemory(cpu_nids);
    }
    // extract data and register as shared pinned memory
    auto sub_indptr = cuda::ExtractIndptr(cpu_nids, indptr);
    std::tie(this->cpu_indptr_, this->cpu_indptr_shmid_) =
        RegisterSharedMemPinnedTensor(sub_indptr, this->device_id_);

    auto sub_indices =
        cuda::ExtractEdgeData(cpu_nids, indptr, sub_indptr, indices);
    std::tie(this->cpu_indices_, this->cpu_indices_shmid_) =
        RegisterSharedMemPinnedTensor(sub_indices, this->device_id_);

    if (this->bias_) {
      auto sub_probs =
          cuda::ExtractEdgeData(cpu_nids, indptr, sub_indptr, probs);
      std::tie(this->cpu_probs_, this->cpu_probs_shmid_) =
          RegisterSharedMemPinnedTensor(sub_probs, this->device_id_);
    }

    // create hashmap and register as shared pinned memory
    torch::Tensor cpu_hash_key, cpu_hash_idx;
    std::tie(cpu_hash_key, cpu_hash_idx) =
        hashmap::cuda::CreateNidsHashMapCUDA(cpu_nids);
    std::tie(this->cpu_hashmap_key_, this->cpu_hashmap_key_shmid_) =
        RegisterSharedMemPinnedTensor(cpu_hash_key, this->device_id_);
    std::tie(this->cpu_hashmap_idx_, this->cpu_hashmap_idx_shmid_) =
        RegisterSharedMemPinnedTensor(cpu_hash_idx, this->device_id_);

    if (!cpu_nids.device().is_cuda()) {
      TensorUnpinMemory(cpu_nids);
    }
  }

  // gpu cache
  if (cache_nids.numel() > 0) {
    gpu_cache_flag_ = true;
    if (!cache_nids.device().is_cuda()) {
      cache_nids = cache_nids.to(
          torch::TensorOptions().device(torch::kCUDA, this->device_id_));
    }
    // extract data and cache
    auto sub_indptr = cuda::ExtractIndptr(cache_nids, indptr);
    this->gpu_indptr_ = new cache::TensorP2PServer(sub_indptr);
    auto sub_indices =
        cuda::ExtractEdgeData(cache_nids, indptr, sub_indptr, indices);
    this->gpu_indices_ = new cache::TensorP2PServer(sub_indices);
    if (this->bias_) {
      auto sub_probs =
          cuda::ExtractEdgeData(cache_nids, indptr, sub_indptr, probs);
      this->gpu_probs_ = new cache::TensorP2PServer(sub_probs);
    }
    // create hashmap
    std::vector<torch::Tensor> devices_cache_nids;
    int64_t all_cache_nids_num = 0;
    if (world_size > 1) {
      devices_cache_nids = nccl::nccl_ctx.NCCLTensorAllGather_(cache_nids);
      torch::Tensor cached_mask = torch::zeros(
          {
              num_nodes,
          },
          torch::TensorOptions()
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

  TensorUnpinMemory(indptr);
  TensorUnpinMemory(indices);
  if (this->bias_) {
    TensorUnpinMemory(probs);
  }
}

P2PCacheSampler::~P2PCacheSampler() {
  if (this->cpu_cache_flag_) {
    FreeSharedMemPinnedTensor(this->cpu_indptr_, this->cpu_indptr_shmid_,
                              this->device_id_);
    FreeSharedMemPinnedTensor(this->cpu_indices_, this->cpu_indices_shmid_,
                              this->device_id_);
    if (this->bias_) {
      FreeSharedMemPinnedTensor(this->cpu_probs_.value(),
                                this->cpu_probs_shmid_, this->device_id_);
    }
    FreeSharedMemPinnedTensor(this->cpu_hashmap_key_,
                              this->cpu_hashmap_key_shmid_, this->device_id_);
    FreeSharedMemPinnedTensor(this->cpu_hashmap_idx_,
                              this->cpu_hashmap_idx_shmid_, this->device_id_);
  }
  if (this->gpu_cache_flag_) {
    delete this->gpu_indptr_;
    delete this->gpu_indices_;
    if (this->bias_) {
      delete this->gpu_probs_;
    }
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
               this->gpu_hashmap_devid_, this->gpu_hashmap_idx_,
               this->cpu_hashmap_key_, this->cpu_hashmap_idx_, fan_out, replace)
        .to_py();
  } else {
    return P2PCacheNodeClassificationSampleUniform(
               seeds, this->cpu_indptr_, this->cpu_indices_, this->gpu_indptr_,
               this->gpu_indices_, this->gpu_hashmap_key_,
               this->gpu_hashmap_devid_, this->gpu_hashmap_idx_,
               this->cpu_hashmap_key_, this->cpu_hashmap_idx_, fan_out, replace)
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

std::tuple<torch::Tensor, torch::Tensor> P2PCacheSampler::GetCPUHashTensors() {
  return std::make_tuple(this->cpu_hashmap_key_, this->cpu_hashmap_idx_);
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