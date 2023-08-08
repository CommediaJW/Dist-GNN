#include <sys/ipc.h>
#include <sys/shm.h>

#include "../common/cuda_context.h"
#include "../common/pin_memory.h"
#include "../hashmap/cuda/ops.h"
#include "../nccl/nccl_context.h"
#include "../nccl/nccl_ops.h"
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

std::tuple<torch::Tensor, int> RegisterSharedMemTensor(torch::Tensor tensor,
                                                       int64_t rank) {
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
  NCCL_CALL(ncclBroadcast(&shmid, &shmid, 1, ncclInt, 0, nccl::global_comm,
                          nccl::nccl_stream));
  nccl::_Barrier();
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

P2PCacheSampler::P2PCacheSampler(torch::Tensor indptr, torch::Tensor indices,
                                 torch::Tensor probs, torch::Tensor cache_nids,
                                 torch::Tensor cpu_nids, int64_t device_id) {
  CHECK(device_id == nccl::local_rank);
  this->device_id_ = device_id;
  int64_t world_size = nccl::world_size;
  int64_t num_nodes = indptr.numel() - 1;

  TensorPinMemory(indptr);
  TensorPinMemory(indices);
  if (probs.numel() > 0) {
    TensorPinMemory(probs);
    this->bias_ = true;
  }

  // cpu data
  if (cpu_nids.numel() > 0) {
    if (!cpu_nids.device().is_cuda()) {
      TensorPinMemory(cpu_nids);
    }
    // extract data and register as shared pinned memory
    auto sub_indptr = cuda::ExtractIndptr(cpu_nids, indptr);
    std::tie(this->cpu_indptr_, this->cpu_indptr_shmid_) =
        RegisterSharedMemTensor(sub_indptr, this->device_id_);

    auto sub_indices =
        cuda::ExtractEdgeData(cpu_nids, indptr, sub_indptr, indices);
    std::tie(this->cpu_indices_, this->cpu_indices_shmid_) =
        RegisterSharedMemTensor(sub_indices, this->device_id_);

    if (this->bias_) {
      auto sub_probs =
          cuda::ExtractEdgeData(cpu_nids, indptr, sub_indptr, probs);
      std::tie(this->cpu_probs_, this->cpu_probs_shmid_) =
          RegisterSharedMemTensor(sub_probs, this->device_id_);
    }

    // create hashmap and register as shared pinned memory
    torch::Tensor cpu_hash_key, cpu_hash_idx;
    std::tie(cpu_hash_key, cpu_hash_idx) =
        hashmap::cuda::CreateNidsHashMapCUDA(cpu_nids);
    std::tie(this->cpu_hashmap_key_, this->cpu_hashmap_key_shmid_) =
        RegisterSharedMemTensor(cpu_hash_key, this->device_id_);
    std::tie(this->cpu_hashmap_idx_, this->cpu_hashmap_idx_shmid_) =
        RegisterSharedMemTensor(cpu_hash_idx, this->device_id_);

    if (!cpu_nids.device().is_cuda()) {
      TensorUnpinMemory(cpu_nids);
    }
  }

  // gpu cache
  if (cache_nids.numel() > 0) {
    if (!cache_nids.device().is_cuda()) {
      cache_nids = cache_nids.to(
          torch::TensorOptions().device(torch::kCUDA, this->device_id_));
    }
    // extract data and cache
    auto sub_indptr = cuda::ExtractIndptr(cache_nids, indptr);
    auto sub_indptr_shape = sub_indptr.sizes();
    this->gpu_indptr_ = cache::TensorP2PServer(
        std::vector<int64_t>(sub_indptr_shape.begin(), sub_indptr_shape.end()),
        torch::typeMetaToScalarType(sub_indptr.dtype()));
    this->gpu_indptr_.LoadDeviceTensorData(sub_indptr);

    auto sub_indices =
        cuda::ExtractEdgeData(cache_nids, indptr, sub_indptr, indices);
    auto sub_indices_shape = sub_indices.sizes();
    this->gpu_indices_ = cache::TensorP2PServer(
        std::vector<int64_t>(sub_indices_shape.begin(),
                             sub_indices_shape.end()),
        torch::typeMetaToScalarType(sub_indices.dtype()));
    this->gpu_indices_.LoadDeviceTensorData(sub_indices);

    if (this->bias_) {
      auto sub_probs =
          cuda::ExtractEdgeData(cache_nids, indptr, sub_indptr, probs);
      auto sub_probs_shape = sub_probs.sizes();
      this->gpu_probs_ = cache::TensorP2PServer(
          std::vector<int64_t>(sub_probs_shape.begin(), sub_probs_shape.end()),
          torch::typeMetaToScalarType(sub_probs.dtype()));
      this->gpu_probs_.LoadDeviceTensorData(sub_probs);
    }

    // create hashmap
    std::vector<torch::Tensor> devices_cache_nids;
    int64_t all_cache_nids_num = 0;
    if (world_size > 1) {
      std::vector<torch::Tensor> send_cache_nids;
      send_cache_nids.resize(world_size);
      send_cache_nids[this->device_id_] = cache_nids;
      devices_cache_nids = nccl::NCCLTensorAlltoAll(send_cache_nids);
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

NodeClassifictionSampledResult P2PCacheSampler::NodeClassifictionSample(
    torch::Tensor seeds, std::vector<int64_t> fan_out, bool replace) {
  if (this->bias_) {
    return P2PCacheNodeClassificationSampleBias(
               seeds, this->cpu_indptr_, this->cpu_indices_,
               this->cpu_probs_.value(), &this->gpu_indptr_,
               &this->gpu_indices_, &this->gpu_probs_, this->gpu_hashmap_key_,
               this->gpu_hashmap_devid_, this->gpu_hashmap_idx_,
               this->cpu_hashmap_key_, this->cpu_hashmap_idx_, fan_out, replace)
        .to_py();
  } else {
    return P2PCacheNodeClassificationSampleUniform(
               seeds, this->cpu_indptr_, this->cpu_indices_, &this->gpu_indptr_,
               &this->gpu_indices_, this->gpu_hashmap_key_,
               this->gpu_hashmap_devid_, this->gpu_hashmap_idx_,
               this->cpu_hashmap_key_, this->cpu_hashmap_idx_, fan_out, replace)
        .to_py();
  }
}

}  // namespace sampling
}  // namespace dgs