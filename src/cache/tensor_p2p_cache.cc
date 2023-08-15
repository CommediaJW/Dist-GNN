#include <torch/python.h>

#include "../common/cuda_context.h"
#include "../common/dgs_headers.h"
#include "../nccl/nccl_context.h"
#include "tensor_p2p_cache.h"

namespace dgs {
namespace cache {

TensorP2PServer::TensorP2PServer(torch::Tensor data) {
  CHECK_CUDA(data);

  auto device_tensor_shapes = data.sizes();
  local_rank_ = nccl::nccl_ctx.local_rank_;
  num_partitions_ = nccl::nccl_ctx.world_size_;

  device_item_num_ = device_tensor_shapes[0];
  CHECK(device_item_num_ > 0);

  dtype_ = torch::typeMetaToScalarType(data.dtype());
  dtype_size_t_ = data.element_size();

  int64_t stride = 1;
  shapes_.assign(device_tensor_shapes.begin(), device_tensor_shapes.end());
  strides_.resize(device_tensor_shapes.size());
  for (int i = device_tensor_shapes.size() - 1; i >= 0; i--) {
    strides_[i] = stride;
    stride *= device_tensor_shapes[i];
  }
  item_stride_ = strides_[0];

  device_ptrs_.resize(num_partitions_);
  for (int i = 0; i < num_partitions_; i++) {
    device_ptrs_[i] = nullptr;
  }

  device_cached_size_ = device_item_num_ * item_stride_ * dtype_size_t_;
  void *uva_device_ptr =
      CUDAContext::cuda_context.raw_alloc(device_cached_size_);
  CUDA_CALL(cudaMemcpy(uva_device_ptr,
                       reinterpret_cast<char *>(data.storage().data()),
                       device_cached_size_, cudaMemcpyDefault));
  device_ptrs_[local_rank_] = uva_device_ptr;

  // CUDA IPC
  if (num_partitions_ > 1) {
    cudaIpcMemHandle_t ipc_device_mem_handle;
    cudaIpcMemHandle_t ipc_device_mem_handle_recvbuff[num_partitions_];

    CUDA_CALL(
        cudaIpcGetMemHandle(&ipc_device_mem_handle, device_ptrs_[local_rank_]));
    // HostRegister for direct communication via nccl;
    CUDA_CALL(cudaHostRegister(&ipc_device_mem_handle,
                               sizeof(cudaIpcMemHandle_t),
                               cudaHostRegisterDefault));
    CUDA_CALL(cudaHostRegister(ipc_device_mem_handle_recvbuff,
                               sizeof(cudaIpcMemHandle_t) * num_partitions_,
                               cudaHostRegisterDefault));
    NCCL_CALL(ncclAllGather(
        &ipc_device_mem_handle, ipc_device_mem_handle_recvbuff,
        sizeof(cudaIpcMemHandle_t), ncclChar, nccl::nccl_ctx.global_comm_,
        nccl::nccl_ctx.nccl_stream_));

    nccl::nccl_ctx.Barrier_();
    CUDA_CALL(cudaHostUnregister(&ipc_device_mem_handle));
    CUDA_CALL(cudaHostUnregister(ipc_device_mem_handle_recvbuff));

    for (int i = 0; i < static_cast<int>(device_ptrs_.size()); i += 1) {
      if (i != local_rank_)
        CUDA_CALL(cudaIpcOpenMemHandle(&device_ptrs_[i],
                                       ipc_device_mem_handle_recvbuff[i],
                                       cudaIpcMemLazyEnablePeerAccess));
    }
  }

  // get All tensor size
  torch::Tensor device_items = torch::full(
      {1}, device_item_num_,
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
  auto all_device_items = nccl::NCCLTensorAllGather(device_items);
  for (auto i : all_device_items) {
    device_items_.push_back(i.item<int64_t>());
  }

  // create wrapper
  wrapper_device_ptrs_ = reinterpret_cast<void **>(
      CUDAContext::cuda_context.raw_alloc(sizeof(void *) * num_partitions_));
  CUDA_CALL(cudaMemcpy(
      wrapper_device_ptrs_, thrust::raw_pointer_cast(device_ptrs_.data()),
      sizeof(void *) * num_partitions_, cudaMemcpyHostToDevice));
  _CreateWrapperPtr();
}

void TensorP2PServer::_CreateWrapperPtr() {
  DGS_VALUE_TYPE_SWITCH(dtype_, ValueType, {
    tensor_p2p_server_wrapper<ValueType> wrapper(wrapper_device_ptrs_);
    wrapper_p2p_server_ptr_ = CUDAContext::cuda_context.raw_alloc(
        sizeof(tensor_p2p_server_wrapper<ValueType>));
    CUDA_CALL(cudaMemcpy(wrapper_p2p_server_ptr_, &wrapper, sizeof(wrapper),
                         cudaMemcpyHostToDevice));
  });
}

void TensorP2PServer::_Free() {
  CUDAContext::cuda_context.raw_delete(wrapper_device_ptrs_);
  CUDAContext::cuda_context.raw_delete(wrapper_p2p_server_ptr_);
  if (num_partitions_ > 1) {
    for (int i = 0; i < num_partitions_; i += 1) {
      if (local_rank_ != i) {
        CUDA_CALL(cudaIpcCloseMemHandle(device_ptrs_[i]));
      }
    }
  }
  nccl::nccl_ctx.Barrier_();

  CUDAContext::cuda_context.raw_delete(device_ptrs_[local_rank_]);
}

torch::Tensor TensorP2PServer::GetLocalDeviceTensor() {
  torch::Tensor ret = torch::from_blob(
      device_ptrs_[local_rank_], shapes_,
      torch::TensorOptions().dtype(dtype_).device(torch::kCUDA));
  return ret;
}

torch::Tensor TensorP2PServer::GetDeviceTensor(int64_t device_id) {
  torch::Tensor ret = torch::from_blob(
      device_ptrs_[device_id], device_items_[device_id] * item_stride_,
      torch::TensorOptions().dtype(dtype_).device(torch::kCUDA));
  return ret;
}

}  // namespace cache
}  // namespace dgs