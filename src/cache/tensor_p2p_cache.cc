#include <torch/python.h>

#include "../common/cuda_context.h"
#include "../common/dgs_headers.h"
#include "../nccl/nccl_context.h"
#include "tensor_p2p_cache.h"

namespace dgs {
namespace cache {

inline void *_getTensorVoidDataPtr(torch::Tensor data) {
  return data.storage().data();
}

inline size_t _getTensorTypeSizeOf(torch::Dtype type) {
  if (type == torch::kInt32) {
    return sizeof(int32_t);
  } else if (type == torch::kInt64) {
    return sizeof(int64_t);
  } else if (type == torch::kFloat) {
    return sizeof(float);
  } else if (type == torch::kDouble) {
    return sizeof(double);
  } else if (type == torch::kBool) {
    return sizeof(bool);
  } else {
    fprintf(stderr, "Error in _getTensorSizeInByte!\n");
    exit(-1);
  }
}

TensorP2PServer::TensorP2PServer(std::vector<int64_t> device_tensor_shapes,
                                 torch::ScalarType dtype) {
  local_rank_ = nccl::local_rank;
  num_partitions_ = nccl::world_size;

  device_item_num_ = device_tensor_shapes[0];
  CHECK(device_item_num_ > 0);

  dtype_ = dtype;
  dtype_size_t_ = _getTensorTypeSizeOf(dtype_);

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

  // Malloc GPU memory for device cached elements
  device_cached_size_ = device_item_num_ * item_stride_ * dtype_size_t_;
  void *uva_device_ptr =
      CUDAContext::cuda_context.raw_alloc(device_cached_size_);
  CUDA_CALL(cudaMemset(uva_device_ptr, -1, device_cached_size_));
  device_ptrs_[local_rank_] = uva_device_ptr;

  // CUDA IPC
  if (num_partitions_ > 1) {
    cudaIpcMemHandle_t ipc_device_mem_handle;
    cudaIpcMemHandle_t ipc_device_mem_handle_recvbuff[num_partitions_];

    CUDA_CALL(cudaIpcGetMemHandle(&ipc_device_mem_handle, uva_device_ptr));
    // HostRegister for direct communication via nccl;
    CUDA_CALL(cudaHostRegister(&ipc_device_mem_handle,
                               sizeof(cudaIpcMemHandle_t),
                               cudaHostRegisterDefault));
    CUDA_CALL(cudaHostRegister(ipc_device_mem_handle_recvbuff,
                               sizeof(cudaIpcMemHandle_t) * num_partitions_,
                               cudaHostRegisterDefault));
    NCCL_CALL(ncclAllGather(&ipc_device_mem_handle,
                            ipc_device_mem_handle_recvbuff,
                            sizeof(cudaIpcMemHandle_t), ncclChar,
                            nccl::global_comm, nccl::nccl_stream));
    nccl::_Barrier();
    CUDA_CALL(cudaHostUnregister(&ipc_device_mem_handle));
    CUDA_CALL(cudaHostUnregister(ipc_device_mem_handle_recvbuff));

    for (int i = 0; i < static_cast<int>(device_ptrs_.size()); i += 1) {
      if (i != local_rank_)
        CUDA_CALL(cudaIpcOpenMemHandle(&device_ptrs_[i],
                                       ipc_device_mem_handle_recvbuff[i],
                                       cudaIpcMemLazyEnablePeerAccess));
    }
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
  nccl::_Barrier();
  CUDAContext::cuda_context.raw_delete(device_ptrs_[local_rank_]);
}

torch::Tensor TensorP2PServer::GetLocalDeviceTensor() {
  torch::Tensor ret = torch::from_blob(
      device_ptrs_[local_rank_], shapes_,
      torch::TensorOptions().dtype(dtype_).device(torch::kCUDA));
  return ret;
}

torch::Tensor TensorP2PServer::GetDeviceTensor(int64_t device_id,
                                               std::vector<int64_t> shapes) {
  torch::Tensor ret = torch::from_blob(
      device_ptrs_[device_id], shapes,
      torch::TensorOptions().dtype(dtype_).device(torch::kCUDA));
  return ret;
}

void TensorP2PServer::LoadDeviceTensorData(torch::Tensor device_tensor_data) {
  CHECK(static_cast<size_t>(device_tensor_data.dim()) == shapes_.size());
  CHECK(device_tensor_data.dtype() == dtype_);

  for (uint64_t i = 0; i < shapes_.size(); i += 1) {
    CHECK(device_tensor_data.size(i) == shapes_[i]);
    CHECK(device_tensor_data.stride(i) == strides_[i]);
  }

  CUDA_CALL(cudaMemcpy(
      device_ptrs_[local_rank_],
      reinterpret_cast<char *>(_getTensorVoidDataPtr(device_tensor_data)),
      device_cached_size_, cudaMemcpyDefault));
}
}  // namespace cache
}  // namespace dgs