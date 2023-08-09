#include "nccl_ops.h"
#include "../common/cuda_common.h"
#include "../common/dgs_headers.h"
#include "nccl_context.h"

namespace dgs {
namespace nccl {

std::vector<torch::Tensor> NCCLTensorAllGather(torch::Tensor local_tensor) {
  CHECK_CUDA(local_tensor);

  // tensor size allgather
  std::vector<int64_t> tensor_size_list;
  tensor_size_list.resize(world_size);
  tensor_size_list[local_rank] = local_tensor.numel();

  CUDA_CALL(cudaHostRegister(&tensor_size_list[0], sizeof(int64_t) * world_size,
                             cudaHostRegisterDefault));

  CUDA_CALL(cudaStreamSynchronize(nccl_stream));
  NCCL_CALL(ncclGroupStart());
  for (int i = 0; i < world_size; i += 1) {
    if (i != local_rank) {
      NCCL_CALL(ncclSend((char *)&tensor_size_list[local_rank], sizeof(int64_t),
                         ncclChar, i, global_comm, nccl_stream));
      NCCL_CALL(ncclRecv((char *)&tensor_size_list[i], sizeof(int64_t),
                         ncclChar, i, global_comm, nccl_stream));
    }
  }
  NCCL_CALL(ncclGroupEnd());
  CUDA_CALL(cudaStreamSynchronize(nccl_stream));

  CUDA_CALL(cudaHostUnregister(&tensor_size_list[0]));

  // create recv tensor buff
  std::vector<torch::Tensor> recv_tensor_list;
  recv_tensor_list.resize(world_size);
  recv_tensor_list[local_rank] = local_tensor;
  for (int i = 0; i < world_size; i += 1) {
    if (i != local_rank) {
      recv_tensor_list[i] = torch::zeros({tensor_size_list[i]},
                                         torch::TensorOptions()
                                             .dtype(local_tensor.dtype())
                                             .device(torch::kCUDA, local_rank));
    }
  }

  // tensor data allgather
  CUDA_CALL(cudaStreamSynchronize(nccl_stream));
  NCCL_CALL(ncclGroupStart());
  void *send_buff, *recv_buff;
  DGS_VALUE_TYPE_SWITCH(local_tensor.dtype(), ValueType, {
    send_buff = reinterpret_cast<void *>(local_tensor.data_ptr<ValueType>());
  });
  for (int i = 0; i < world_size; i += 1) {
    if (i != local_rank) {
      DGS_VALUE_TYPE_SWITCH(local_tensor.dtype(), ValueType, {
        recv_buff =
            reinterpret_cast<void *>(recv_tensor_list[i].data_ptr<ValueType>());
      });
      NCCL_CALL(ncclSend((char *)send_buff,
                         local_tensor.numel() * local_tensor.element_size(),
                         ncclChar, i, global_comm, nccl_stream));
      NCCL_CALL(ncclRecv(
          (char *)recv_buff,
          recv_tensor_list[i].numel() * recv_tensor_list[i].element_size(),
          ncclChar, i, global_comm, nccl_stream));
    }
  }
  NCCL_CALL(ncclGroupEnd());
  CUDA_CALL(cudaStreamSynchronize(nccl_stream));

  return recv_tensor_list;
}

}  // namespace nccl
}  // namespace dgs