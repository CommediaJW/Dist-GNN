#include <torch/script.h>
#include <vector>

#include "../common/cuda_common.h"
#include "../common/dgs_headers.h"
#include "nccl_context.h"

namespace dgs {
namespace nccl {

NCCLContext nccl_ctx;

std::vector<int64_t> GetUniqueId() {
  std::vector<int64_t> unique_id(AlignUp(sizeof(DGSUniqueId), sizeof(int64_t)));
  DGSUniqueId *ptr = (DGSUniqueId *)unique_id.data();
  ncclGetUniqueId(&ptr->nccl_unique_id_);
  return unique_id;
};

void SetNCCL(int64_t nranks, std::vector<int64_t> unique_id_array,
             int64_t rank) {
  nccl_ctx.SetNCCL_(nranks, unique_id_array, rank);
}

std::vector<torch::Tensor> NCCLTensorAllGather(torch::Tensor local_tensor) {
  return nccl_ctx.NCCLTensorAllGather_(local_tensor);
}

void Barrier() { nccl_ctx.Barrier_(); }

int GetLocalRank() { return nccl_ctx.local_rank_; }
int GetWorldSize() { return nccl_ctx.world_size_; }
bool IsInitialized() { return nccl_ctx.initialized_; }

void NCCLContext::SetNCCL_(int64_t nranks, std::vector<int64_t> unique_id_array,
                           int64_t rank) {
  DGSUniqueId unique_id;
  memcpy(&unique_id, unique_id_array.data(), sizeof(unique_id));
  NCCL_CALL(
      ncclCommInitRank(&global_comm_, nranks, unique_id.nccl_unique_id_, rank));
  NCCL_CALL(ncclCommUserRank(global_comm_, &local_rank_));
  NCCL_CALL(ncclCommCount(global_comm_, &world_size_));
  nccl_stream_ = 0;
  CUDA_CALL(cudaMalloc(&device_buffer_, sizeof(float)));

  initialized_ = true;
}

void NCCLContext::Barrier_() {
  NCCL_CALL(ncclAllReduce(device_buffer_, device_buffer_, 1, ncclFloat, ncclSum,
                          global_comm_, nccl_stream_));
  CUDA_CALL(cudaStreamSynchronize(nccl_stream_));
}

std::vector<torch::Tensor> NCCLContext::NCCLTensorAllGather_(
    torch::Tensor local_tensor) {
  CHECK_CUDA(local_tensor);

  // step1: tensor size allgather
  std::vector<int64_t> tensor_size_list;
  tensor_size_list.resize(world_size_);
  tensor_size_list[local_rank_] = local_tensor.numel();

  CUDA_CALL(cudaHostRegister(&tensor_size_list[0],
                             sizeof(int64_t) * world_size_,
                             cudaHostRegisterDefault));

  CUDA_CALL(cudaStreamSynchronize(nccl_stream_));
  NCCL_CALL(ncclGroupStart());
  for (int i = 0; i < world_size_; i += 1) {
    if (i != local_rank_) {
      NCCL_CALL(ncclSend((char *)&tensor_size_list[local_rank_],
                         sizeof(int64_t), ncclChar, i, global_comm_,
                         nccl_stream_));
      NCCL_CALL(ncclRecv((char *)&tensor_size_list[i], sizeof(int64_t),
                         ncclChar, i, global_comm_, nccl_stream_));
    }
  }
  NCCL_CALL(ncclGroupEnd());
  CUDA_CALL(cudaStreamSynchronize(nccl_stream_));

  CUDA_CALL(cudaHostUnregister(&tensor_size_list[0]));

  // create recv tensor buff
  std::vector<torch::Tensor> recv_tensor_list;
  recv_tensor_list.resize(world_size_);
  recv_tensor_list[local_rank_] = local_tensor;
  for (int i = 0; i < world_size_; i += 1) {
    if (i != local_rank_) {
      recv_tensor_list[i] =
          torch::empty({tensor_size_list[i]}, local_tensor.options());
    }
  }

  // step2: tensor data allgather
  CUDA_CALL(cudaStreamSynchronize(nccl_stream_));
  NCCL_CALL(ncclGroupStart());
  void *send_buff = local_tensor.storage().data();
  for (int i = 0; i < world_size_; i += 1) {
    if (i != local_rank_) {
      void *recv_buff = recv_tensor_list[i].storage().data();
      NCCL_CALL(ncclSend((char *)send_buff,
                         local_tensor.numel() * local_tensor.element_size(),
                         ncclChar, i, global_comm_, nccl_stream_));
      NCCL_CALL(ncclRecv(
          (char *)recv_buff,
          recv_tensor_list[i].numel() * recv_tensor_list[i].element_size(),
          ncclChar, i, global_comm_, nccl_stream_));
    }
  }
  NCCL_CALL(ncclGroupEnd());
  CUDA_CALL(cudaStreamSynchronize(nccl_stream_));

  return recv_tensor_list;
}

}  // namespace nccl

}  // namespace dgs