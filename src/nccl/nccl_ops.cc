#include "nccl_ops.h"
#include "../common/cuda_common.h"
#include "../common/dgs_headers.h"
#include "nccl_context.h"

namespace dgs {
namespace nccl {

std::vector<torch::Tensor> NCCLTensorAlltoAll(
    std::vector<torch::Tensor> input_list) {
  // tensor size all-to-all
  std::vector<int *> send_size_list, recv_size_list;
  send_size_list.resize(world_size);
  recv_size_list.resize(world_size);
  for (int i = 0; i < world_size; i += 1) {
    if (i != local_rank) {
      int size = input_list[i].size(0);
      CUDA_CALL(cudaMalloc((void **)&send_size_list[i], sizeof(int)));
      CUDA_CALL(cudaMalloc((void **)&recv_size_list[i], sizeof(int)));
      CUDA_CALL(cudaMemcpy(send_size_list[i], &size, sizeof(int),
                           cudaMemcpyHostToDevice));
    }
  }
  CUDA_CALL(cudaStreamSynchronize(nccl_stream));
  NCCL_CALL(ncclGroupStart());
  for (int i = 0; i < world_size; i += 1) {
    if (i != local_rank) {
      NCCL_CALL(ncclSend((char *)send_size_list[i], sizeof(int), ncclChar, i,
                         global_comm, nccl_stream));
      NCCL_CALL(ncclRecv((char *)recv_size_list[i], sizeof(int), ncclChar, i,
                         global_comm, nccl_stream));
    }
  }
  NCCL_CALL(ncclGroupEnd());
  CUDA_CALL(cudaStreamSynchronize(nccl_stream));
  for (int i = 0; i < world_size; i += 1) {
    if (i != local_rank) {
      CUDA_CALL(cudaFree(send_size_list[i]));
    }
  }

  // create recv tensor buff
  std::vector<torch::Tensor> output_list;
  output_list.resize(world_size);
  for (int i = 0; i < world_size; i += 1) {
    int size = 0;
    if (i == local_rank) {
      output_list[i] = input_list[i];
    } else {
      CUDA_CALL(cudaMemcpy(&size, recv_size_list[i], sizeof(int),
                           cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaFree(recv_size_list[i]));
      output_list[i] = torch::zeros({size}, torch::TensorOptions()
                                                .dtype(input_list[0].dtype())
                                                .device(torch::kCUDA));
    }
  }

  // tensor all-to-all
  CUDA_CALL(cudaStreamSynchronize(nccl_stream));
  NCCL_CALL(ncclGroupStart());
  for (int i = 0; i < world_size; i += 1) {
    if (i != local_rank) {
      int64_t *send_buff = input_list[i].data_ptr<int64_t>();
      int64_t send_size = input_list[i].element_size() * input_list[i].size(0);
      int64_t *recv_buff = output_list[i].data_ptr<int64_t>();
      int64_t recv_size =
          output_list[i].element_size() * output_list[i].size(0);
      NCCL_CALL(ncclSend((char *)send_buff, send_size, ncclChar, i, global_comm,
                         nccl_stream));
      NCCL_CALL(ncclRecv((char *)recv_buff, recv_size, ncclChar, i, global_comm,
                         nccl_stream));
    } else {
      output_list[i] = input_list[i];
    }
  }
  NCCL_CALL(ncclGroupEnd());
  CUDA_CALL(cudaStreamSynchronize(nccl_stream));

  return output_list;
}

}  // namespace nccl
}  // namespace dgs