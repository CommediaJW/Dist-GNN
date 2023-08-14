#ifndef DGS_NCCL_CONTEXT_H_
#define DGS_NCCL_CONTEXT_H_

#include <nccl.h>

namespace dgs {
namespace nccl {

class NCCLContext {
 public:
  NCCLContext() {
    nccl_stream_ = nullptr;
    device_buffer_ = nullptr;
  }
  NCCLContext(const NCCLContext&) = delete;
  NCCLContext& operator=(const NCCLContext&) = delete;

  void SetNCCL_(int64_t nrank, std::vector<int64_t> unique_id_array,
                int64_t rank);
  void Barrier_();
  std::vector<torch::Tensor> NCCLTensorAllGather_(torch::Tensor local_tensor);

  ncclComm_t global_comm_;
  int local_rank_;
  int world_size_;
  cudaStream_t nccl_stream_;
  float* device_buffer_;
};

typedef struct {
  ncclUniqueId nccl_unique_id_;
} DGSUniqueId;

static NCCLContext nccl_ctx;

std::vector<int64_t> GetUniqueId();
void SetNCCL(int64_t nranks, std::vector<int64_t> unique_id_array,
             int64_t rank);

std::vector<torch::Tensor> NCCLTensorAllGather(torch::Tensor local_tensor);
int GetLocalRank();
int GetWorldSize();

}  // namespace nccl
}  // namespace dgs

#endif