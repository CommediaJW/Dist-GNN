#include "pin_memory.h"
#include "cuda_common.h"
#include "dgs_headers.h"

namespace dgs {

void TensorPinMemory(torch::Tensor data) {
  if (data.is_pinned()) return;
  void *mem_ptr = data.storage().data();
  CUDA_CALL(cudaHostRegister(mem_ptr, data.numel() * data.element_size(),
                             cudaHostRegisterDefault));
}

void TensorUnpinMemory(torch::Tensor data) {
  if (data.is_pinned()) return;
  
  void *mem_ptr = data.storage().data();
  CUDA_CALL(cudaHostUnregister(mem_ptr));
}

}  // namespace dgs