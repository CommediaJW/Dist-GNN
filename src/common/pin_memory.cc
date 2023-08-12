#include "cuda_common.h"
#include "dgs_headers.h"
#include "pin_memory.h"

namespace dgs {

void TensorPinMemory(torch::Tensor data) {
  DGS_VALUE_TYPE_SWITCH(data.dtype(), ValueType, {
    void *mem_ptr = reinterpret_cast<void *>(data.data_ptr<ValueType>());
    CUDA_CALL(cudaHostRegister(mem_ptr, data.numel() * data.element_size(),
                               cudaHostRegisterDefault));
  });
}

void TensorUnpinMemory(torch::Tensor data) {
  DGS_VALUE_TYPE_SWITCH(data.dtype(), ValueType, {
    void *mem_ptr = reinterpret_cast<void *>(data.data_ptr<ValueType>());
    CUDA_CALL(cudaHostUnregister(mem_ptr));
  });
}

}  // namespace dgs