#ifndef DGS_PIN_MEMORY_H_
#define DGS_PIN_MEMORY_H_

#include <torch/script.h>

namespace dgs {

void TensorPinMemory(torch::Tensor data);
void TensorUnpinMemory(torch::Tensor data);

}  // namespace dgs

#endif