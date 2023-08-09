#ifndef DGS_NCCL_OPS_H_
#define DGS_NCCL_OPS_H_

#include <torch/script.h>
#include <vector>

namespace dgs {
namespace nccl {

std::vector<torch::Tensor> NCCLTensorAllGather(torch::Tensor local_tensor);

}  // namespace nccl
}  // namespace dgs

#endif