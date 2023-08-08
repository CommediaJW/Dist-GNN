#ifndef DGS_NCCL_OPS_H_
#define DGS_NCCL_OPS_H_

#include <torch/script.h>
#include <vector>

namespace dgs {
namespace nccl {

std::vector<torch::Tensor> NCCLTensorAlltoAll(
    std::vector<torch::Tensor> input_list);

}  // namespace nccl
}  // namespace dgs

#endif