#ifndef DGS_FEATURE_CUDA_OPS_H_
#define DGS_FEATURE_CUDA_OPS_H_

#include <torch/script.h>

namespace dgs {
namespace feature {
namespace cuda {

torch::Tensor IndexCUDA(torch::Tensor data, torch::Tensor nid);

}  // namespace cuda
}  // namespace feature
}  // namespace dgs
#endif