#ifndef DGS_HASHMAP_CUDA_OPS_H_
#define DGS_HASHMAP_CUDA_OPS_H_

#include <torch/script.h>
#include <vector>

namespace dgs {
namespace hashmap {
namespace cuda {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CreateNidsP2PCacheHashMapCUDA(std::vector<torch::Tensor> devices_cache_nids,
                              int64_t all_devices_cache_nids_num,
                              int64_t local_device_id);

}  // namespace cuda
}  // namespace hashmap
}  // namespace dgs
#endif