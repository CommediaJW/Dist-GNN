#ifndef DGS_TENSOR_P2P_CACHE_H_
#define DGS_TENSOR_P2P_CACHE_H_

#include <pybind11/pybind11.h>
#include <torch/script.h>
#include "../common/cuda_common.h"

namespace dgs {
namespace cache {

template <typename ValueType>
struct tensor_p2p_server_wrapper {
  void **device_ptrs_;

  __host__ tensor_p2p_server_wrapper(void **device_ptrs) {
    device_ptrs_ = device_ptrs;
  }

  ~tensor_p2p_server_wrapper(){};

  __device__ inline ValueType At(int64_t device_id, int64_t index) {
    return reinterpret_cast<ValueType *>(device_ptrs_[device_id])[index];
  }
};

class TensorP2PServer {
 public:
  TensorP2PServer(std::vector<int64_t> device_tensor_shapes,
                  pybind11::object dtype);
  ~TensorP2PServer() { _Free(); }

  void LoadDeviceTensorData(torch::Tensor device_tensor_data);

  torch::Tensor GetLocalDeviceTensor();
  torch::Tensor GetDeviceTensor(int64_t device_id, std::vector<int64_t> shapes);

  torch::ScalarType dtype_;
  int64_t dtype_size_t_;

  int64_t num_partitions_;

  int64_t local_rank_;

  std::vector<int64_t> strides_;
  std::vector<int64_t> shapes_;
  int64_t item_stride_;

  int64_t device_item_num_;
  int64_t device_cached_size_;

  thrust::host_vector<void *> device_ptrs_;
  void **wrapper_device_ptrs_ = nullptr;
  void *wrapper_p2p_server_ptr_ = nullptr;

 private:
  void _CreateWrapperPtr();
  void _Free();
};
}  // namespace cache
}  // namespace dgs

#endif