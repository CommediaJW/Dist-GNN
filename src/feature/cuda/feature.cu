#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "ops.h"

#define BLOCK_SIZE 128

namespace dgs {
namespace feature {
namespace cuda {

template <typename IdType, typename FloatType>
__global__ void _IndexOneDimKernel(const int64_t num_nids,
                                   const IdType *__restrict__ const in_nids,
                                   const FloatType *__restrict__ const data,
                                   FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_node < num_nids) {
    out_data[out_node] = data[in_nids[out_node]];
  }
}

template <typename IdType, typename FloatType, int TILE_SIZE>
__global__ void _IndexKernel(const int64_t num_nids, const int64_t stride,
                             const IdType *__restrict__ const in_nids,
                             const FloatType *__restrict__ const data,
                             FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * TILE_SIZE;
  const int64_t last_node =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_nids);

  while (out_node < last_node) {
    for (int idx = threadIdx.x; idx < stride; idx += BLOCK_SIZE) {
      out_data[out_node * stride + idx] =
          data[in_nids[out_node] * stride + idx];
    }
    out_node += 1;
  }
}

torch::Tensor IndexCUDA(torch::Tensor data, torch::Tensor nid) {
  DGS_ID_TYPE_SWITCH(nid.dtype(), IdType, {
    DGS_VALUE_TYPE_SWITCH(data.dtype(), FloatType, {
      int64_t num_items = nid.numel();

      int64_t stride = 1;
      for (int i = 1; i < data.sizes().size(); i += 1) {
        stride *= data.sizes()[i];
      }
      std::vector<int64_t> buff_shape;
      buff_shape.resize(data.sizes().size());
      buff_shape.assign(data.sizes().begin(), data.sizes().end());
      buff_shape[0] = num_items;
      torch::Tensor data_buff = torch::empty(
          buff_shape,
          torch::TensorOptions().dtype(data.dtype()).device(torch::kCUDA));

      if (stride > 1) {
        constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
        _IndexKernel<IdType, FloatType, TILE_SIZE><<<grid, block>>>(
            num_items, stride, nid.data_ptr<IdType>(),
            data.data_ptr<FloatType>(), data_buff.data_ptr<FloatType>());
      } else if (stride == 1) {
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + BLOCK_SIZE - 1) / BLOCK_SIZE);
        _IndexOneDimKernel<IdType, FloatType><<<grid, block>>>(
            num_items, nid.data_ptr<IdType>(), data.data_ptr<FloatType>(),
            data_buff.data_ptr<FloatType>());
      }

      return data_buff;
    });
  });

  return torch::Tensor();
}

}  // namespace cuda
}  // namespace feature
}  // namespace dgs