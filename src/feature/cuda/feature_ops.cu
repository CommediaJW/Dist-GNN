#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "../../hashmap/cuda/hashmap.h"
#include "ops.h"

#define BLOCK_SIZE 128

namespace dgs {
namespace feature {
namespace cuda {

template <typename NType, typename FloatType>
__global__ void _IndexOneDimP2PCacheKernel(
    const int64_t num_nids, const NType *__restrict__ const nids,
    const FloatType *__restrict__ const cpu_data,
    cache::tensor_p2p_server_wrapper<FloatType> *__restrict__ gpu_data,
    NType *__restrict__ const pos_list,
    NType *__restrict__ const gpu_hashmap_idx,
    NType *__restrict__ const gpu_hashmap_devid,
    FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_node < num_nids) {
    NType pos = pos_list[out_node];
    NType output;

    if (pos == -1) {
      output = cpu_data[nids[out_node]];
    } else {
      output = gpu_data->At(gpu_hashmap_devid[pos], gpu_hashmap_idx[pos]);
    }

    out_data[out_node] = output;
  }
}

template <typename NType, typename FloatType, int TILE_SIZE>
__global__ void _IndexP2PCacheKernel(
    const int64_t num_nids, const int64_t stride,
    const NType *__restrict__ const nids,
    const FloatType *__restrict__ const cpu_data,
    cache::tensor_p2p_server_wrapper<FloatType> *__restrict__ gpu_data,
    NType *__restrict__ const pos_list,
    NType *__restrict__ const gpu_hashmap_idx,
    NType *__restrict__ const gpu_hashmap_devid,
    FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * TILE_SIZE;
  const int64_t last_node =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_nids);

  while (out_node < last_node) {
    NType pos = pos_list[out_node];

    if (pos == -1) {
      for (int idx = threadIdx.x; idx < stride; idx += BLOCK_SIZE) {
        out_data[out_node * stride + idx] =
            cpu_data[nids[out_node] * stride + idx];
      }
    } else {
      NType dev_id = gpu_hashmap_devid[pos];
      NType target_index = gpu_hashmap_idx[pos];
      for (int idx = threadIdx.x; idx < stride; idx += BLOCK_SIZE) {
        out_data[out_node * stride + idx] =
            gpu_data->At(dev_id, target_index * stride + idx);
      }
    }

    out_node += 1;
  }
}

torch::Tensor GetFeaturesP2PCacheCUDA(torch::Tensor nids,
                                      torch::Tensor cpu_data,
                                      cache::TensorP2PServer *gpu_data,
                                      torch::Tensor gpu_hashmap_key,
                                      torch::Tensor gpu_hashmap_idx,
                                      torch::Tensor gpu_hashmap_devid) {
  DGS_ID_TYPE_SWITCH(gpu_hashmap_key.dtype(), NType, {
    DGS_VALUE_TYPE_SWITCH(cpu_data.dtype(), FloatType, {
      int64_t num_items = nids.numel();
      NType gpu_hashmap_size = gpu_hashmap_key.numel();

      cache::tensor_p2p_server_wrapper<FloatType> *gpu_data_wrapper_ptr =
          reinterpret_cast<cache::tensor_p2p_server_wrapper<FloatType> *>(
              gpu_data->wrapper_p2p_server_ptr_);

      int64_t stride = gpu_data->item_stride_;

      torch::Tensor pos_list =
          torch::empty({num_items}, torch::TensorOptions()
                                        .dtype(gpu_hashmap_key.dtype())
                                        .device(torch::kCUDA));

      using it = thrust::counting_iterator<int64_t>;
      thrust::for_each(
          thrust::device, it(0), it(num_items),
          [nids = nids.data_ptr<NType>(), pos = pos_list.data_ptr<NType>(),
           gpu_hashmap_key = gpu_hashmap_key.data_ptr<NType>(),
           gpu_hashmap_idx = gpu_hashmap_idx.data_ptr<NType>(),
           gpu_hashmap_size] __device__(int64_t i) mutable {
            NType nid = nids[i];
            hashmap::cuda::Hashmap<NType, NType> gpu_table(
                gpu_hashmap_key, gpu_hashmap_idx, gpu_hashmap_size);
            pos[i] = gpu_table.SearchForPos(nid);
          });

      torch::Tensor ret = torch::empty(
          {num_items, stride},
          torch::TensorOptions().dtype(cpu_data.dtype()).device(torch::kCUDA));

      if (stride == 1) {
        constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + BLOCK_SIZE - 1) / BLOCK_SIZE);
        _IndexOneDimP2PCacheKernel<NType, FloatType><<<grid, block>>>(
            num_items, nids.data_ptr<NType>(), cpu_data.data_ptr<FloatType>(),
            gpu_data_wrapper_ptr, pos_list.data_ptr<NType>(),
            gpu_hashmap_idx.data_ptr<NType>(),
            gpu_hashmap_devid.data_ptr<NType>(), ret.data_ptr<FloatType>());
      } else {
        constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
        _IndexP2PCacheKernel<NType, FloatType, TILE_SIZE><<<grid, block>>>(
            num_items, stride, nids.data_ptr<NType>(),
            cpu_data.data_ptr<FloatType>(), gpu_data_wrapper_ptr,
            pos_list.data_ptr<NType>(), gpu_hashmap_idx.data_ptr<NType>(),
            gpu_hashmap_devid.data_ptr<NType>(), ret.data_ptr<FloatType>());
      }

      return ret;
    });
  });
  return torch::Tensor();
}

template <typename NType, typename FloatType>
__global__ void _IndexOneDimKernel(const int64_t num_nids,
                                   const NType *__restrict__ const in_nids,
                                   const FloatType *__restrict__ const data,
                                   FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_node < num_nids) {
    out_data[out_node] = data[in_nids[out_node]];
  }
}

template <typename NType, typename FloatType, int TILE_SIZE>
__global__ void _IndexKernel(const int64_t num_nids, const int64_t stride,
                             const NType *__restrict__ const in_nids,
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

torch::Tensor GetFeaturesCUDA(torch::Tensor data, torch::Tensor nid) {
  DGS_ID_TYPE_SWITCH(nid.dtype(), NType, {
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
        _IndexKernel<NType, FloatType, TILE_SIZE><<<grid, block>>>(
            num_items, stride, nid.data_ptr<NType>(),
            data.data_ptr<FloatType>(), data_buff.data_ptr<FloatType>());
      } else if (stride == 1) {
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + BLOCK_SIZE - 1) / BLOCK_SIZE);
        _IndexOneDimKernel<NType, FloatType><<<grid, block>>>(
            num_items, nid.data_ptr<NType>(), data.data_ptr<FloatType>(),
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