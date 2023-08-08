#include "../../common/cuda/cub_function.h"
#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "ops.h"

#define BLOCK_SIZE 128

namespace dgs {
namespace sampling {
namespace cuda {

torch::Tensor ExtractIndptr(torch::Tensor nids, torch::Tensor indptr) {
  DGS_ID_TYPE_SWITCH(nids.dtype(), NType, {
    DGS_ID_TYPE_SWITCH(indptr.dtype(), EType, {
      NType num_items = nids.numel();
      torch::Tensor sub_indptr = torch::empty(
          {
              num_items + 1,
          },
          torch::TensorOptions().dtype(indptr.dtype()).device(torch::kCUDA));
      using it = thrust::counting_iterator<NType>;
      thrust::for_each(
          thrust::device, it(0), it(num_items),
          [in_nids = nids.data_ptr<NType>(),
           in_indptr = indptr.data_ptr<EType>(),
           out_indptr =
               sub_indptr.data_ptr<EType>()] __device__(int64_t i) mutable {
            NType nid = in_nids[i];
            EType begin = in_indptr[nid];
            EType end = in_indptr[nid + 1];
            out_indptr[i] = end - begin;
          });
      thrust::device_ptr<EType> item_prefix(
          static_cast<EType *>(sub_indptr.data_ptr<EType>()));
      common::cuda::cub_exclusiveSum<EType>(
          thrust::raw_pointer_cast(item_prefix), num_items + 1);

      return sub_indptr;
    });
  });
  return torch::Tensor();
}

template <typename NType, typename EType, typename ValueType, int TILE_SIZE>
__global__ void _ExtractEdgeDataKernel(
    const NType num_items, const NType *__restrict__ const nids,
    const EType *__restrict__ const indptr,
    const ValueType *__restrict__ const edge_data,
    const EType *__restrict__ const sub_indptr,
    ValueType *__restrict__ const sub_edge_data) {
  assert(blockDim.x == BLOCK_SIZE);

  NType curr_item = blockIdx.x * TILE_SIZE;
  const NType last_item =
      min(static_cast<NType>(blockIdx.x + 1) * TILE_SIZE, num_items);

  while (curr_item < last_item) {
    const NType nid = nids[curr_item];
    const EType in_start = indptr[nid];
    const EType degree = indptr[nid + 1] - in_start;
    const EType out_start = sub_indptr[curr_item];

    for (int idx = threadIdx.x; idx < degree; idx += BLOCK_SIZE) {
      sub_edge_data[out_start + idx] = edge_data[in_start + idx];
    }

    curr_item += 1;
  }
}

torch::Tensor ExtractEdgeData(torch::Tensor nids, torch::Tensor indptr,
                              torch::Tensor sub_indptr,
                              torch::Tensor edge_data) {
  DGS_ID_TYPE_SWITCH(nids.dtype(), NType, {
    DGS_ID_TYPE_SWITCH(indptr.dtype(), EType, {
      DGS_VALUE_TYPE_SWITCH(edge_data.dtype(), ValueType, {
        NType num_items = nids.numel();
        thrust::device_ptr<EType> item_prefix(
            static_cast<EType *>(sub_indptr.data_ptr<EType>()));
        torch::Tensor sub_edge_data = torch::empty(
            {
                item_prefix[num_items],
            },
            torch::TensorOptions()
                .dtype(edge_data.dtype())
                .device(torch::kCUDA));
        constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
        _ExtractEdgeDataKernel<NType, EType, ValueType, TILE_SIZE>
            <<<grid, block>>>(
                num_items, nids.data_ptr<NType>(), indptr.data_ptr<EType>(),
                edge_data.data_ptr<ValueType>(), sub_indptr.data_ptr<EType>(),
                sub_edge_data.data_ptr<ValueType>());

        return sub_edge_data;
      });
    });
  });
  return torch::Tensor();
}

}  // namespace cuda
}  // namespace sampling
}  // namespace dgs