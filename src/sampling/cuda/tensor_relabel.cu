#include "../../common/cuda/atomic.h"
#include "../../common/cuda/cub_function.h"
#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "ops.h"

namespace dgs {
namespace sampling {
namespace cuda {

template <typename NType>
struct RelabelHashmap {
  __device__ inline RelabelHashmap(NType* __restrict__ Kptr,
                                   NType* __restrict__ Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Update(NType key, NType value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    NType prev = common::cuda::AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = common::cuda::AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    common::cuda::AtomicMin(vptr + pos, value);
  }

  __device__ inline NType SearchForPos(NType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return pos;
      }
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline NType SearchForValue(NType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return vptr[pos];
      };
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline uint32_t hash(int32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(int64_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint64_t key) { return key & (capacity - 1); }

  NType kEmptyKey{-1};
  NType* kptr;
  NType* vptr;
  uint32_t capacity{0};
};

inline int _UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}

template <typename NType>
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Unique(
    torch::Tensor total_tensor) {
  int num_items = total_tensor.numel();
  int dir_size = _UpPower(num_items);

  NType MAX = std::numeric_limits<NType>::max();
  torch::Tensor key_tensor = torch::full(
      {
          dir_size,
      },
      -1, total_tensor.options());
  torch::Tensor index_tensor = torch::full(
      {
          dir_size,
      },
      MAX, total_tensor.options());

  // insert
  using it = thrust::counting_iterator<NType>;
  thrust::for_each(it(0), it(num_items),
                   [key = key_tensor.data_ptr<NType>(),
                    index = index_tensor.data_ptr<NType>(),
                    in = total_tensor.data_ptr<NType>(), num_items,
                    dir_size] __device__(NType i) mutable {
                     RelabelHashmap<NType> table(key, index, dir_size);
                     table.Update(in[i], i);
                   });

  // prefix sum
  torch::Tensor item_prefix_tensor =
      torch::empty(num_items + 1, total_tensor.options());
  thrust::device_ptr<NType> item_prefix(
      static_cast<NType*>(item_prefix_tensor.data_ptr<NType>()));
  thrust::for_each(it(0), it(num_items),
                   [key = key_tensor.data_ptr<NType>(),
                    index = index_tensor.data_ptr<NType>(),
                    in = total_tensor.data_ptr<NType>(),
                    count = thrust::raw_pointer_cast(item_prefix), num_items,
                    dir_size] __device__(NType i) mutable {
                     RelabelHashmap<NType> table(key, index, dir_size);
                     count[i] = table.SearchForValue(in[i]) == i ? 1 : 0;
                   });
  common::cuda::cub_exclusiveSum<NType>(thrust::raw_pointer_cast(item_prefix),
                                        num_items + 1);

  // unique
  int tot = item_prefix[num_items];
  torch::Tensor unique_tensor = torch::empty(
      {
          tot,
      },
      total_tensor.options());

  torch::Tensor value_tensor = torch::full(
      {
          dir_size,
      },
      -1, total_tensor.options());

  thrust::for_each(it(0), it(num_items),
                   [key = key_tensor.data_ptr<NType>(),
                    index = index_tensor.data_ptr<NType>(),
                    in = total_tensor.data_ptr<NType>(),
                    prefix = thrust::raw_pointer_cast(item_prefix),
                    unique = unique_tensor.data_ptr<NType>(),
                    cache_value = value_tensor.data_ptr<NType>(), num_items,
                    dir_size] __device__(NType i) mutable {
                     RelabelHashmap<NType> table(key, index, dir_size);
                     NType pos = table.SearchForPos(in[i]);
                     if (index[pos] == i) {
                       unique[prefix[i]] = in[i];
                       cache_value[pos] = prefix[i];
                     }
                   });

  return {unique_tensor, key_tensor, value_tensor};
}

template <typename NType>
inline torch::Tensor Relabel(torch::Tensor total_tensor,
                             torch::Tensor key_tensor,
                             torch::Tensor value_tensor) {
  int num_items = total_tensor.numel();
  using it = thrust::counting_iterator<NType>;
  torch::Tensor relabel_tensor = torch::zeros_like(total_tensor);
  int dir_size = key_tensor.numel();

  thrust::for_each(it(0), it(num_items),
                   [key = key_tensor.data_ptr<NType>(),
                    value = value_tensor.data_ptr<NType>(),
                    in = total_tensor.data_ptr<NType>(),
                    out = relabel_tensor.data_ptr<NType>(),
                    dir_size] __device__(NType i) mutable {
                     RelabelHashmap<NType> table(key, value, dir_size);
                     out[i] = table.SearchForValue(in[i]);
                   });
  return relabel_tensor;
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> TensorRelabelCUDA(
    std::vector<torch::Tensor> mapping_tensors,
    std::vector<torch::Tensor> requiring_relabel_tensors) {
  std::vector<int64_t> split_sizes;
  for (auto d : requiring_relabel_tensors) {
    split_sizes.push_back(d.numel());
  }

  torch::Tensor total_tensor = torch::cat(requiring_relabel_tensors, 0);

  torch::Tensor unique_tensor, key_tensor, value_tenosr;
  torch::Tensor reindex_tensor;

  DGS_ID_TYPE_SWITCH(total_tensor.dtype(), NType, {
    std::tie(unique_tensor, key_tensor, value_tenosr) =
        Unique<NType>(torch::cat(mapping_tensors, 0));
    reindex_tensor = Relabel<NType>(total_tensor, key_tensor, value_tenosr);
  });

  std::vector<torch::Tensor> ret =
      reindex_tensor.split_with_sizes(split_sizes, 0);

  return std::make_tuple(unique_tensor, ret);
}

}  // namespace cuda
}  // namespace sampling
}  // namespace dgs