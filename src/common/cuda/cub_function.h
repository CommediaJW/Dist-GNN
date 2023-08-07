#ifndef DGS_CUDA_CUB_FUNCTION_H_
#define DGS_CUDA_CUB_FUNCTION_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

namespace dgs {
namespace common {
namespace cuda {

template <typename IdType>
inline void cub_exclusiveSum(IdType *arrays, const IdType array_length) {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);

  c10::Allocator *cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
  d_temp_storage = _temp_data.get();
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);
}

template <typename IdType, typename KeyType, typename ValueType>
inline void cub_sortPairsDescending(const IdType *offsets,
                                    const KeyType *in_keys,
                                    const ValueType *in_values,
                                    KeyType *out_keys, ValueType *out_values,
                                    const IdType num_items,
                                    const IdType num_segments) {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, in_keys, out_keys, in_values,
      out_values, num_items, num_segments, offsets, offsets + 1);

  c10::Allocator *cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
  d_temp_storage = _temp_data.get();
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, in_keys, out_keys, in_values,
      out_values, num_items, num_segments, offsets, offsets + 1);
}

}  // namespace cuda
}  // namespace common
}  // namespace dgs

#endif