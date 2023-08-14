#ifndef DGS_HASHMAP_CUDA_H_
#define DGS_HASHMAP_CUDA_H_

#include <torch/script.h>
#include "../../common/cuda/atomic.h"
#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"

namespace dgs {
namespace hashmap {
namespace cuda {
template <typename IdType, typename ValueType>
struct Hashmap {
  __device__ inline Hashmap(IdType *__restrict__ Kptr,
                            ValueType *__restrict__ Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline IdType Update(IdType key, ValueType value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    IdType prev = common::cuda::AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = common::cuda::AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    vptr[pos] = value;

    return pos;
  }

  __device__ inline IdType SearchForPos(IdType key) {
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

  // 32 bit Murmur3 hash
  __device__ inline uint32_t Hash32Shift(uint32_t k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
  }

  // 64 bit Murmur3 hash
  __device__ inline uint64_t Hash64Shift(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccd;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53;
    k ^= k >> 33;
    return k;
  }

  __device__ inline uint32_t hash(int32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(int64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  IdType kEmptyKey{-1};
  IdType *kptr;
  ValueType *vptr;
  uint32_t capacity{0};
};

inline int _UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}
}  // namespace cuda
}  // namespace hashmap
}  // namespace dgs

#endif