#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <torch/python.h>
#include <unistd.h>

#include "../context/context.h"
#include "../nccl/nccl_context.h"
#include "dgs_headers.h"
#include "shared_mem.h"

namespace dgs {

SharedMemory::SharedMemory(const std::string &name) {
  this->name = name;
  this->own_ = false;
  this->fd_ = -1;
  this->ptr_ = nullptr;
  this->size_ = 0;
}

SharedMemory::~SharedMemory() {
  if (ptr_ && size_ != 0) CHECK(munmap(ptr_, size_) != -1) << strerror(errno);
  if (fd_ != -1) close(fd_);
  if (own_) {
    if (name != "") {
      shm_unlink(name.c_str());
    }
  }
}

void *SharedMemory::CreateNew(size_t sz) {
  this->own_ = true;

  // We need to create a shared-memory file.
  int flag = O_RDWR | O_CREAT;
  fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd_, -1) << "fail to open " << name << ": " << strerror(errno);
  // Shared memory cannot be deleted if the process exits abnormally in Linux.
  auto res = ftruncate(fd_, sz);
  CHECK_NE(res, -1) << "Failed to truncate the file. " << strerror(errno);
  ptr_ = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  CHECK_NE(ptr_, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error "
      << strerror(errno);
  this->size_ = sz;
  return ptr_;
}

void *SharedMemory::Open(size_t sz) {
  int flag = O_RDWR;
  fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd_, -1) << "fail to open " << name << ": " << strerror(errno);
  ptr_ = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  CHECK_NE(ptr_, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error "
      << strerror(errno);
  this->size_ = sz;
  return ptr_;
}

bool SharedMemory::Exist(const std::string &name) {
  int fd = shm_open(name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
  if (fd >= 0) {
    close(fd);
    return true;
  } else {
    return false;
  }
}

inline torch::ScalarType _pybindObj2TorchDtype(pybind11::object dtype) {
  return torch::python::detail::py_object_to_dtype(dtype);
}

inline size_t _getTensorTypeSizeOf(torch::Dtype type) {
  if (type == torch::kInt32) {
    return sizeof(int32_t);
  } else if (type == torch::kInt64) {
    return sizeof(int64_t);
  } else if (type == torch::kFloat) {
    return sizeof(float);
  } else if (type == torch::kDouble) {
    return sizeof(double);
  } else if (type == torch::kBool) {
    return sizeof(bool);
  } else {
    fprintf(stderr, "Error in _getTensorSizeInByte!\n");
    exit(-1);
  }
}

inline std::string _getSharedMemName(uint32_t id) {
  return "shared" + std::to_string(id);
}

SharedTensor::SharedTensor(std::vector<int64_t> shapes,
                           pybind11::object dtype) {
  int64_t rank = dgs::nccl::GetLocalRank();

  auto torch_dtype = _pybindObj2TorchDtype(dtype);
  int64_t numel = 1;
  for (uint64_t i = 0; i < shapes.size(); i += 1) {
    numel *= shapes[i];
  }
  this->size_ = _getTensorTypeSizeOf(torch_dtype) * numel;

  uint32_t shmid;
  if (rank == 0) {
    shmid = dgs::ctx::randn_uint64();
    while (SharedMemory::Exist(_getSharedMemName(shmid))) {
      shmid += 1;
    }
  } else {
    shmid = 0;
  }

  CUDA_CALL(cudaHostRegister(&shmid, sizeof(int), cudaHostRegisterDefault));
  NCCL_CALL(ncclBroadcast(&shmid, &shmid, 1, ncclInt, 0,
                          dgs::nccl::nccl_ctx.global_comm_,
                          dgs::nccl::nccl_ctx.nccl_stream_));
  dgs::nccl::nccl_ctx.Barrier_();
  CUDA_CALL(cudaHostUnregister(&shmid));

  this->mem_ = std::make_shared<SharedMemory>(_getSharedMemName(shmid));
  if (rank == 0) {
    this->data_ = this->mem_->CreateNew(this->size_);
  }
  dgs::nccl::nccl_ctx.Barrier_();
  if (rank != 0) {
    this->data_ = this->mem_->Open(this->size_);
  }
  this->tensor_ = torch::from_blob(
      this->data_, shapes,
      torch::TensorOptions().dtype(torch_dtype).device(torch::kCPU));
}

void SharedTensor::LoadFromTensor(torch::Tensor data) {
  if (dgs::nccl::GetLocalRank() == 0) {
    CHECK(this->tensor_.dtype() == data.dtype());
    CHECK(data.device() == torch::kCPU);
    for (uint64_t i = 0; i < this->tensor_.sizes().size(); i += 1) {
      CHECK(this->tensor_.size(i) == data.size(i));
    }

    this->tensor_.copy_(data);
  }
}

void SharedTensor::LoadFromDisk(std::string filename) {
  if (dgs::nccl::GetLocalRank() == 0) {
    int fd = open(filename.data(), O_RDONLY);
    CHECK_NE(fd, -1) << "fail to open " << filename << ": " << strerror(errno);
    void *ptr = mmap(NULL, this->size_, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK_NE(ptr, MAP_FAILED) << "mmap failed with error " << strerror(errno);
    memcpy(this->data_, ptr, this->size_);
    CHECK(munmap(ptr, this->size_) != -1) << strerror(errno);
    close(fd);
  }
}

}  // namespace dgs