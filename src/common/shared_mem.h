#ifndef DGS_SHARED_TENSOR_H_
#define DGS_SHARED_TENSOR_H_

#include <torch/script.h>

namespace dgs {

class SharedMemory {
  /**
   * @brief whether the shared memory is owned by the object.
   *
   * If shared memory is created in the object, it'll be owned by the object
   * and will be responsible for deleting it when the object is destroyed.
   */
  bool own_;

  /* @brief the file descripter of the shared memory. */
  int fd_;
  /* @brief the address of the shared memory. */
  void *ptr_;
  /* @brief the size of the shared memory. */
  size_t size_;

  /**
   * @brief the name of the object.
   *
   * In Unix, shared memory is identified by a file. Thus, `name` is actually
   * the file name that identifies the shared memory.
   */
  std::string name;

 public:
  /* @brief Get the filename of shared memory file
   */
  std::string GetName() const { return name; }

  /**
   * @brief constructor of the shared memory.
   * @param name The file corresponding to the shared memory.
   */
  explicit SharedMemory(const std::string &name);
  /**
   * @brief destructor of the shared memory.
   * It deallocates the shared memory and removes the corresponding file.
   */
  ~SharedMemory();
  /**
   * @brief create shared memory.
   * It creates the file and shared memory.
   * @param sz the size of the shared memory.
   * @return the address of the shared memory
   */
  void *CreateNew(size_t sz);
  /**
   * @brief allocate shared memory that has been created.
   * @param sz the size of the shared memory.
   * @return the address of the shared memory
   */
  void *Open(size_t sz);

  /**
   * @brief check if the shared memory exist.
   * @param name the name of the shared memory.
   * @return a boolean value to indicate if the shared memory exists.
   */
  static bool Exist(const std::string &name);
};

class SharedTensor {
 public:
  SharedTensor(std::vector<int64_t> shapes, pybind11::object dtype);
  ~SharedTensor(){};

  void LoadFromTensor(torch::Tensor data);
  void LoadFromDisk(std::string filename);
  torch::Tensor Tensor() { return tensor_; };

 private:
  std::shared_ptr<SharedMemory> mem_;
  void *data_;
  size_t size_;
  torch::Tensor tensor_;
};

void SaveTensor2Disk(torch::Tensor tensor, std::string filename);

}  // namespace dgs

#endif