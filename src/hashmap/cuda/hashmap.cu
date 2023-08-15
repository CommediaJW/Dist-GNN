#include "../../common/cuda_common.h"
#include "../../common/dgs_headers.h"
#include "../../nccl/nccl_context.h"

#include "gpu_hash_table.cuh"
#include "hashmap.h"
#include "ops.h"

#define BLOCK_SIZE 128

namespace dgs {
namespace hashmap {
namespace cuda {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CreateNidsP2PCacheHashMapCUDA(std::vector<torch::Tensor> devices_cache_nids,
                              int64_t all_devices_cache_nids_num,
                              int64_t local_device_id) {
  DGS_ID_TYPE_SWITCH(devices_cache_nids[0].dtype(), IdType, {
    int64_t dir_size = _UpPower(all_devices_cache_nids_num) * 2;
    torch::Tensor key_tensor =
        torch::full({dir_size}, -1,
                    torch::TensorOptions()
                        .dtype(devices_cache_nids[0].dtype())
                        .device(torch::kCUDA));
    torch::Tensor in_gpu_nids_tensor =
        torch::full({dir_size}, -1,
                    torch::TensorOptions()
                        .dtype(devices_cache_nids[0].dtype())
                        .device(torch::kCUDA));
    torch::Tensor gpu_id_tensor =
        torch::full({dir_size}, -1,
                    torch::TensorOptions()
                        .dtype(devices_cache_nids[0].dtype())
                        .device(torch::kCUDA));

    for (int device_id = 0; device_id < devices_cache_nids.size();
         device_id += 1) {
      int index =
          (device_id + nccl::nccl_ctx.local_rank_) % devices_cache_nids.size();

      if (index != local_device_id) {
        int64_t num_items = devices_cache_nids[index].size(0);

        using it = thrust::counting_iterator<IdType>;
        thrust::for_each(it(0), it(num_items),
                         [in_key = devices_cache_nids[index].data_ptr<IdType>(),
                          key_buff = key_tensor.data_ptr<IdType>(),
                          gpu_nids_buff = in_gpu_nids_tensor.data_ptr<IdType>(),
                          gpu_id_buff = gpu_id_tensor.data_ptr<IdType>(),
                          dir_size, index] __device__(int64_t i) mutable {
                           Hashmap<IdType, IdType> table(
                               key_buff, gpu_nids_buff, dir_size);
                           IdType pos = table.Update(in_key[i], i);
                           gpu_id_buff[pos] = index;
                         });
      }
    }

    int64_t num_items = devices_cache_nids[local_device_id].size(0);
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        it(0), it(num_items),
        [in_key = devices_cache_nids[local_device_id].data_ptr<IdType>(),
         key_buff = key_tensor.data_ptr<IdType>(),
         gpu_nids_buff = in_gpu_nids_tensor.data_ptr<IdType>(),
         gpu_id_buff = gpu_id_tensor.data_ptr<IdType>(), dir_size,
         local_device_id] __device__(int64_t i) mutable {
          Hashmap<IdType, IdType> table(key_buff, gpu_nids_buff, dir_size);
          IdType pos = table.Update(in_key[i], i);
          gpu_id_buff[pos] = local_device_id;
        });

    return std::make_tuple(key_tensor, in_gpu_nids_tensor, gpu_id_tensor);
  });
  return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor());
}

}  // namespace cuda
}  // namespace hashmap
}  // namespace dgs