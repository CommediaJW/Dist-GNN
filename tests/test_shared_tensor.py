import DistGNN
import time
import torch
import torch.distributed as dist
from DistGNN.dist import create_communicator

dist.init_process_group(backend='nccl', init_method="env://")
torch.cuda.set_device(dist.get_rank())
torch.set_num_threads(1)
torch.manual_seed(1)

create_communicator(dist.get_world_size())

tensor = None
if dist.get_rank() == 0:
    tensor = torch.arange(0, 20)
    broadcast_list = [tensor.shape, tensor.dtype]
else:
    broadcast_list = [None, None]

dist.broadcast_object_list(broadcast_list, 0)
shared_tensor = DistGNN.capi.classes.SharedTensor(broadcast_list[0],
                                                  broadcast_list[1])

if dist.get_rank() == 0:
    shared_tensor._CAPI_load_from_tensor(tensor)

time.sleep(1)
dist.barrier()
if dist.get_rank() == 0:
    print("Rank 0")
    print(shared_tensor._CAPI_get_tensor())

time.sleep(1)
dist.barrier()
if dist.get_rank() == 1:
    print("Rank 1")
    print(shared_tensor._CAPI_get_tensor())

time.sleep(1)
dist.barrier()
if dist.get_rank() == 0:
    print("Rank 0, add 1")
    tensor = shared_tensor._CAPI_get_tensor()
    tensor += 1
    print(shared_tensor._CAPI_get_tensor())

time.sleep(1)
dist.barrier()
if dist.get_rank() == 1:
    print("Rank 1")
    print(shared_tensor._CAPI_get_tensor())