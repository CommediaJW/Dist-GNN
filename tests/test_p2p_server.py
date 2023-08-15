import DistGNN
from DistGNN.dist import create_communicator
import torch
import torch.distributed as dist
import dgs

dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())
torch.manual_seed(1)
create_communicator(dist.get_world_size(), dist.get_rank())

#unique_ids = dgs.ops._CAPI_get_unique_id()
#dgs.ops._CAPI_set_nccl(1, unique_ids, 0)
#print(dgs.ops._Test_GetLocalRank())

print("rank: {}/{}".format(dgs.ops._Test_GetLocalRank(),
                           dgs.ops._Test_GetWorldSize()))

indptr = torch.tensor([0, 4, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10]).pin_memory()

if dgs.ops._Test_GetLocalRank() == 0:
    cache_nids = torch.tensor([0, 3]).cuda()
elif dgs.ops._Test_GetLocalRank() == 1:
    cache_nids = torch.tensor([3, 5]).cuda()

sub_indptr = dgs.ops._Test_ExtractIndptr(cache_nids, indptr)

print("{}: {}".format(dgs.ops._Test_GetLocalRank(), sub_indptr))

p2p_server = DistGNN.capi.classes.TensorP2PServer(sub_indptr)

if dgs.ops._Test_GetLocalRank() == 0:
    print(p2p_server._CAPI_get_local_device_tensor())

    for i in range(dgs.ops._Test_GetWorldSize()):
        print(p2p_server._CAPI_get_device_tensor(i))

import time

time.sleep(1)

if dgs.ops._Test_GetLocalRank() == 1:
    print(p2p_server._CAPI_get_local_device_tensor())

    for i in range(dgs.ops._Test_GetWorldSize()):
        print(p2p_server._CAPI_get_device_tensor(i))
