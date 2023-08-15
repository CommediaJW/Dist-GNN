import DistGNN
from DistGNN.dist import create_communicator
import torch
import torch.distributed as dist
import dgs
import time

dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())
torch.manual_seed(1)
create_communicator(dist.get_world_size(), dist.get_rank())

indptr = torch.tensor([0, 4, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10]).pin_memory()
indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).pin_memory()
#probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4,
#                      0.5]).pin_memory()

if dgs.ops._Test_GetLocalRank() == 0:
    cache_nids = torch.tensor([0, 3])
elif dgs.ops._Test_GetLocalRank() == 1:
    cache_nids = torch.tensor([3, 5])

print("rank: {}/{}".format(dgs.ops._Test_GetLocalRank(),
                           dgs.ops._Test_GetWorldSize()))

sampler = DistGNN.capi.classes.P2PCacheSampler(indptr, indices, torch.Tensor(),
                                               cache_nids,
                                               dgs.ops._Test_GetLocalRank())

for i in sampler._CAPI_sample_node_classifiction(
        torch.tensor([0, 3, 5]).cuda(), [10, 10], False):
    if dgs.ops._Test_GetLocalRank() == 0:
        for j in i:
            print(dgs.ops._Test_GetLocalRank(), j)

    if dgs.ops._Test_GetLocalRank() == 1:
        time.sleep(3)
        for j in i:
            print(dgs.ops._Test_GetLocalRank(), j)
