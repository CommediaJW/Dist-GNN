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

indptr = torch.tensor([0, 4, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10]).pin_memory()
indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).pin_memory()
probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4,
                      0.5]).pin_memory()

if dgs.ops._Test_GetLocalRank() == 0:
    cache_nids = torch.tensor([0, 3])
elif dgs.ops._Test_GetLocalRank() == 1:
    cache_nids = torch.tensor([3, 5])

print("rank: {}/{}".format(dgs.ops._Test_GetLocalRank(),
                           dgs.ops._Test_GetWorldSize()))

sampler = DistGNN.capi.classes.P2PCacheSampler(indptr, indices, probs,
                                               cache_nids,
                                               dgs.ops._Test_GetLocalRank())

for i in sampler._CAPI_get_cpu_structure_tensors():
    if dgs.ops._Test_GetLocalRank() == 1:
        print(i)

for i in sampler._CAPI_get_local_cache_structure_tensors():
    if dgs.ops._Test_GetLocalRank() == 1:
        print(i)

for i in sampler._CAPI_get_local_cache_hashmap_tensors():
    if dgs.ops._Test_GetLocalRank() == 1:
        print(i)
