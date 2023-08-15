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

feature = torch.arange(0, 100, 1).float()  #.reshape(10, 10)

# print(feature)

if dgs.ops._Test_GetLocalRank() == 0:
    cache_nids = torch.tensor([0, 3]).cuda()
elif dgs.ops._Test_GetLocalRank() == 1:
    cache_nids = torch.tensor([3, 5]).cuda()

feature_server = DistGNN.capi.classes.P2PCacheFeatureServer(
    feature, cache_nids, dgs.ops._Test_GetLocalRank())

if dgs.ops._Test_GetLocalRank() == 0:
    print(feature_server._CAPI_get_cpu_feature())

if dgs.ops._Test_GetLocalRank() == 0:
    print(feature_server._CAPI_get_gpu_feature())

if dgs.ops._Test_GetLocalRank() == 0:
    print(feature_server._CAPI_get_feature(torch.tensor([0, 3, 5, 7]).cuda()))
