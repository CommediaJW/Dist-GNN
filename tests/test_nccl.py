import DistGNN
from DistGNN.dist import create_communicator
import torch
import torch.distributed as dist
import dgs
import random

dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())
torch.manual_seed(1)
create_communicator(dist.get_world_size(), dist.get_rank())

data = torch.ones(random.randint(0, 10)).cuda() * dist.get_rank()

print("rank: {}/{}, data: {}".format(dgs.ops._Test_GetLocalRank(),
                                     dgs.ops._Test_GetWorldSize(), data))

for i in dgs.ops._Test_NCCLTensorAllGather(data):
    if dist.get_rank() == 0:
        print(i)
