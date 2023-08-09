import dgs
import torch
import torch.distributed as dist


def create_p2p_communicator(group_size, group_rank, local_group=None):
    if group_rank == 0:
        unique_id_array = dgs.ops._CAPI_get_unique_id()
        broadcast_list = [unique_id_array]
    else:
        broadcast_list = [None]

    dist.broadcast_object_list(broadcast_list, 0, local_group)
    unique_ids = broadcast_list[0]
    dgs.ops._CAPI_set_nccl(group_size, unique_ids, group_rank)


dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())
torch.manual_seed(1)
create_p2p_communicator(dist.get_world_size(), dist.get_rank())

indptr = torch.tensor([0, 4, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10])
indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5])

if dist.get_rank() == 0:
    cache_nids = torch.tensor([0, 3])
elif dist.get_rank() == 1:
    cache_nids = torch.tensor([3, 5])
cpu_nids = torch.tensor([1, 2, 4, 6, 7, 8, 9, 10])

sampler = dgs.classes.P2PCacheSampler(indptr, indices, probs, cache_nids,
                                      cpu_nids, dist.get_rank())
seeds = torch.tensor([0, 3, 5]).cuda()
print(sampler._CAPI_get_cpu_structure_tensors())
print(sampler._CAPI_get_cpu_hashmap_tensors())
print(sampler._CAPI_get_local_cache_structure_tensors())
print(sampler._CAPI_get_local_cache_hashmap_tensors())
batch = sampler._CAPI_sample_node_classifiction(seeds, [2, 3], False)
print(batch)
del sampler

sampler_uniform = dgs.classes.P2PCacheSampler(indptr, indices,
                                              torch.tensor([]), cache_nids,
                                              cpu_nids, dist.get_rank())
seeds = torch.tensor([0, 3, 5]).cuda()
print(sampler_uniform._CAPI_get_cpu_structure_tensors())
print(sampler_uniform._CAPI_get_cpu_hashmap_tensors())
print(sampler_uniform._CAPI_get_local_cache_structure_tensors())
print(sampler_uniform._CAPI_get_local_cache_hashmap_tensors())
batch = sampler_uniform._CAPI_sample_node_classifiction(seeds, [2, 3], False)
print(batch)
del sampler_uniform
