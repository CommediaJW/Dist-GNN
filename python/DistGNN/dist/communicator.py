import torch.distributed as dist
import dgs


def create_communicator(group_size, group=None):
    group_rank = dist.get_rank() % group_size
    group_root = dist.get_rank() - dist.get_rank() % group_size

    if group_rank == 0:
        unique_id_array = dgs.ops._CAPI_get_unique_id()
        broadcast_list = [unique_id_array]
    else:
        broadcast_list = [None]

    dist.broadcast_object_list(broadcast_list, group_root, group)
    unique_ids = broadcast_list[0]
    dgs.ops._CAPI_set_nccl(group_size, unique_ids, group_rank)
