import torch
import torch.distributed as dist
import dgs


def shared_tensor(tensor: torch.Tensor, root: int = 0):
    assert dgs.ops._CAPI_nccl_is_initialized() == True
    assert dist.is_initialized() == True

    if dist.get_rank() == root:
        broadcast_list = [tensor.shape, tensor.dtype]
    else:
        broadcast_list = [None, None]
    dist.broadcast_object_list(broadcast_list, root)
    return dgs.classes.SharedTensor(broadcast_list[0], broadcast_list[1])
