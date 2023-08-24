import torch
import torch.distributed as dist
import dgs


def get_node_heat(indptr: torch.Tensor,
                  indices: torch.Tensor,
                  node_ids: torch.Tensor,
                  fan_outs: list,
                  probs: torch.Tensor = None,
                  mode: str = "uva"):
    assert mode in ["uva", "cuda"]

    if mode == "uva":
        dgs.ops._CAPI_tensor_pin_memory(indptr)
        dgs.ops._CAPI_tensor_pin_memory(indices)
        if probs is not None:
            dgs.ops._CAPI_tensor_pin_memory(probs)
    else:
        indptr = indptr.cuda()
        indices = indices.cuda()
        if probs is not None:
            probs = probs.cuda()

    num_nodes = indptr.shape[0] - 1

    sampling_heat = torch.zeros(num_nodes).cuda()
    seeds_heat = torch.zeros(num_nodes).cuda()
    seeds_heat[node_ids] = 1
    seeds = node_ids.cuda()

    for num_picks in reversed(fan_outs):
        if probs == None:
            frontier_heat = dgs.ops._CAPI_compute_frontier_heat(
                seeds, indptr, indices, seeds_heat, num_picks, 0)

        else:
            frontier_heat = dgs.ops._CAPI_compute_frontier_heat_with_bias(
                seeds, indptr, indices, probs, seeds_heat, num_picks, 0)

        sampling_heat += seeds_heat
        seeds_heat += frontier_heat
        seeds = torch.nonzero(seeds_heat > 0).squeeze(1)

    feature_heat = sampling_heat + frontier_heat

    if mode == "uva":
        dgs.ops._CAPI_tensor_unpin_memory(indptr)
        dgs.ops._CAPI_tensor_unpin_memory(indices)
        if probs is not None:
            dgs.ops._CAPI_tensor_unpin_memory(probs)

    return sampling_heat, feature_heat


# sampling_heat & feature_heat: heat of all the nodes
def get_hot_nids_local(sampling_heat: torch.Tensor,
                       feature_heat: torch.Tensor):
    return torch.nonzero(sampling_heat).flatten(), torch.nonzero(
        feature_heat).flatten()


# Assign nids to gpus, making sure that a node is hottest on the assigned gpu
# sampling_heat & feature_heat: heat of all the nodes
def get_hot_nids_p2p_global(sampling_heat: torch.Tensor,
                            feature_heat: torch.Tensor,
                            group=None):
    assert dist.is_initialized() == True

    group_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)
    global_group_root = dist.get_rank() - dist.get_rank() % group_size

    if group_rank == 0:
        sampling_heat_list = [
            torch.zeros_like(sampling_heat, device="cuda")
            for _ in range(group_size)
        ]
        feature_heat_list = [
            torch.zeros_like(feature_heat, device="cuda")
            for _ in range(group_size)
        ]
    else:
        sampling_heat_list = None
        feature_heat_list = None
    dist.gather(sampling_heat, sampling_heat_list, global_group_root, group)
    dist.gather(feature_heat, feature_heat_list, global_group_root, group)

    if group_rank == 0:
        sampling_heat_list = torch.stack(sampling_heat_list)
        feature_heat_list = torch.stack(feature_heat_list)

        sampling_gpu_ids = torch.argmax(sampling_heat_list, dim=0)
        feature_gpu_ids = torch.argmax(feature_heat_list, dim=0)

        del sampling_heat_list
        del feature_heat_list

        sampling_scatter_list = []
        feature_scatter_list = []
        sampling_tensor_size_scatter_list = []
        feature_tensor_size_scatter_list = []
        for i in range(group_size):
            sampling_scatter_list.append(
                torch.argwhere(sampling_gpu_ids == i).flatten())
            feature_scatter_list.append(
                torch.argwhere(feature_gpu_ids == i).flatten())
            sampling_tensor_size_scatter_list.append(
                torch.tensor([sampling_scatter_list[i].numel()],
                             device="cuda"))
            feature_tensor_size_scatter_list.append(
                torch.tensor([feature_scatter_list[i].numel()], device="cuda"))
        del sampling_gpu_ids
        del feature_gpu_ids
    else:
        sampling_scatter_list = None
        feature_scatter_list = None
        sampling_tensor_size_scatter_list = None
        feature_tensor_size_scatter_list = None

    sampling_nids_tensor_size = torch.tensor([0], device="cuda")
    feature_nids_tensor_size = torch.tensor([0], device="cuda")

    dist.scatter(sampling_nids_tensor_size, sampling_tensor_size_scatter_list,
                 global_group_root, group)
    dist.scatter(feature_nids_tensor_size, feature_tensor_size_scatter_list,
                 global_group_root, group)

    if group_rank == 0:
        for i in range(group_size):
            if i == 0:
                sampling_nids = sampling_scatter_list[0]
                feature_nids = feature_scatter_list[0]
            else:
                dist.send(sampling_scatter_list[i], global_group_root + i,
                          group)
                dist.send(feature_scatter_list[i], global_group_root + i,
                          group)
    else:
        sampling_nids = torch.zeros((sampling_nids_tensor_size[0].item(), ),
                                    dtype=torch.int64,
                                    device="cuda")
        feature_nids = torch.zeros((feature_nids_tensor_size[0].item(), ),
                                   dtype=torch.int64,
                                   device="cuda")
        dist.recv(sampling_nids, global_group_root, group)
        dist.recv(feature_nids, global_group_root, group)

    sampling_nids = sampling_nids[sampling_heat[sampling_nids] > 0]
    feature_nids = feature_nids[feature_heat[feature_nids] > 0]

    return sampling_nids, feature_nids


def get_structure_space(nids: torch.Tensor, graph: dict, probs: str = "None"):
    assert "indices" in graph
    assert "indptr" in graph

    indptr = graph["indptr"].to(nids.device)
    in_degrees = indptr[nids + 1] - indptr[nids]
    if probs is not None:
        assert probs in graph
        return in_degrees * (graph["indices"].element_size() + graph[probs].
                             element_size()) + indptr.element_size()
    else:
        return in_degrees * graph["indices"].element_size(
        ) + indptr.element_size()


def get_feature_space(graph: dict):
    assert "features" in graph
    assert "indptr" in graph
    return int(graph["features"].element_size() * graph["features"].numel() /
               (graph["indptr"].numel() - 1))


# sampling_heat & feature_heat: heat of a part of nodes
def get_node_value(heat: torch.Tensor, space_bytes, reduced_time: float):
    assert isinstance(space_bytes, int) or isinstance(space_bytes,
                                                      torch.Tensor)
    return heat / space_bytes * reduced_time


# locally decide which nids to be cached on gpu
def get_cache_nids_local(
        sampling_nids: torch.Tensor, sampling_space: torch.Tensor,
        sampling_value: torch.Tensor, feature_nids: torch.Tensor,
        feature_space: torch.Tensor, feature_value: torch.Tensor,
        free_capacity_bytes: int):
    all_value = torch.cat([sampling_value, feature_value])
    sorted_ids = torch.argsort(all_value, descending=True)
    cache_bytes = torch.cat([sampling_space, feature_space])[sorted_ids]

    cache_bytes_prefix_sum = torch.cumsum(cache_bytes, 0)
    threshold_id = torch.searchsorted(
        cache_bytes_prefix_sum,
        torch.tensor([free_capacity_bytes], device="cuda"))

    cached_ids = sorted_ids[:threshold_id]

    mask = cached_ids < sampling_nids.numel()

    structure_cache_nids = sampling_nids[cached_ids[mask]]
    feature_cache_nids = feature_nids[cached_ids[~mask] -
                                      sampling_nids.numel()]

    return structure_cache_nids, feature_cache_nids, cache_bytes_prefix_sum[
        threshold_id - 1].item()


# sampling_heat & feature_heat: heat of all the nodes
def get_cache_nids_selfish(graph: dict,
                           sampling_heat: torch.Tensor,
                           feature_heat: torch.Tensor,
                           available_mem,
                           bandwidth_gpu,
                           sampling_read_bytes_gpu,
                           feature_read_bytes_gpu,
                           bandwidth_host,
                           sampling_read_bytes_host,
                           feature_read_bytes_host,
                           probs: str = None):
    sampling_reduced_time = sampling_read_bytes_host / bandwidth_host - sampling_read_bytes_gpu / bandwidth_gpu
    feature_reduced_time = feature_read_bytes_host / bandwidth_host - feature_read_bytes_gpu / bandwidth_gpu

    sampling_hot_nids, feature_hot_nids = get_hot_nids_local(
        sampling_heat, feature_heat)

    sampling_space = get_structure_space(sampling_hot_nids, graph, probs=probs)
    sampling_value = get_node_value(sampling_heat[sampling_hot_nids],
                                    sampling_space, sampling_reduced_time)

    feature_space = get_feature_space(graph)
    feature_value = get_node_value(feature_heat[feature_hot_nids],
                                   feature_space, feature_reduced_time)
    feature_space = torch.full_like(feature_hot_nids, feature_space)

    sampling_cache_nids, feature_cache_nids, _ = get_cache_nids_local(
        sampling_hot_nids, sampling_space, sampling_value, feature_hot_nids,
        feature_space, feature_value, available_mem)

    return sampling_cache_nids, feature_cache_nids


# sampling_heat & feature_heat: heat of all the nodes
def get_cache_nids_selfless(graph: dict,
                            sampling_heat: torch.Tensor,
                            feature_heat: torch.Tensor,
                            available_mem,
                            bandwidth_gpu,
                            sampling_read_bytes_gpu,
                            feature_read_bytes_gpu,
                            bandwidth_host,
                            sampling_read_bytes_host,
                            feature_read_bytes_host,
                            probs: str = None,
                            group=None):
    sampling_reduced_time = sampling_read_bytes_host / bandwidth_host - sampling_read_bytes_gpu / bandwidth_gpu
    feature_reduced_time = feature_read_bytes_host / bandwidth_host - feature_read_bytes_gpu / bandwidth_gpu

    sampling_hot_nids, feature_hot_nids = get_hot_nids_p2p_global(
        sampling_heat, feature_heat, group=group)

    sampling_space = get_structure_space(sampling_hot_nids, graph, probs=probs)
    sampling_value = get_node_value(sampling_heat[sampling_hot_nids],
                                    sampling_space, sampling_reduced_time)

    feature_space = get_feature_space(graph)
    feature_value = get_node_value(feature_heat[feature_hot_nids],
                                   feature_space, feature_reduced_time)
    feature_space = torch.full_like(feature_hot_nids, feature_space)

    sampling_cache_nids, feature_cache_nids, consumed_mem = get_cache_nids_local(
        sampling_hot_nids, sampling_space, sampling_value, feature_hot_nids,
        feature_space, feature_value, available_mem)

    del sampling_hot_nids, feature_hot_nids, sampling_space, feature_space, sampling_value, feature_value

    if available_mem - consumed_mem > 0:
        sampling_heat_backup = sampling_heat[sampling_cache_nids]
        feature_heat_backup = feature_heat[feature_cache_nids]
        sampling_heat[sampling_cache_nids] = 0
        feature_heat[feature_cache_nids] = 0

        sampling_cache_nids_local, feature_cache_nids_local = get_cache_nids_selfish(
            graph,
            sampling_heat,
            feature_heat,
            available_mem - consumed_mem,
            bandwidth_gpu,
            sampling_read_bytes_gpu,
            feature_read_bytes_gpu,
            bandwidth_host,
            sampling_read_bytes_host,
            feature_read_bytes_host,
            probs=probs)

        sampling_heat[sampling_cache_nids] = sampling_heat_backup
        feature_heat[feature_cache_nids] = feature_heat_backup
        del sampling_heat_backup, feature_heat_backup

        sampling_cache_nids = torch.cat(
            [sampling_cache_nids, sampling_cache_nids_local])
        feature_cache_nids = torch.cat(
            [feature_cache_nids, feature_cache_nids_local])

        sampling_cache_nids = sampling_cache_nids[torch.argsort(
            sampling_heat[sampling_cache_nids], descending=True)]
        feature_cache_nids = feature_cache_nids[torch.argsort(
            feature_heat[feature_cache_nids], descending=True)]

    return sampling_cache_nids, feature_cache_nids


# sampling_heat & feature_heat: heat of all the nodes
def compute_total_value_selfish(graph,
                                sampling_heat,
                                feature_heat,
                                sampling_cache_nids,
                                feature_cache_nids,
                                bandwidth_gpu,
                                sampling_read_bytes_gpu,
                                feature_read_bytes_gpu,
                                bandwidth_host,
                                sampling_read_bytes_host,
                                feature_read_bytes_host,
                                probs: str = None):
    value = 0

    sampling_reduced_time = sampling_read_bytes_host / bandwidth_host - sampling_read_bytes_gpu / bandwidth_gpu
    sampling_space = get_structure_space(sampling_cache_nids,
                                         graph,
                                         probs=probs)
    sampling_value = get_node_value(sampling_heat[sampling_cache_nids],
                                    sampling_space, sampling_reduced_time)
    value += torch.sum(sampling_value).item()
    del sampling_space, sampling_value

    feature_reduced_time = feature_read_bytes_host / bandwidth_host - feature_read_bytes_gpu / bandwidth_gpu
    feature_space = get_feature_space(graph)
    feature_value = get_node_value(feature_heat[feature_cache_nids],
                                   feature_space, feature_reduced_time)
    value += torch.sum(feature_value).item()

    return value


# sampling_heat & feature_heat: heat of all the nodes
def compute_total_value_selfless(graph,
                                 sampling_heat,
                                 feature_heat,
                                 sampling_cache_nids,
                                 feature_cache_nids,
                                 bandwidth_gpu,
                                 bandwidth_nvlink,
                                 num_gpu,
                                 sampling_read_bytes_gpu,
                                 feature_read_bytes_gpu,
                                 bandwidth_host,
                                 sampling_read_bytes_host,
                                 feature_read_bytes_host,
                                 probs: str = None,
                                 group=None):
    assert dist.is_initialized() == True

    bandwidth_local = bandwidth_gpu - (num_gpu - 1) * bandwidth_nvlink
    local_value = compute_total_value_selfish(graph,
                                              sampling_heat,
                                              feature_heat,
                                              sampling_cache_nids,
                                              feature_cache_nids,
                                              bandwidth_local,
                                              sampling_read_bytes_gpu,
                                              feature_read_bytes_gpu,
                                              bandwidth_host,
                                              sampling_read_bytes_host,
                                              feature_read_bytes_host,
                                              probs=probs)

    # get known of remote nids
    sampling_nids_mask = torch.zeros(graph["indptr"].numel() - 1,
                                     device="cuda",
                                     dtype=torch.bool)
    sampling_nids_mask[sampling_cache_nids] = True
    feature_nids_mask = torch.zeros(graph["indptr"].numel() - 1,
                                    device="cuda",
                                    dtype=torch.bool)
    feature_nids_mask[feature_cache_nids] = True

    dist.all_reduce(sampling_nids_mask, dist.ReduceOp.SUM, group)
    dist.all_reduce(feature_nids_mask, dist.ReduceOp.SUM, group)

    sampling_nids_mask[sampling_cache_nids] = False
    feature_nids_mask[feature_cache_nids] = False

    remote_sampling_cache_nids = torch.nonzero(sampling_nids_mask).flatten()
    remote_feature_cache_nids = torch.nonzero(feature_nids_mask).flatten()

    remote_value = compute_total_value_selfish(graph,
                                               sampling_heat,
                                               feature_heat,
                                               remote_sampling_cache_nids,
                                               remote_feature_cache_nids,
                                               bandwidth_nvlink,
                                               sampling_read_bytes_gpu,
                                               feature_read_bytes_gpu,
                                               bandwidth_host,
                                               sampling_read_bytes_host,
                                               feature_read_bytes_host,
                                               probs=probs)

    return local_value + remote_value


def get_available_memory(device, reserved_mem):
    available_mem = int(
        torch.cuda.mem_get_info(device)[1] -
        torch.cuda.memory_allocated(device=device) - reserved_mem)
    available_mem = max(available_mem, 0)
    return available_mem
