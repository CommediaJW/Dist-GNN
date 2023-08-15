import argparse
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import time
import numpy as np
from utils.models import SAGE

import DistGNN
from DistGNN.cache import get_node_heat, get_cache_nids_selfish, get_cache_nids_selfless, compute_total_value_selfish, compute_total_value_selfless, get_available_memory, get_structure_space, get_feature_space
from DistGNN.dataloading import SeedGenerator, load_dataset
from DistGNN.dist import create_communicator


def build_blocks(batch):
    blocks = []
    for layer in batch:
        seeds, frontier, coo_row, coo_col = layer
        block = dgl.create_block((coo_col, coo_row),
                                 num_src_nodes=frontier.numel(),
                                 num_dst_nodes=seeds.numel())
        block.srcdata[dgl.NID] = frontier
        block.dstdata[dgl.NID] = seeds
        blocks.insert(0, block)
    return blocks


def print_memory():
    print("max_memory_allocated: {:.2f} GB, max_memory_reserved {:.2f} GB".
          format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                 torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024))
    print("memory_allocated {:.2f} GB, memory_reserved {:.2f} GB".format(
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024 / 1024))


def run(rank, world_size, data, args):
    graph, num_classes, train_nids_list = data

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    create_communicator(world_size, rank)

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    train_nids = torch.from_numpy(train_nids_list[rank])

    if args.bias:
        probs_key = "probs"
        sampling_heat, feature_heat = get_node_heat(graph["indptr"],
                                                    graph["indices"],
                                                    train_nids,
                                                    fan_out,
                                                    probs=graph[probs_key])
    else:
        probs_key = None
        sampling_heat, feature_heat = get_node_heat(graph["indptr"],
                                                    graph["indices"],
                                                    train_nids, fan_out)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # create model
    model = SAGE(graph["features"].shape[1], 256, num_classes, len(fan_out))
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # create dataloader
    train_dataloader = SeedGenerator(train_nids.cuda(),
                                     args.batch_size,
                                     shuffle=True)

    available_mem = get_available_memory(rank, 7 * 1024 * 1024 * 1024)
    print("GPU {}, available memory size = {:.3f} GB".format(
        rank, available_mem / 1024 / 1024 / 1024))

    # get cache nids
    bandwidth_gpu = 120.62
    bandwidth_host = 8.32
    sampling_read_bytes_gpu = 480
    sampling_read_bytes_host = 480
    feature_read_bytes_gpu = 480
    feature_read_bytes_host = 512
    sampling_cache_nids_selfish, feature_cache_nids_selfish = get_cache_nids_selfish(
        graph,
        sampling_heat,
        feature_heat,
        available_mem,
        bandwidth_gpu,
        sampling_read_bytes_gpu,
        feature_read_bytes_gpu,
        bandwidth_host,
        sampling_read_bytes_host,
        feature_read_bytes_host,
        probs=probs_key)

    print(
        "GPU {}, selfish sampling cache nids num = {}, cache size = {:.2f} GB".
        format(
            rank, sampling_cache_nids_selfish.numel(),
            torch.sum(
                get_structure_space(
                    sampling_cache_nids_selfish, graph, probs=probs_key)) /
            1024 / 1024 / 1024))
    print(
        "GPU {}, selfish feature cache nids num = {}, cache size = {:.2f} GB".
        format(
            rank, feature_cache_nids_selfish.numel(),
            get_feature_space(graph) * feature_cache_nids_selfish.numel() /
            1024 / 1024 / 1024))

    for key in graph:
        DistGNN.capi.ops._CAPI_tensor_pin_memory(graph[key])
    if args.bias:
        probs = graph["probs"]
    else:
        probs = torch.Tensor()

    # cache
    torch.cuda.empty_cache()
    sampler = DistGNN.capi.classes.P2PCacheSampler(
        graph["indptr"], graph["indices"], probs, sampling_cache_nids_selfish,
        rank)

    torch.cuda.empty_cache()
    feature_server = DistGNN.capi.classes.P2PCacheFeatureServer(
        graph["features"], feature_cache_nids_selfish, rank)

    dist.barrier()
    if rank == 0:
        print('start training...')

    iteration_time_log = []
    sampling_time_log = []
    loading_time_log = []
    training_time_log = []
    epoch_iterations_log = []
    epoch_time_log = []

    for epoch in range(args.num_epochs):
        model.train()
        epoch_start = time.time()
        for it, seed_nids in enumerate(train_dataloader):
            torch.cuda.synchronize()
            sampling_start = time.time()
            batch = sampler._CAPI_sample_node_classifiction(
                seed_nids, fan_out, False)
            blocks = build_blocks(batch)
            torch.cuda.synchronize()
            sampling_end = time.time()

            loading_start = time.time()
            batch_inputs = feature_server._CAPI_get_feature(
                blocks[0].srcdata[dgl.NID])
            batch_labels = torch.index_select(graph["labels"], 0,
                                              seed_nids.cpu()).cuda()
            torch.cuda.synchronize()
            loading_end = time.time()

            training_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = F.cross_entropy(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            training_end = time.time()

            sampling_time_log.append(sampling_end - sampling_start)
            loading_time_log.append(loading_end - loading_start)
            training_time_log.append(training_end - training_start)
            iteration_time_log.append(training_end - sampling_start)

        torch.cuda.synchronize()
        epoch_end = time.time()
        epoch_iterations_log.append(it)
        epoch_time_log.append(epoch_end - epoch_start)

    print(
        "Rank {} | Sampling {:.3f} ms | Loading {:.3f} ms | Training {:.3f} ms | Iteration {:.3f} ms | Epoch iterations num {} | Epoch time {:.3f} ms"
        .format(rank,
                np.mean(sampling_time_log[3:]) * 1000,
                np.mean(loading_time_log[3:]) * 1000,
                np.mean(training_time_log[3:]) * 1000,
                np.mean(iteration_time_log[3:]) * 1000,
                np.mean(epoch_iterations_log),
                np.mean(epoch_time_log) * 1000))

    for key in graph:
        DistGNN.capi.ops._CAPI_tensor_unpin_memory(graph[key])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpu',
                        default='8',
                        type=int,
                        help='The number GPU participated in the training.')
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument("--batch-size",
                        default="1000",
                        type=int,
                        help="The number of seeds of sampling.")
    parser.add_argument('--fan-out', type=str, default='10,10,10')
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument("--dataset",
                        default="ogbn-papers100M",
                        choices=["ogbn-products", "ogbn-papers100M"])
    args = parser.parse_args()

    n_procs = min(args.num_gpu, torch.cuda.device_count())
    args.num_gpu = n_procs
    print(args)

    torch.manual_seed(1)

    if args.dataset == "ogbn-products":
        graph, num_classes = load_dataset(args.root, "ogbn-products")
    elif args.dataset == "ogbn-papers100M":
        graph, num_classes = load_dataset(args.root, "ogbn-papers100M")

    if args.bias:
        graph["probs"] = torch.randn(
            (graph["indices"].shape[0], )).abs().float()

    # partition train nodes
    train_nids = graph.pop("train_idx")

    train_nids = train_nids[torch.randperm(train_nids.shape[0])]
    num_train_nids_per_gpu = (train_nids.shape[0] + n_procs - 1) // n_procs
    print("#train nodes {} | #train nodes per gpu {}".format(
        train_nids.shape[0], num_train_nids_per_gpu))
    train_nids_list = []
    for device in range(n_procs):
        local_train_nids = train_nids[device *
                                      num_train_nids_per_gpu:(device + 1) *
                                      num_train_nids_per_gpu]
        train_nids_list.append(local_train_nids.numpy())

    graph["labels"] = graph["labels"].long()

    data = graph, num_classes, train_nids_list

    import torch.multiprocessing as mp
    mp.spawn(run, args=(n_procs, data, args), nprocs=n_procs)
