import argparse
import torch
import DistGNN
from DistGNN.dataloading import load_dataset


def run(graph, args):
    torch.cuda.set_device(0)
    if args.indptr_device == "gpu":
        indptr = graph["indptr"].cuda()
    else:
        indptr = graph["indptr"]
        DistGNN.capi.ops._CAPI_tensor_pin_memory(indptr)

    if args.indices_device == "gpu":
        indices = graph["indices"].cuda()
    else:
        indices = graph["indices"]
        DistGNN.capi.ops._CAPI_tensor_pin_memory(indices)

    if args.probs_device == "gpu" and args.bias:
        probs = graph["probs"].cuda()
    else:
        probs = graph["probs"]
        DistGNN.capi.ops._CAPI_tensor_pin_memory(probs)

    graph_node_num = indptr.numel() - 1
    seeds = torch.randint(0, graph_node_num,
                          (args.batch_size, )).unique().long().cuda()
    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]

    fact_seeds_num = 0
    for num_picks in fan_out:
        fact_seeds_num += seeds.numel()

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("sampling")

        if args.bias:
            coo_row, coo_col = DistGNN.capi.ops._CAPI_cuda_sample_neighbors_bias(
                seeds, indptr, indices, probs, num_picks, False)
        else:
            coo_row, coo_col = DistGNN.capi.ops._CAPI_cuda_sample_neighbors(
                seeds, indptr, indices, num_picks, False)

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        frontier, (
            coo_row,
            coo_col) = DistGNN.capi.ops._CAPI_cuda_sampled_tensor_relabel(
                [seeds, coo_col], [coo_row, coo_col])

        seeds = frontier

    print("Seeds_num {}".format(fact_seeds_num))

    if args.indptr_device == "cpu":
        DistGNN.capi.ops._CAPI_tensor_unpin_memory(indptr)
    if args.indices_device == "cpu":
        DistGNN.capi.ops._CAPI_tensor_unpin_memory(indices)
    if args.probs_device == "cpu" and args.bias:
        DistGNN.capi.ops._CAPI_tensor_unpin_memory(probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=None,
                        choices=["ogbn-products", "ogbn-papers100M"])
    parser.add_argument("--root",
                        default="dataset/",
                        help='Path of the dataset.')
    parser.add_argument("--bias",
                        action="store_true",
                        default=False,
                        help="Sample with bias.")
    parser.add_argument("--indptr-device",
                        choices=["gpu", "cpu"],
                        default="cpu")
    parser.add_argument("--indices-device",
                        choices=["gpu", "cpu"],
                        default="cpu")
    parser.add_argument("--probs-device",
                        choices=["gpu", "cpu", None],
                        default=None)
    parser.add_argument("--fan-out", type=str, default='10,10,10')
    parser.add_argument("--batch-size",
                        default="5000",
                        type=int,
                        help="The number of seeds of sampling.")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)

    if args.dataset == "ogbn-products":
        graph, _ = load_dataset(args.root, "ogbn-products")
    elif args.dataset == "ogbn-papers100M":
        graph, _ = load_dataset(args.root, "ogbn-papers100M")

    if args.bias:
        graph["probs"] = torch.randn(
            (graph["indices"].shape[0], )).abs().float()

    run(graph, args)
