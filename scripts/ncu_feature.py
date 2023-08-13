import argparse
import torch
import DistGNN


def run(features, args):
    if args.device == "gpu":
        torch.cuda.set_device(0)
        features = features.cuda()
    else:
        DistGNN.capi.ops._CAPI_tensor_pin_memory(features)

    nids = torch.randint(0, args.num_nodes,
                         (args.batch_size, )).unique().long().cuda()

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("loading")
    _ = DistGNN.capi.ops._CAPI_cuda_index(features, nids)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("Nids_num {}".format(nids.numel()))

    if args.device == "cpu":
        DistGNN.capi.ops._CAPI_tensor_unpin_memory(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=10000000)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)

    features = torch.ones((args.num_nodes, args.feat_dim)).float()

    run(features, args)
