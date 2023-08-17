# GraphSAGE Training Benchmark

## Dataset

* ogbn-products
* ogbn-papers100M
* ogbn-papers400M

Run [python/DistGNN/dataloading/dataset_preprocess.py](../../python/DistGNN/dataloading/dataset_preprocess.py) to preprocess the dataset:

```shell
python3 dataset_preprocess.py \
    --dataset ogbn-papers100M \
    --root ${path to the unzipped raw dataset} \
    --save-path ${directory to save the processed dataset}
```

To generate ogbn-papers400M, `--root` should be the path to ogbn-papers100M.

## Run

### Single Node Multi GPUs Training

```shell
python3 node_classification.py \
    --num-gpu 4 \
    --num-epochs 10 \
    --batch-size 1024 \
    --fan-out 5,10,15 \
    --dataset ogbn-papers100M \
    --root ${path to dataset} \
    --cache-policy auto \
    [--bias]
```

`--dataset` can be `ogbn-products`, `ogbn-papers100M` or `ogbn-papers400M`. The flag `--bias` is optional, deciding whether to sample with probability or not. `--cache-policy` can be `auto`, `selfless` or `selfish`.

### Multi Nodes Multi GPUs Training

Run this commond on each node:

```shell
OMP_NUM_THREADS=${#CPUs} torchrun \
    --master_addr ${IP address of master node} \
    --master_port ${free port number on master node} \
    --nnodes ${number of nodes} \
    --node_rank ${rank of this node} \
    node_classification_dist.py \
        --num-gpu 4 \
        --num-epochs 10 \
        --batch-size 1024 \
        --fan-out 5,10,15 \
        --dataset ogbn-papers100M \
        --root ${path to dataset} \
        --cache-policy auto \
        [--bias]
```
