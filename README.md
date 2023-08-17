# Dist-GNN

## Install

Requirement:
* CUDA >= 11.3
* NCCL >= 2.x
* PyTorch >= 1.12.1
* DGL >= 0.9.1
* pybind11 >= 2.10.4

Install python dependencies.
```shell
$ pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
$ pip3 install --pre dgl -f https://data.dgl.ai/wheels/cu116/repo.html
$ pip3 install pybind11
```

Install the system packages for building the shared library.
```shell
$ sudo apt-get update
$ sudo apt-get install -y build-essential python3-dev make cmake
```

Download the source files.
```shell
$ git clone https://github.com/CommediaJW/Dist-GNN.git
```

Build and install.

```shell
$ bash install.sh
```

## Run Benchmark

see [example/graphsage](./example/graphsage/README.md)
