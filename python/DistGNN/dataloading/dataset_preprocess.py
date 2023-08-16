import argparse
import numpy as np
import pandas as pd
import torch
import os
from scipy.sparse import coo_matrix


def process_products(dataset_path, save_path):
    print("Process ogbn-products...")

    print("Read raw data...")
    edges = pd.read_csv(os.path.join(dataset_path, "raw/edge.csv.gz"),
                        compression="gzip",
                        header=None).values.T
    features = pd.read_csv(os.path.join(dataset_path, "raw/node-feat.csv.gz"),
                           compression="gzip",
                           header=None).values
    labels = pd.read_csv(os.path.join(dataset_path, "raw/node-label.csv.gz"),
                         compression="gzip",
                         header=None).values.T[0]
    train_idx = pd.read_csv(os.path.join(dataset_path,
                                         "split/sales_ranking/train.csv.gz"),
                            compression="gzip",
                            header=None).values.T[0]
    valid_idx = pd.read_csv(os.path.join(dataset_path,
                                         "split/sales_ranking/valid.csv.gz"),
                            compression="gzip",
                            header=None).values.T[0]
    test_idx = pd.read_csv(os.path.join(dataset_path,
                                        "split/sales_ranking/test.csv.gz"),
                           compression="gzip",
                           header=None).values.T[0]

    print("Process data...")
    num_nodes = features.shape[0]
    src = np.concatenate((edges[0], edges[1]))
    dst = np.concatenate((edges[1], edges[0]))
    data = np.zeros(src.shape)
    coo = coo_matrix((data, (dst, src)),
                     shape=(num_nodes, num_nodes),
                     dtype=np.int64)
    csc = coo.tocsr()
    indptr = csc.indptr
    indices = csc.indices
    num_edges = indices.shape[0]

    print("Save data...")
    torch.save(
        torch.from_numpy(features).float(),
        os.path.join(save_path, "features.pt"))
    torch.save(
        torch.from_numpy(labels).long(), os.path.join(save_path, "labels.pt"))
    torch.save(
        torch.from_numpy(indptr).long(), os.path.join(save_path, "indptr.pt"))
    torch.save(
        torch.from_numpy(indices).long(), os.path.join(save_path,
                                                       "indices.pt"))
    torch.save(
        torch.from_numpy(train_idx).long(),
        os.path.join(save_path, "train_idx.pt"))
    torch.save(
        torch.from_numpy(valid_idx).long(),
        os.path.join(save_path, "valid_idx.pt"))
    torch.save(
        torch.from_numpy(test_idx).long(),
        os.path.join(save_path, "test_idx.pt"))

    print("Generate meta data...")
    num_classes = np.unique(labels[~np.isnan(labels)]).shape[0]
    feature_dim = features.shape[1]
    num_train_nodes = train_idx.shape[0]
    num_valid_nodes = valid_idx.shape[0]
    num_test_nodes = test_idx.shape[0]

    print("Save meta data...")
    meta_data = {
        "dataset": "ogbn-products",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "num_train_nodes": num_train_nodes,
        "num_valid_nodes": num_valid_nodes,
        "num_test_nodes": num_test_nodes
    }
    torch.save(meta_data, os.path.join(save_path, "metadata.pt"))


def process_papers100M(dataset_path, save_path):
    print("Process ogbn-papers100M...")

    print("Read raw data...")
    data_file = np.load(os.path.join(dataset_path, "raw/data.npz"))
    label_file = np.load(os.path.join(dataset_path, "raw/node-label.npz"))
    features = data_file["node_feat"]
    labels = label_file["node_label"]
    edge_index = data_file["edge_index"]
    train_idx = pd.read_csv(os.path.join(dataset_path,
                                         'split/time/train.csv.gz'),
                            compression='gzip',
                            header=None).values.T[0]
    valid_idx = pd.read_csv(os.path.join(dataset_path,
                                         'split/time/valid.csv.gz'),
                            compression='gzip',
                            header=None).values.T[0]
    test_idx = pd.read_csv(os.path.join(dataset_path,
                                        'split/time/test.csv.gz'),
                           compression='gzip',
                           header=None).values.T[0]

    print("Process data...")
    num_nodes = features.shape[0]
    src = edge_index[0]
    dst = edge_index[1]
    data = np.zeros(src.shape)
    coo = coo_matrix((data, (dst, src)),
                     shape=(num_nodes, num_nodes),
                     dtype=np.int64)
    csc = coo.tocsr()
    indptr = csc.indptr
    indices = csc.indices
    num_edges = indices.shape[0]

    print("Save data...")
    torch.save(
        torch.from_numpy(features).float(),
        os.path.join(save_path, "features.pt"))
    torch.save(
        torch.from_numpy(labels).float().squeeze(1),
        os.path.join(save_path, "labels.pt"))
    torch.save(
        torch.from_numpy(indptr).long(), os.path.join(save_path, "indptr.pt"))
    torch.save(
        torch.from_numpy(indices).long(), os.path.join(save_path,
                                                       "indices.pt"))
    torch.save(
        torch.from_numpy(train_idx).long(),
        os.path.join(save_path, "train_idx.pt"))
    torch.save(
        torch.from_numpy(valid_idx).long(),
        os.path.join(save_path, "valid_idx.pt"))
    torch.save(
        torch.from_numpy(test_idx).long(),
        os.path.join(save_path, "test_idx.pt"))

    print("Generate meta data...")
    num_classes = np.unique(labels[~np.isnan(labels)]).shape[0]
    feature_dim = features.shape[1]
    num_train_nodes = train_idx.shape[0]
    num_valid_nodes = valid_idx.shape[0]
    num_test_nodes = test_idx.shape[0]

    print("Save meta data...")
    meta_data = {
        "dataset": "ogbn-papers100M",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "num_train_nodes": num_train_nodes,
        "num_valid_nodes": num_valid_nodes,
        "num_test_nodes": num_test_nodes
    }
    torch.save(meta_data, os.path.join(save_path, "metadata.pt"))


def generate_papers400M(papers100M_path, save_path):
    print("Read ogbn-papers100M raw data...")
    data_file = np.load(os.path.join(papers100M_path, "raw/data.npz"))
    label_file = np.load(os.path.join(papers100M_path, "raw/node-label.npz"))
    original_features = data_file["node_feat"]
    original_labels = label_file["node_label"]
    edge_index = data_file["edge_index"]

    print("Generate ogbn-papers400M csc graph...")
    original_src = edge_index[0]
    original_dst = edge_index[1]
    n_nodes = original_features.shape[0]
    n_edges = edge_index[0].shape[0]
    intra_src = np.concatenate([
        np.arange(0, n_nodes, dtype=np.int64).repeat(3).flatten(),
        np.arange(n_nodes, 2 * n_nodes, dtype=np.int64).repeat(3).flatten(),
        np.arange(2 * n_nodes, 3 * n_nodes,
                  dtype=np.int64).repeat(3).flatten(),
        np.arange(3 * n_nodes, 4 * n_nodes,
                  dtype=np.int64).repeat(3).flatten()
    ])
    intra_dst = np.concatenate([
        np.arange(n_nodes, 4 * n_nodes, dtype=np.int64),
        np.concatenate([
            np.arange(0 * n_nodes, 1 * n_nodes, dtype=np.int64),
            np.arange(2 * n_nodes, 4 * n_nodes, dtype=np.int64)
        ]),
        np.concatenate([
            np.arange(0 * n_nodes, 2 * n_nodes, dtype=np.int64),
            np.arange(3 * n_nodes, 4 * n_nodes, dtype=np.int64)
        ]),
        np.arange(0, 3 * n_nodes, dtype=np.int64)
    ])
    sm = np.random.randint(0, 4, (2 * n_edges, ), dtype=np.int64)
    dm = np.random.randint(0, 4, (2 * n_edges, ), dtype=np.int64)
    src = np.concatenate([
        original_src + sm[:n_edges] * n_nodes,
        original_dst + sm[n_edges:] * n_nodes, intra_src
    ])
    dst = np.concatenate([
        original_dst + dm[:n_edges] * n_nodes,
        original_src + dm[n_edges:] * n_nodes, intra_dst
    ])
    data = np.zeros(src.shape, dtype=np.int64)

    num_nodes = 4 * n_nodes
    coo = coo_matrix((data, (dst, src)),
                     shape=(num_nodes, num_nodes),
                     dtype=np.int64)
    csc = coo.tocsr()
    indptr = csc.indptr
    indices = csc.indices
    num_edges = indices.shape[0]
    print("Save csc graph...")
    torch.save(
        torch.from_numpy(indptr).long(), os.path.join(save_path, "indptr.pt"))
    torch.save(
        torch.from_numpy(indices).long(), os.path.join(save_path,
                                                       "indices.pt"))
    del indptr, indices, coo, csc, original_src, original_dst, intra_src, intra_dst, sm, dm, src, dst, data

    print("Generate features...")
    features = np.concatenate([
        original_features, original_features, original_features,
        original_features
    ],
                              axis=0)
    print("Save features...")
    torch.save(
        torch.from_numpy(features).float(),
        os.path.join(save_path, "features.pt"))
    feature_dim = features.shape[1]
    del features

    print("Generate labels...")
    labels = np.concatenate(
        [original_labels, original_labels, original_labels, original_labels])
    print("Save labels...")
    torch.save(
        torch.from_numpy(labels).float().squeeze(1),
        os.path.join(save_path, "labels.pt"))
    num_classes = np.unique(labels[~np.isnan(labels)]).shape[0]
    del labels

    print("Read papers100M train, valid, test idx...")
    original_train_idx = pd.read_csv(os.path.join(papers100M_path,
                                                  'split/time/train.csv.gz'),
                                     compression='gzip',
                                     header=None).values.T[0]
    original_valid_idx = pd.read_csv(os.path.join(papers100M_path,
                                                  'split/time/valid.csv.gz'),
                                     compression='gzip',
                                     header=None).values.T[0]
    original_test_idx = pd.read_csv(os.path.join(papers100M_path,
                                                 'split/time/test.csv.gz'),
                                    compression='gzip',
                                    header=None).values.T[0]

    print("Generate train idx...")
    train_idx = np.concatenate([
        original_train_idx, original_train_idx + n_nodes,
        original_train_idx + 2 * n_nodes, original_train_idx + 3 * n_nodes
    ])
    print("Save train idx...")
    torch.save(
        torch.from_numpy(train_idx).long(),
        os.path.join(save_path, "train_idx.pt"))
    num_train_nodes = train_idx.shape[0]
    del train_idx

    print("Generate valid idx...")
    valid_idx = np.concatenate([
        original_valid_idx, original_valid_idx + n_nodes,
        original_valid_idx + 2 * n_nodes, original_valid_idx + 3 * n_nodes
    ])
    print("Save valid idx...")
    torch.save(
        torch.from_numpy(valid_idx).long(),
        os.path.join(save_path, "valid_idx.pt"))
    num_valid_nodes = valid_idx.shape[0]
    del valid_idx

    print("Generate test idx...")
    test_idx = np.concatenate([
        original_test_idx, original_test_idx + n_nodes,
        original_test_idx + 2 * n_nodes, original_test_idx + 3 * n_nodes
    ])
    print("Save test idx...")
    torch.save(
        torch.from_numpy(test_idx).long(),
        os.path.join(save_path, "test_idx.pt"))
    num_test_nodes = test_idx.shape[0]
    del test_idx

    print("Save meta data...")
    meta_data = {
        "dataset": "ogbn-papers400M",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "num_train_nodes": num_train_nodes,
        "num_valid_nodes": num_valid_nodes,
        "num_test_nodes": num_test_nodes
    }
    torch.save(meta_data, os.path.join(save_path, "metadata.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="ogbn-papers100M",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-papers400M"])
    parser.add_argument("--root", help="Path of the dataset.")
    parser.add_argument("--save-path", help="Path to save the processed data.")
    args = parser.parse_args()
    print(args)

    if args.dataset == "ogbn-papers100M":
        process_papers100M(args.root, args.save_path)
    elif args.dataset == "ogbn-products":
        process_products(args.root, args.save_path)
    elif args.dataset == "ogbn-papers400M":
        generate_papers400M(args.root, args.save_path)
