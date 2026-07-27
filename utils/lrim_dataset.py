"""
LRIM (Long-Range Ising Model) benchmark loader for the ECHO training pipeline.

Node-level regression: predict the per-node energy of an Ising spin configuration on
a periodic L x L grid. It is long-range because at the critical temperature a node's
energy depends on distant spins (https://lrim-graphbenchmark.com).

Per graph (L x L grid, N = L*L nodes):
    x          : (N, 1) float   spin per node
    y          : (N,)   float   per-node energy target  [1-D to match ECHO node tasks]
    edge_index : (2, E) long    periodic nearest-neighbour grid, undirected

Files live on HuggingFace 'jmathys/lrim_graph_benchmark' as lrim_{L}_{sigma}_{count}.pt,
e.g. lrim_16_0.6_10k.pt = 16x16 grid, sigma=0.6 (hard), 10k graphs.
Splits are by index: 80% train / 10% val / 10% test.
"""

import os
import re
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset


class LRIM(InMemoryDataset):
    def __init__(self, root, name='lrim_16_0.6_10k',
                 hf_repo='jmathys/lrim_graph_benchmark',
                 pre_transform=None, transform=None):
        self.name = name
        self.hf_repo = hf_repo
        super().__init__(root, transform=transform, pre_transform=pre_transform)
        # old-style PyG InMemoryDataset load (matches echo_dataset.py)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    # namespace raw/processed under the dataset name so LRIM variants don't collide
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.name}.pt']

    @property
    def processed_file_names(self):
        return [f'{self.name}_data.pt']

    def download(self):
        # one file holds all graphs, e.g. lrim_16_0.6_10k.pt
        from huggingface_hub import hf_hub_download  # lazy: only needed when data is absent
        print(f'Downloading {self.name}.pt from HuggingFace: {self.hf_repo}')
        hf_hub_download(
            repo_id=self.hf_repo,
            filename=f'{self.name}.pt',
            repo_type='dataset',
            local_dir=self.raw_dir,
        )

    def process(self):
        raw = osp.join(self.raw_dir, f'{self.name}.pt')
        data_list = torch.load(raw, weights_only=False)  # list[Data]

        for data in data_list:
            data.x = data.x.float().view(-1, 1)          # (N, 1) spins
            data.y = data.y.float().view(-1)             # (N,)   per-node energy (1-D for MSE path)
            edge_index = data.edge_index.long()

            # the stored grid keeps one direction per edge; add reverse edges to make it undirected
            if edge_index.size(1) < 4 * data.x.size(0):
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
            data.edge_index = edge_index

            for attr in list(data.keys()):               # drop coordinates and any extras
                if attr not in ('x', 'edge_index', 'y'):
                    del data[attr]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_lrim_dataset(root: str, name: str = 'lrim_16_0.6_10k'):
    """Load LRIM and split 80/10/10 by index (same return signature as get_dataset).

    Returns: (train, val, test, num_features=1, num_classes=1) -- scalar node regression.
    """
    full = LRIM(root=root, name=name)

    m = re.search(r'(\d+)k', name, re.IGNORECASE)
    if m is None:
        raise ValueError(f"LRIM name '{name}' must contain a size like '10k'")
    total = int(m.group(1)) * 1000

    n_train = int(0.8 * total)
    n_val = int(0.1 * total)

    train = full[torch.arange(0, n_train)]
    val = full[torch.arange(n_train, n_train + n_val)]
    test = full[torch.arange(n_train + n_val, total)]

    return train, val, test, 1, 1
