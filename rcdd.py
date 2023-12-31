import os
import os.path as osp
from typing import Callable, List, Optional
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import (HeteroData, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.utils import index_to_mask


class RCDD(InMemoryDataset):
    url = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/AliRCD_ICDM.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ['AliRCD_ICDM_nodes.csv',
             'AliRCD_ICDM_edges.csv',
             'AliRCD_ICDM_train_labels.csv',
             'AliRCD_ICDM_test_labels.csv',
             'AliRCD_ICDM_test_ids.csv']
        return x

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    @property
    def num_classes(self) -> int:
        return 2

    def process(self):
        import pandas as pd

        data = HeteroData()
        # AliRCD_ICDM_nodes.csv
        nodes = pd.read_csv(self.raw_paths[0], header=None,
                            names=['node_id', 'node_type', 'node_feat'])
        # map global node id to local one for each node type
        mapper = torch.zeros(len(nodes), dtype=torch.long)
        for node_type in tqdm(nodes['node_type'].unique(), desc='Processing node info...'):
            subset = nodes.query(f"node_type == '{node_type}'")
            num_nodes = len(subset)
            mapper[subset['node_id'].values] = torch.arange(num_nodes)
            data[node_type].num_nodes = num_nodes
            x = np.vstack([np.asarray(f.split(':'), dtype=np.float32)
                          for f in subset['node_feat']])
            data[node_type].x = torch.from_numpy(x)
            del x

        # AliRCD_ICDM_edges.csv
        edges = pd.read_csv(self.raw_paths[1], header=None,
                            names=['src_id', 'dst_id',
                                   'src_type', 'dst_type', 'edge_type'])
        for edge_type in tqdm(edges['edge_type'].unique(), desc='Processing edge info...'):
            subset = edges.query(f"edge_type == '{edge_type}'")
            src_type = subset['src_type'].iloc[0]
            dst_type = subset['dst_type'].iloc[0]
            src = mapper[subset['src_id'].values]
            dst = mapper[subset['dst_id'].values]
            data[src_type, edge_type, dst_type].edge_index = torch.stack([
                                                                         src, dst], dim=0)

        # AliRCD_ICDM_train_labels.csv
        train_labels = pd.read_csv(self.raw_paths[2], header=None,
                                   names=['node_id', 'label'], dtype=int)
        # AliRCD_ICDM_test_labels.csv
        test_labels = pd.read_csv(self.raw_paths[3], header=None, sep='\t',
                                  names=['node_id', 'label'], dtype=int)

        train_idx = mapper[train_labels['node_id'].values]
        test_idx = mapper[test_labels['node_id'].values]

        y = torch.full((data['item'].num_nodes,), -1, dtype=torch.long)
        y[train_idx] = torch.from_numpy(train_labels['label'].values)
        y[test_idx] = torch.from_numpy(test_labels['label'].values)

        train_mask = index_to_mask(train_idx, data['item'].num_nodes)
        test_mask = index_to_mask(test_idx, data['item'].num_nodes)

        data['item'].y = y
        data['item'].train_mask = train_mask
        data['item'].test_mask = test_mask

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
