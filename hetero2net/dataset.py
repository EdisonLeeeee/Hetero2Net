import argparse
from copy import copy

import torch
import torch_geometric.transforms as T

from torch_geometric import seed_everything
from torch_geometric.datasets import OGB_MAG, HGBDataset, RCDD
from torch_geometric.loader import ImbalancedSampler, NeighborLoader
from torch_geometric.utils import sort_edge_index, index_to_mask
from torch_geometric.transforms import RandomNodeSplit
    
def load_dataset(dataset, root='./data'):

    transforms = T.Compose([
        # T.ToUndirected(),
        T.NormalizeFeatures(),
    ])
    
    if dataset == 'ACM':
        data = HGBDataset(root=f'{root}/HGBDataset', name='acm')[0]
        # data['term'].x = torch.eye(data['term'].num_nodes)
        data['term'].x = torch.zeros(data['term'].num_nodes, 1)
        metapaths = [
                    [('author', 'paper'), ('paper', 'author')],
                    [('author', 'paper'), ('paper', 'term'),
                     ('term', 'paper'), ('paper', 'author')],
        ]
    elif dataset == 'DBLP':
        data = HGBDataset(root=f'{root}/HGBDataset', name='dblp', transform=transforms)[0]
        data['venue'].x = torch.eye(data['venue'].num_nodes)
        # data['term'].x = torch.eye(data['term'].num_nodes)
        # data['paper'].x = torch.eye(data['paper'].num_nodes)
        # data['conference'].x = torch.eye(data['paper'].num_nodes)
        metapaths = [
                    [('paper', 'author'), ('author', 'paper')],
                    [('author', 'paper'), ('paper', 'venue'),
                     ('venue', 'paper'), ('paper', 'author')],
        ]
    elif dataset == 'IMDB':
        data = HGBDataset(root=f'{root}/HGBDataset', name='imdb')[0]
        data['keyword'].x = torch.zeros(data['keyword'].num_nodes, 1)
        metapaths = [
                    [('director', 'movie'), ('movie', 'actor')],
                    [('movie', 'keyword'), ('keyword', 'movie')],
        ]
        
    elif dataset == 'FreeBase':
        data = HGBDataset(root=f'{root}/HGBDataset', name='freebase')[0]
        for nt in data.node_types:
            if data[nt].get('x') is None:
                data[nt].x = torch.eye(data[nt].num_nodes)
        metapaths = None
    elif dataset == 'MAG':
        # root = '~/public_data/pyg_data'
        data = OGB_MAG(root, preprocess='metapath2vec',
                       transform=T.ToUndirected())[0]
        metapaths = [
                    [('paper', 'author'), ('author', 'institution'),
                     ('institution', 'author'), ('author', 'paper')],
                    [('author', 'paper'), ('paper', 'field_of_study'),
                     ('field_of_study', 'paper'), ('paper', 'author')],
        ]
    elif dataset == 'RCDD':
        # root = '~/public_data/pyg_data'
        data = RCDD(f'{root}/RCDD')[0]
        metapaths = [
                    [('f', 'item'), ('item', 'b')],
                    [('b', 'item'), ('item', 'f')],
        ]
        train_mask = data['item'].train_mask
        train_idx = train_mask.nonzero().view(-1)
        num_valid_samples = int(train_idx.size(0)*0.2)
        valid_idx = train_idx[torch.randperm(train_idx.size(0))[:num_valid_samples]]
        train_mask[valid_idx] = False
        val_mask = index_to_mask(valid_idx, data['item'].num_nodes)
        data['item'].train_mask = train_mask
        data['item'].val_mask = val_mask
    else:
        raise ValueError(dataset)

    if dataset not in ['RCDD', 'MAG']:
        node_type = [t for t in data.node_types if data[t].get('y') is not None][0]
        data = RandomNodeSplit(num_val=0.2, num_test=0.2)(data)
    return data, metapaths