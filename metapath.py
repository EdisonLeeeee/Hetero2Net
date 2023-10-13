from typing import Optional, Tuple, Union

import copy
import torch
from torch import Tensor
try:
    import torch_cluster
except ImportError:
    torch_cluster = None

from torch_geometric.utils import subgraph, degree, sort_edge_index

from torch_geometric.utils.num_nodes import maybe_num_nodes

def drop_path(edge_index: Tensor,
              r: Optional[Union[float, Tensor]] = 0.5,
              walks_per_node: int = 2,
              walk_length: int = 4,
              p: float = 1, q: float = 1,
              training: bool = True,
              num_nodes: int = None,
              by: str = 'uniform',
              return_dropped: bool = False) -> Tuple[Tensor, Tensor]:

    if torch_cluster is None:
        raise ImportError("`torch_cluster` is not installed.")

    assert by in {'degree', 'uniform'}
    # edge_index = sort_edge_index(edge_index, num_nodes=num_nodes)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)

    if isinstance(r, (int, float)):
        if r <= 0. or r > 1.:
            raise ValueError(f'Root node sampling ratio `r` has to be between 0 and 1 '
                             f'(got {r}')
        num_starts = int(r * num_nodes)
        if by == 'degree':
            prob = deg / deg.sum()
            start = prob.multinomial(num_samples=num_starts, replacement=True)
        else:
            start = torch.randperm(num_nodes, device=edge_index.device)[
                :num_starts]
    elif torch.is_tensor(r):
        start = r.to(edge_index)
    else:
        raise ValueError('Root node sampling ratio `r` must be '
                         f'`float`, `torch.Tensor`, but got {r}.')

    if walks_per_node:
        start = start.repeat(walks_per_node)

    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    n_id, e_id = torch.ops.torch_cluster.random_walk(
        rowptr, col, start, walk_length, p, q)
    e_id = e_id[e_id > 0]

    mask = row.new_ones(row.size(0), dtype=torch.bool)

    if e_id.numel() > 0:
        mask[e_id.view(-1)] = False

    if return_dropped:
        return edge_index[:, mask], edge_index[:, ~mask]
    else:
        return edge_index[:, mask]
    
    

    
def drop_metapath(data,          
                  metapaths,
                  r: float = 0.1,
                  walks_per_node: int = 1,
                  training: bool = True):

    if torch_cluster is None:
        raise ImportError("`torch_cluster` is not installed.")

    if not training:
        return data
    
    dropped_data = copy.copy(data)
    
    if not isinstance(r, (list, tuple)):
        rs = [r] * len(metapaths[0])
    else:
        for metapath in metapaths:
            assert len(r) == len(metapath)
        rs = r
        
    dropped_data.metapath_dict = {}

    walk_length = 1
    edge_types = data.edge_types  # original edge types

    for j, metapath in enumerate(metapaths):
        for edge_type in metapath:
            assert data._to_canonical(edge_type) in edge_types, f"'{edge_type}' not present"    
            
        start = None
        for i, (edge_type, r) in enumerate(zip(metapath, rs)):
            assert 0 < r <= 1.0
            
            if start is None:
                start = r
            else:
                start = start[: int(start.size(0)*r)]
                
            num_nodes = data[edge_type].size(0)
            
            if data[edge_type].edge_index.size(1) == 0:
                break
            edge_index, path = drop_path(data[edge_type].edge_index, 
                                            r=start,
                                            walks_per_node=walks_per_node,
                                            walk_length=walk_length,
                                            num_nodes=num_nodes,
                                            training=training,
                                            return_dropped=True)
            start = edge_index[1]
            dropped_data[edge_type].edge_index = edge_index
            dropped_data.metapath_dict[edge_type] = path
    return dropped_data