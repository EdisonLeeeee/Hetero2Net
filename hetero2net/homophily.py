import torch
from torch_geometric.utils import degree, scatter

def attribute_homophily(edge_index, x):
    row, col = edge_index
    src_degree = degree(row, num_nodes=x.size(0)).clamp(min=1)    
    dst_degree = degree(col, num_nodes=x.size(0)).clamp(min=1)
    out = []
    for r, c in zip(row, col):
        energy = x[r]/src_degree[r].sqrt() - x[c] / dst_degree[c].sqrt()
        energy = energy.norm(p=2).square().item()
        out.append(energy)
    out = torch.tensor(out) / 4
    return out.mean().item()
    
def label_homophily(edge_index, y):
    row, col = edge_index
    y = y.squeeze(-1) if y.dim() > 1 else y
    out = torch.zeros(row.size(0), device=row.device)
    if y.dim() > 1:
        out[(y[row] == y[col]).all(dim=1)] = 1.
    else:
        out[y[row] == y[col]] = 1.
    return out.mean().item()

def attribute_homophily_matrix(edge_index, x):
    row, col = edge_index
    src_degree = degree(row, num_nodes=x.size(0)).clamp(min=1)    
    dst_degree = degree(col, num_nodes=x.size(0)).clamp(min=1)
    out = []
    for r, c in zip(row, col):
        energy = x[r]/src_degree[r].sqrt() - x[c]/dst_degree[c].sqrt()
        energy = energy.norm(p=2).square().item()
        out.append(energy)
    out = torch.tensor(out) / 4
    out = scatter(out, col, 0, dim_size=x.size(0), reduce='mean')
    return out
    
def label_homophily_matrix(edge_index, y):
    row, col = edge_index
    y = y.squeeze(-1) if y.dim() > 1 else y
    out = torch.zeros(row.size(0), device=row.device)
    if y.dim() > 1:
        out[(y[row] == y[col]).all(dim=1)] = 1.
    else:
        out[y[row] == y[col]] = 1.
    out = scatter(out, col, 0, dim_size=y.size(0), reduce='mean')
    return out
    
    
def get_metapaths(start, edge_types, metapath, results, hop, first_start=None):
    if hop < 1:
        if first_start is None:
            results.append(metapath)
        elif start == first_start:
            results.append(metapath)
        return
    for edge_type in edge_types:
        src, rel, dst = edge_type    
        if src == dst: continue
        if src == start:
            get_metapaths(dst, edge_types, metapath + [(src, dst)], results, hop-1, first_start)
    return results