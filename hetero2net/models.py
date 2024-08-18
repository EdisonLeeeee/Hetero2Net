import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero, HeteroConv, GraphConv, Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.module_dict import ModuleDict
from collections import defaultdict

from hetero2net.layers import DisenConv

class EdgeDecoder(nn.Module):
    def __init__(self, hidden_channels, dropout=0):
        super().__init__()
        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, dst, edge_label_index):
        row, col = edge_label_index
        z = src[row] * dst[col]

        z = self.lin1(z).relu()
        z = self.dropout(z)
        z = self.lin2(z)
        return z.view(-1)
    

class HeteroGNN(nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, 
                 out_channels, 
                 dropout: float = 0.,
                 num_layers=2, 
                 bn=True, 
                 project=True):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        
        for i in range(num_layers):
            second_channels = out_channels if (i == num_layers - 1 and not project) else hidden_channels
            if i == 0:
                layer = DisenConv
            else:
                layer = DisenConv
                
            conv = ModuleDict({
                '__'.join(edge_type): layer(-1, second_channels)
                for edge_type in metadata[1]
            })
            batchnorm = nn.ModuleDict(
                {
                    node_type: bn(second_channels)
                    for node_type in metadata[0]
                }
            )
            self.bns.append(batchnorm)
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        
        if project:
            self.lin = nn.Sequential(Linear(-1, out_channels))  
        
        self.edge_decoder = EdgeDecoder(hidden_channels)
        self.out_channels = out_channels
        self.project = project
        
        self.embedding = nn.Embedding(out_channels + 1, in_channels)
        nn.init.xavier_normal_(self.embedding.weight)        
    
    def hetero_conv(self, layer, x_dict, edge_index_dict):
        out_dict = defaultdict(list)
        homo_dict = {}
        hetero_dict = {}
        
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in layer:
                continue

            conv = layer[str_edge_type]

            if src == dst:
                out = conv(x_dict[src], edge_index)
            else:
                out = conv((x_dict[src], x_dict[dst]), edge_index)

            if isinstance(out, tuple):
                out, x_homo, x_hetero = out
                homo_dict[edge_type] = x_homo
                hetero_dict[edge_type] = x_hetero
            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, 'sum')

        return out_dict, homo_dict, hetero_dict
    

    def forward(self, x_dict, edge_index_dict, y_emb=None):
        if y_emb is not None:
            if y_emb.ndim > 1:
                # y_emb [N, C]
                mask = (y_emb != self.out_channels).float().unsqueeze(-1) # [N, C, 1]
                y_emb = self.dropout(self.embedding(y_emb) * mask) # [N, C, F]
                y_emb = y_emb.sum(1) # [N, F]
            else:
                mask = (y_emb != self.out_channels).float().unsqueeze(1)
                y_emb = self.dropout(self.embedding(y_emb) * mask)
            x_dict[self.node_type] = x_dict[self.node_type] + y_emb

        xs = [x_dict[self.node_type]]
        homos = []
        heteros = []
        for i, conv in enumerate(self.convs):
            x_dict, homo_dict, hetero_dict = self.hetero_conv(conv, x_dict, edge_index_dict)
             
            
            if i == len(self.convs) - 1 and not self.project:
                xs.append(x_dict[self.node_type])
            else:
                x_dict = {
                    key: self.dropout(self.activation(self.bns[i][key](x)))
                    for key, x in x_dict.items()
                }
                xs.append(x_dict[self.node_type])
                
            homos.append(homo_dict)
            heteros.append(hetero_dict)   
            
        if self.project:
            x = jumping_knowledge(xs, 'last')
            out = self.lin(x)
        else:
            x = jumping_knowledge(xs, 'last')
            out = x
            
        if self.training:
            return out, homos, heteros
        else:
            return out

        
def jumping_knowledge(xs, mode='last'):
    if mode == 'cat':
        return torch.cat(xs, dim=-1)    
    if mode == 'last':
        return xs[-1]
        
def group(xs, aggr):
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out