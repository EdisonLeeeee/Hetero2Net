from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm, to_torch_csr_tensor
from torch import nn
import torch

def to_sparse_tensor(edge_index, size):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=size
    )

class DisenConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        super().__init__(aggr, **kwargs)


        self.lin_homo = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_hetero = Linear(in_channels[0], out_channels, bias=bias)
        
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_homo.reset_parameters()
        self.lin_hetero.reset_parameters()
        
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        x_homo = self.lin_homo(x[0])
        x_hetero = self.lin_hetero(x[0])
        
        edge_index = to_sparse_tensor(edge_index.flip(0), size=(x[1].size(0), x[0].size(0)))
        
        # propagate_type: (x: OptPairTensor)
        out_homo = self.propagate(edge_index, x=(x_homo, x[1]), size=size)
        
        # propagate_type: (x: OptPairTensor)
        out_hetero = self.propagate(edge_index, x=(x_hetero, x[1]), size=size)

        # # propagate_type: (x: OptPairTensor)
        # out = self.propagate(edge_index, x=x, size=size)
        # out_homo = self.lin_homo(out)
        # out_hetero = self.lin_homo(out)   
        
        out = out_homo + out_hetero
        # out = torch.cat([out_homo, out_hetero], dim=1)
        
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)
            # out = torch.cat([out, self.lin_r(x_r)], dim=1)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, out_homo, out_hetero

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')

    


    
    