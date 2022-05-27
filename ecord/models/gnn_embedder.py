import math
import time

import torch
import torch_geometric.nn as gnn
from torch import Tensor
from torch import nn
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_mean
from torch_sparse import SparseTensor, matmul

from ecord.models._base import _DevicedModule


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)

class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        \mathbf{\Theta} \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1`)

    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def _prepare_input(self, x):
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        return x

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, batch=None) -> Tensor:
        """"""
        x = self._prepare_input(x)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight,
                               size=None)
            x = self.rnn(m, x)

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)


class NodeModel(nn.Module):
    def __init__(self, out_channels: int, **kwargs):
        super().__init__()

        self.out_channels = out_channels

        self.msg_snd = torch.nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(),
        )
        self.msg_rec = torch.nn.Sequential(
            nn.Linear(2 * out_channels, 2 * out_channels),
            nn.LeakyReLU(),
        )

        self.rnn = torch.nn.GRUCell(out_channels*2, out_channels, bias=True)

    def _prepare_input(self, x):
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        return x

    def forward(self, x, edge_index, edge_attr, batch):
        """"""
        x = self._prepare_input(x)

        row, col = edge_index  # row/col <--> send/recieve

        msg = scatter_mean(
            self.msg_snd(x)[row] *  edge_attr.view(-1, 1),
            col, dim=0, dim_size=x.size(0)
        )
        x = self.rnn(
            self.msg_rec(
                torch.cat([msg, x], dim=-1)
            ),
            x)

        return x


class GNNEmbedder(nn.Module, _DevicedModule):

    def __init__(
            self,
            dim_in,
            dim_embedding,
            num_layers,
            use_layer_norm,
            return_dummy_graph_embeddings = False,
            device=None,
            out_device='cpu'
    ):
        super().__init__()

        self.embed = nn.Linear(dim_in, dim_embedding)
        self.return_dummy_graph_embeddings = return_dummy_graph_embeddings

        self.num_layers = num_layers
        self.node_model = NodeModel(
                out_channels=dim_embedding,
                num_layers=1,
                aggr="mean",
            )

        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.norm = gnn.LayerNorm(dim_embedding, affine=False)

        self.set_devices(device, out_device)

    def forward(self, x, edge_index, edge_attr, batch, *args, **kwargs):

        x, edge_index, edge_attr, batch = (x.to(self.device),
                                   edge_index.to(self.device),
                                   edge_attr.to(self.device),
                                   batch.to(self.device))

        x_in = x.clone()


        x = self.embed(x)
        if self.use_layer_norm:
            x = self.norm(x)

        for _ in range(self.num_layers):
            x = self.node_model(x, edge_index, edge_attr, batch)
            if self.use_layer_norm:
                x = self.norm(x)

        res = []
        for _ in range(10):

            x = x_in.clone()

            t = time.time()

            x = self.embed(x_in)
            if self.use_layer_norm:
                x = self.norm(x)

            for _ in range(self.num_layers):
                x = self.node_model(x, edge_index, edge_attr, batch)
                if self.use_layer_norm:
                    x = self.norm(x)

            res.append(x[0])

        x = x.to(self.out_device)

        if self.return_dummy_graph_embeddings:
            x = torch.zeros_like(x)

        return x


