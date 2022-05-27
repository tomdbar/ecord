import math
from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch_geometric.nn as gnn
from torch import nn
from torch_scatter import scatter_mean

from ecord.models._base import _DevicedModule


class RNNOutput(Enum):
    DOT = 0
    MLP_Q_VALUE = 1
    MLP_ACTION_STATE = 2
    MLP_ADVANTAGE_STATE = 3
    MLP_V_VALUE = 4

class _RNNDecoderBase(nn.Module, _DevicedModule, ABC):

    def __init__(self,
                 dim_node_in,
                 dim_msg_in,
                 dim_node_embedding,
                 dim_internal_embedding,
                 num_layers=2,
                 learn_graph_embedding=False,
                 learn_scale=False,
                 project_rnn_to_gnn=False,
                 output_type = RNNOutput.DOT,
                 output_activation = torch.tanh,
                 return_dummy_hidden_state = False, # For ablations only.
                 device=None,
                 out_device="cpu"):

        super().__init__()

        self.dim_node_in = dim_node_in
        self.dim_msg_in = dim_msg_in
        self.dim_msg_out = 64
        self.dim_node_embedding = dim_node_embedding
        self.dim_internal_embedding = dim_internal_embedding
        self.learn_graph_embedding = learn_graph_embedding
        self.learn_scale = learn_scale
        self.num_layers = num_layers
        self.project_rnn_to_gnn = project_rnn_to_gnn
        self.return_dummy_hidden_state = return_dummy_hidden_state

        if self.learn_graph_embedding:
            d_pool_in = self.dim_node_in
            gate_nn = nn.Sequential(
                nn.Linear(d_pool_in, 1)
            )
            feat_nn = nn.Sequential(
                nn.Linear(d_pool_in, 2 * d_pool_in),
                nn.LeakyReLU(),
                nn.Linear(2 * d_pool_in, dim_internal_embedding)
            )
            self.graph_pool = gnn.GlobalAttention(gate_nn, feat_nn)

            dim_graph = 2 * dim_internal_embedding

        else:
            dim_graph = dim_internal_embedding

        self.output_type = output_type

        if self.output_type is RNNOutput.DOT:
            self._dim_dot = 2 * self.dim_node_embedding
            self.lin_node = nn.Sequential(
                nn.Linear(self.dim_node_embedding, self._dim_dot, bias=False),
            )
            self.lin_graph = nn.Sequential(
                nn.Linear(dim_graph, self._dim_dot, bias=False),
            )

        else:
            # self.output type in [RNNOutput.MLP_V_VALUE, RNNOutput.MLP_Q_VALUE,
            # RNNOutput.MLP_ACTION_STATE, RNNOutput.MLP_ADVANTAGE_STATE]
            d = self.dim_node_embedding
            if self.project_rnn_to_gnn:
                d_state = self.dim_node_embedding
                self.proj_rnn_to_gnn = nn.Sequential(
                    nn.Tanh(),
                    nn.Linear(self.dim_internal_embedding, self.dim_internal_embedding//2),
                    nn.LayerNorm(self.dim_internal_embedding//2), nn.LeakyReLU(),
                    nn.Linear(self.dim_internal_embedding//2, self.dim_node_embedding),
                    nn.LayerNorm(self.dim_node_embedding), nn.LeakyReLU(),
                )
            else:
                # d = self.dim_internal_embedding + self.dim_node_embedding
                d_state = self.dim_internal_embedding
                self.proj_rnn_to_gnn = nn.LayerNorm(self.dim_internal_embedding)
            d += d_state

            if self.learn_graph_embedding:
                # [hidden_state, dim_node, graph_embedding]
                d += self.dim_internal_embedding
            if self.output_type is RNNOutput.MLP_V_VALUE:
                self.val_out = nn.Sequential(
                    nn.Linear(self.dim_internal_embedding, self.dim_internal_embedding),
                    nn.LayerNorm(self.dim_internal_embedding // 2), nn.LeakyReLU(),
                    nn.Linear(self.dim_internal_embedding, 1),
                )

            elif self.output_type in [RNNOutput.MLP_ACTION_STATE, RNNOutput.MLP_ADVANTAGE_STATE]:
                self.val_out = nn.Sequential(
                    nn.Tanh(),
                    nn.Linear(d_state, d_state),
                    nn.LeakyReLU(),
                    nn.Linear(d_state, 1)
                )

            self.mlp_out = nn.Sequential(
                nn.Linear(d, d),
                nn.LayerNorm(d), nn.LeakyReLU(),
                nn.Linear(d, 1)
            )

        self.msg_in = nn.Sequential(
            nn.Linear(self.dim_msg_in, self.dim_msg_out),
            nn.LeakyReLU(),
        )

        self.graph_rnn = self._init_graph_rnn(self.num_layers)

        self.output_activation = output_activation

        if self.learn_scale:
            # self.beta = torch.nn.Parameter(data=torch.FloatTensor([1]), requires_grad=True)
            self.scale_net = nn.Sequential(
                nn.Linear(self.dim_internal_embedding, 1)
            )

        self.set_devices(device, out_device)

    @abstractmethod
    def _init_graph_rnn(self):
        raise NotImplementedError

    @abstractmethod
    def init_embeddings(self, node_features, batch, num_tradj):
        '''
        node_features : [graph * num_nodes_max, dim_embedding]
        '''
        raise NotImplementedError

    @abstractmethod
    def update_embeddings(self, update_embeddings):
        '''
        update_embeddings : [num_graphs, num_tradj, dim_msg_in]
        '''
        raise NotImplementedError

    @abstractmethod
    def get_internal_state(self):
        raise NotImplementedError

    @abstractmethod
    def set_internal_state(self):
        raise NotImplementedError

    def _get_current_embeddings(self, *args, **kwargs):
        '''
        Returns graph_embeddings of shape:
            graph_embeddings : [num_graphs, num_tradj, dim]
        '''
        if self.hidden_embeddings is None:
            raise Exception("Embeddings must be set by init_embeddings(...) before forward call.")
        else:
            if type(self.hidden_embeddings) is list:
                embeddings = self.hidden_embeddings[-1]
            else:
                embeddings = self.hidden_embeddings

        if self.learn_graph_embedding:
            embeddings = torch.cat([embeddings, self.graph_embeddings], dim=-1)

        if self.return_dummy_hidden_state:
            # Return zeros, essentially ablating the RNN.
            embeddings = torch.zeros_like(embeddings)

        return embeddings

    def _get_scale(self):
        if not self.learn_scale:
            return 1
        else:
            # return self.beta
            # self._get_current_embeddings() --> [num_graphs, num_tradj, dim]
            # self.scale_net(...) --> [num_graphs, num_tradj, 1]
            return self.scale_net(self._get_current_embeddings())

    def forward(self, embeddings, batch=None, out_device=None):
        '''
        Takes embeddings with shape:
            embeddings : [num_graphs, num_nodes_max, num_tradj, dim_embeddings] if batch is None
            embeddings : [num_nodes_in_batch, num_tradj, dim_embeddings] otherwise

        and returns attention weights to every node:
            out :  [num_graphs, num_tradj, num_nodes_max]  if batch is None
            out:   [num_nodes_in_batch, num_tradj] otherwise

        It is assumed that masking of this attention is handled outside of the
        decoder class.
        '''
        if batch is not None:
            batch = batch.to(self.device)

        if self.output_type is RNNOutput.DOT:
            out = self._forward_dot(embeddings, batch)
        elif self.output_type is RNNOutput.MLP_V_VALUE:
            out = self._forward_value_with_agg(embeddings, batch)
        elif self.output_type is RNNOutput.MLP_Q_VALUE:
            out = self._forward_value(embeddings, batch)
        else:
            out = self._forward_action_value(embeddings, batch)
        if self.output_activation is not None:
            out = self.output_activation(out)

        if out.dim() > 2 and out.size(-1)==1:
            # [num nodes, num_tradj, 1] --> [num nodes, num_tradj]
            out.squeeze_(-1)

        if self.learn_scale:
            # scale: [num_nodes_in_batch, num_tradj]
            scale = self._get_scale()
            if out.dim() > 2:
                # scale: [num_nodes_in_batch, num_tradj, 1]
                scale.unsqueeze_(-1)
            out = scale * out

        if out_device is None:
            out_device = self.out_device
        return out.to(out_device)

    def _forward_dot(self, embeddings, batch=None):
        internal_embeddings = self._get_current_embeddings()
        if batch is None:
            n = embeddings.size(1)
            attn = torch.einsum('gtni, gtni -> gtn',
                                self.lin_graph(internal_embeddings).unsqueeze(2).repeat(1, 1, n, 1),
                                self.lin_node(embeddings.permute(0, 2, 1, 3))
                                )
        else:
            g = self.lin_graph(internal_embeddings)
            n = self.lin_node(embeddings)
            # attn : [num_nodes, num_tradj]
            attn = (n * g[batch]).sum(-1)

        return attn / math.sqrt(self._dim_dot)

    def _forward_value_with_agg(self, embeddings, batch=None):
        if batch is None:
            raise NotImplementedError()

        internal_embeddings = self._get_current_embeddings()

        val = self.val_out(internal_embeddings)

        inp = torch.cat([
            internal_embeddings[batch],
            embeddings
        ], dim=-1)

        action_val = self.mlp_out(inp)

        out = action_val + val[batch]

        return out

    def _forward_value(self, embeddings, batch=None):
        internal_embeddings = self._get_current_embeddings()

        inp = torch.cat([
            self.proj_rnn_to_gnn(internal_embeddings)[batch],
            embeddings
        ], dim=-1)

        out = self.mlp_out(inp)

        return out

    def _forward_action_value(self, embeddings, batch=None):
        if batch is None:
            raise NotImplementedError()

        internal_embeddings = self.proj_rnn_to_gnn(self._get_current_embeddings())

        inp = torch.cat([
            internal_embeddings[batch],
            embeddings
        ], dim=-1)

        action_val = self.mlp_out(inp)

        if self.output_type in [RNNOutput.MLP_ACTION_STATE, RNNOutput.MLP_ADVANTAGE_STATE]:
            # Calculate state value from graph embeddings.
            state_val = self.val_out(internal_embeddings)
            if self.output_type is RNNOutput.MLP_ADVANTAGE_STATE:
                # Convert action value to advantage value.
                action_val = action_val - scatter_mean(action_val, batch, dim=0)[batch]
            out = action_val + state_val[batch]
        else:
            # Use action values as output.
            out = action_val

            out.squeeze_(-1)

        return out

class GRUDecoder(_RNNDecoderBase):

    def _init_graph_rnn(self, num_layers=1):
        graph_rnns = []
        for i in range(num_layers):
            graph_rnn = nn.GRUCell(
                input_size=64 if i == 0 else self.dim_internal_embedding,
                hidden_size=self.dim_internal_embedding,
                bias=True
            )
            graph_rnns.append(graph_rnn)
        return nn.ModuleList(graph_rnns)

    def get_internal_state(self):
        def _pack_embeddings(embeddings):
            if type(embeddings) is list:
                embeddings = torch.cat(embeddings, dim=-1)
            return embeddings

        state = {
            'hidden_embeddings': _pack_embeddings(self.hidden_embeddings),
        }
        if self.learn_graph_embedding:
            state['graph_embeddings'] = _pack_embeddings(self.graph_embeddings)

        return state

    def set_internal_state(self, hidden_embeddings, graph_embeddings=None):
        def _unpack_embeddings(embeddings):
            if torch.is_tensor(embeddings):
                embeddings = list(embeddings.chunk(self.num_layers, -1))
            return embeddings

        self.hidden_embeddings = _unpack_embeddings(hidden_embeddings)

        if graph_embeddings is None and self.learn_graph_embedding:
            if self.hidden_embeddings is not None:
                self.graph_embeddings = torch.zeros_like(self.hidden_embeddings)
            else:
                self.graph_embeddings = None
        elif self.learn_graph_embedding:
            self.graph_embeddings = graph_embeddings

    def init_embeddings(self, node_features, batch, num_tradj):
        '''
        node_features : [graph * num_nodes_max, dim_embedding]
        '''
        if self.learn_graph_embedding:
            self.graph_embeddings = self.graph_pool(node_features, batch.tg.batch.to(node_features.device))
            # graph_embeddings --> [num_graphs, num_tradj, dim_graph_embedding]
            self.graph_embeddings = self.graph_embeddings.unsqueeze(1).repeat(1, num_tradj, 1)
        self.hidden_embeddings = [torch.zeros(
            batch.tg.batch.max() + 1, num_tradj, self.dim_internal_embedding,
            device=node_features.device
        )  for _ in range(self.num_layers)]

    def update_embeddings(self, update_embeddings):
        '''
        update_embeddings : [num_graphs, num_tradj, dim_msg_in]
        '''
        msg = self.msg_in(update_embeddings).view(-1, self.dim_msg_out)

        for i in range(self.num_layers):
            hidden_embeddings = self.graph_rnn[i](
                msg,
                self.hidden_embeddings[i].view(-1, self.dim_internal_embedding)
            )
            msg = hidden_embeddings
            self.hidden_embeddings[i] = hidden_embeddings.view(*self.hidden_embeddings[i].shape)

class LSTMDecoder(_RNNDecoderBase):

    def _init_graph_rnn(self, num_layers):
        graph_rnns = []
        for i in range(num_layers):
            graph_rnn = nn.LSTMCell(
                input_size=self.dim_msg_in if i==0 else self.dim_internal_embedding,
                hidden_size=self.dim_internal_embedding,
                bias=True
            )
            graph_rnns.append(graph_rnn)
        return nn.ModuleList(graph_rnns)

    def get_internal_state(self):
        def _pack_embeddings(embeddings):
            if type(embeddings) is list:
                embeddings = torch.cat(embeddings, dim=-1)
            return embeddings

        state = {
            'hidden_embeddings': _pack_embeddings(self.hidden_embeddings),
            'cell_embeddings': _pack_embeddings(self.cell_embeddings),
        }
        if self.learn_graph_embedding:
            state['graph_embeddings'] = _pack_embeddings(self.graph_embeddings)

        return state

    def set_internal_state(self, hidden_embeddings, cell_embeddings=None, graph_embeddings=None):

        def _unpack_embeddings(embeddings):
            if torch.is_tensor(embeddings):
                embeddings = list(embeddings.chunk(self.num_layers, -1))
            return embeddings

        self.hidden_embeddings = _unpack_embeddings(hidden_embeddings)

        if cell_embeddings is not None:
            self.cell_embeddings = _unpack_embeddings(cell_embeddings)
        else:
            if self.hidden_embeddings is not None:
                self.cell_embeddings = [torch.zeros_like(self.hidden_embeddings) for _ in range(self.num_layers)]
            else:
                self.cell_embeddings = None

        if graph_embeddings is None and self.learn_graph_embedding:
            if self.hidden_embeddings is not None:
                self.graph_embeddings = torch.zeros_like(self.hidden_embeddings)
            else:
                self.graph_embeddings = None
        elif self.learn_graph_embedding:
            self.graph_embeddings = graph_embeddings

    def init_embeddings(self, node_features, batch, num_tradj):
        '''
        node_features : [graph * num_nodes_max, dim_embedding]
        '''
        if self.learn_graph_embedding:
            self.graph_embeddings = self.graph_pool(node_features, batch.tg.batch.to(node_features.device))
            # graph_embeddings --> [num_graphs, num_tradj, dim_graph_embedding]
            self.graph_embeddings = self.graph_embeddings.unsqueeze(1).repeat(1, num_tradj, 1)
        self.hidden_embeddings = [torch.zeros(
            batch.tg.batch.max() + 1, num_tradj, self.dim_internal_embedding,
            device=node_features.device
        )  for _ in range(self.num_layers)]

        self.cell_embeddings = [torch.zeros_like(self.hidden_embeddings[i])  for i in range(self.num_layers)]

    def update_embeddings(self, update_embeddings):
        '''
        update_embeddings : [num_graphs, num_tradj, dim_msg_in]
        '''
        msg = self.msg_in(update_embeddings).view(-1, self.self.dim_msg_out)

        for i in range(self.num_layers):
            hidden_embeddings, cell_embeddings = self.graph_rnn[i](
                msg,
                (self.hidden_embeddings[i].view(-1, self.dim_internal_embedding), self.cell_embeddings[i].view(-1, self.dim_internal_embedding))
            )
            msg = hidden_embeddings
            self.hidden_embeddings[i] = hidden_embeddings.view(*self.hidden_embeddings[i].shape)
            self.cell_embeddings[i] = cell_embeddings.view(*self.cell_embeddings[i].shape)