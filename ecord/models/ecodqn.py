import torch
from torch import nn
from torch_scatter import scatter

from ecord.models._base import _DevicedModule


class ECODQN_layer(nn.Module):

    def __init__(self, dim_embedding):
        super().__init__()

        # eq (6)
        self.message = nn.Sequential(
            nn.Linear(2 * dim_embedding, dim_embedding),
            nn.ReLU()
        )

        # eq (7)
        self.update = nn.Sequential(
            nn.Linear(2 * dim_embedding, dim_embedding),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, x_agg_emb, *args, **kwargs):
        col, row = edge_index

        x_agg = scatter(
            edge_attr.unsqueeze(-1) * x[col],
            row,
            reduce='mean',
            dim=0,
            dim_size=len(x)
        )
        m = self.message(
            torch.cat([x_agg, x_agg_emb], dim=-1)
        )
        x = self.update(torch.cat([x, m], dim=-1))
        return x


class GNN_ECODQN(nn.Module, _DevicedModule):

    def __init__(
            self,
            dim_in,
            dim_embedding,
            num_layers,
            device=None,
            out_device='cpu'
    ):
        super().__init__()

        # eq (4)
        self.embed_node = nn.Sequential(
            nn.Linear(dim_in, dim_embedding),
            nn.ReLU()
        )

        # eq (5) inner
        self.embed_node_and_edge = nn.Sequential(
            nn.Linear(dim_in + 1, dim_embedding - 1),
            nn.ReLU()
        )

        # eq (5) outer
        self.embed_agg_nodes_and_degree = nn.Sequential(
            nn.Linear(dim_embedding, dim_embedding),
            nn.ReLU()
        )

        # eq (6-7)
        self.layers = nn.ModuleList([
            ECODQN_layer(dim_embedding) for _ in range(num_layers)
        ])

        # eq (8) inner
        self.agg_nodes = nn.Sequential(
            nn.Linear(dim_embedding, dim_embedding),
            # nn.ReLU()
        )

        # eq (8) outer
        self.readout = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*dim_embedding, 1),
        )



        self.set_devices(device, out_device)

    def _expand_as_over_tradj(self, trg, src):
        assert src.dim() == 3, "Source should have shape [nodes, tradj, -1]"
        _, t, _ = src.size()
        while trg.dim() < src.dim():
            trg = trg.unsqueeze(-1)
        trg = trg.repeat(1,t,1)
        return trg

    def forward(self, x, edge_index, edge_attr, batch, degree, *args, **kwargs):

        x, edge_index, edge_attr, batch, degree = [
            t.to(self.device) for t in [x, edge_index, edge_attr, batch, degree]
        ]
        col, row = edge_index


        x_emb = self.embed_node(x)

        node_edge = scatter(
            torch.cat([x[col], self._expand_as_over_tradj(edge_attr, x)], dim=-1),
            row,
            reduce='mean',
            dim=0,
            dim_size = len(batch)
        )
        node_edge = self.embed_node_and_edge(node_edge)

        degree_norm = scatter(degree, batch, reduce='max', dim=0)[batch]
        degree_norm = self._expand_as_over_tradj(degree_norm, node_edge)
        x_agg_emb = self.embed_agg_nodes_and_degree(
            torch.cat([node_edge, degree_norm], dim=-1)
        )

        for layer in self.layers:
            x_emb = layer(x_emb, edge_index, edge_attr, x_agg_emb)

        g_agg = scatter(
            x_emb, batch, reduce='mean', dim=0,
        )
        g_agg = self.agg_nodes(g_agg)

        inp = torch.cat([g_agg[batch], x_emb], dim=-1)
        q_vals = self.readout(inp)

        q_vals = q_vals.to(self.out_device)

        # [nodes, tradj, 1] --> [nodes, tradj]
        return q_vals.squeeze(-1)


