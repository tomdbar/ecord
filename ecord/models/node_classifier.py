import torch
from torch import nn

from ecord.models._base import _DevicedModule


class NodeClassifier(nn.Module, _DevicedModule):

    def __init__(self, dim_in, num_probabilty_maps):
        super().__init__()

        self._dim_in = dim_in
        self._num_maps = num_probabilty_maps

        self.heads = []
        for _ in range(self._num_maps):
            head = nn.Sequential(
                nn.Linear(dim_in, 1),
            )
            self.heads.append(head)
        self.heads = nn.ModuleList(self.heads)

    def forward(self, x, head_idx=None):
        if head_idx:
            out = self.heads[head_idx](x)
        else:
            out = torch.cat([head(x) for head in self.heads], dim=-1)
        return out