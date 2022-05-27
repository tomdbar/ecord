from torch import nn

from ecord.models._base import _DevicedModule


class GNN2RNN(nn.Module, _DevicedModule):

    def __init__(self, dim_node_in, dim_node_out,
                 device=None, out_device='cpu'):
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(dim_node_in, dim_node_out),
        )
        self.set_devices(device, out_device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.node_projection(x)
        return x.to(self.out_device)