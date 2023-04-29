
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, LayerNorm


from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_layers,
                 dropout):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj_t):
        # Execute conv -> relu -> dropout sequence

        bn = BatchNorm1d(256).to(x.device)
        x = bn(x).to(x.device)

        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_t)
        x = F.relu(x)

        return x