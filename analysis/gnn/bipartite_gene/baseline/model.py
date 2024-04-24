from torch_geometric.nn import GCNConv
from torch.nn import Module


class GCN(Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv = GCNConv(num_features, num_features)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.conv(h, edge_index)
        return h
