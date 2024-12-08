import torch
import torch.nn.functional as F

from CS224W_Project import *
from pytorch_geometric_temporal.torch_geometric_temporal.nn.recurrent import DCRNN


class DCRNNModel(torch.nn.Module):
    """
    https://github.com/liyaguang/DCRNN/blob/128dc26e265491aad88bc4cb67d6ab2d2193532b/dcrnn_train.py
    """
    def __init__(self, num_features: int, hidden_channels: int, filter_size: int):
        super().__init__()
        self.dcrnn1 = DCRNN(in_channels=num_features, out_channels=hidden_channels, K=filter_size)
        self.dcrnn2 = DCRNN(in_channels=hidden_channels, out_channels=hidden_channels, K=filter_size)
        self.linear = torch.nn.Linear(hidden_channels, 12)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, features, timesteps = x.size()
        h1 = None
        h2 = None
        for t in range(timesteps):
            current_x = x[:, :, :, t]
            current_x = current_x.reshape(batch_size * num_nodes, features)
            h1 = self.dcrnn1(current_x, edge_index, edge_weight, h1)
            h1 = F.relu(h1)
            h2 = self.dcrnn2(h1, edge_index, edge_weight, h2)
            h2 = F.relu(h2)
        out = self.linear(h2)
        return out.reshape(batch_size, num_nodes, -1)
