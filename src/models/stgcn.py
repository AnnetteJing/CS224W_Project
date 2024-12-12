from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from CS224W_Project import *
from pytorch_geometric_temporal.torch_geometric_temporal.nn.attention import STConv


class TemporalGatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # Conv kernel outputs double channels for GLU
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, (1, kernel_size))
        
    def forward(self, x):
        # Input shape: [batch, channels, nodes, time_steps]
        # No padding to maintain causality
        out = self.conv(x)
        # Split channels for GLU
        P, Q = torch.chunk(out, 2, dim=1)
        # GLU operation: P ⊙ σ(Q)
        return P * torch.sigmoid(Q)


class SimpleGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, channels, nodes, time]
        batch_size, channels, num_nodes, time_steps = x.size()
        
        # Normalized Laplacian for symmetrized adjacency matrix
        if not hasattr(self, "laplaican"):
            # Get adjacency matrix
            adj_mat = torch.sparse_coo_tensor(
                edge_index, edge_weight, (num_nodes, num_nodes)
            ).to_dense() # [V, V]
            # Symmetrize the adjacency matrix
            adj_mat_symm = (adj_mat + adj_mat.T) / 2
            # Compute degrees for the symmetrized adjacency matrix
            deg = adj_mat_symm.sum(axis=1) # [V,]
            # Compute normalized Laplacian with in-degrees & adjacency matrix, L = I - D^{-1/2} A D^{-1/2}
            inv_sqrt_deg = torch.where(deg > 0, 1 / torch.sqrt(deg), 0) # [V,]
            self.laplacian = torch.eye(num_nodes) - torch.outer(inv_sqrt_deg, inv_sqrt_deg) * adj_mat_symm # [V, V]
        
        # Reshape input for multiplication
        x = x.permute(0, 3, 2, 1)  # [batch, time, nodes, channels]
        x = x.reshape(-1, num_nodes, channels)  # [batch*time, nodes, channels]
        
        # Graph convolution
        out = torch.matmul(self.laplacian, x)  # [batch*time, nodes, channels]
        out = torch.matmul(out, self.weight)  # [batch*time, nodes, out_channels]
        
        # Reshape back
        out = out.reshape(batch_size, time_steps, num_nodes, -1)  # [batch, time, nodes, channels]
        out = out.permute(0, 3, 2, 1)  # [batch, channels, nodes, time]
        
        return out


class STGCNModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 64
    ):
        """
        https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/main.py
        """
        super().__init__()
        # First ST-Conv Block
        self.temp1_1 = TemporalGatedConv(num_features, hidden_channels, kernel_size=3)
        self.graph1 = SimpleGraphConv(hidden_channels, hidden_channels)
        self.temp1_2 = TemporalGatedConv(hidden_channels, hidden_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        # Second ST-Conv Block
        self.temp2_1 = TemporalGatedConv(hidden_channels, hidden_channels, kernel_size=3)
        self.graph2 = SimpleGraphConv(hidden_channels, hidden_channels)
        self.temp2_2 = TemporalGatedConv(hidden_channels, hidden_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        # Output layers
        self.final_conv = TemporalGatedConv(hidden_channels, hidden_channels, kernel_size=3)
        self.fc = nn.Linear(hidden_channels, 12)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, nodes, features, time_steps]
        x = x.permute(0, 2, 1, 3)  # -> [batch, features, nodes, time_steps]
        # First ST-Conv Block
        x = self.temp1_1(x)
        x = F.relu(self.graph1(x, edge_index, edge_weight))
        x = self.temp1_2(x)
        x = self.bn1(x)
        # Second ST-Conv Block
        x = self.temp2_1(x)
        x = F.relu(self.graph2(x, edge_index, edge_weight))
        x = self.temp2_2(x)
        x = self.bn2(x)
        # Output processing
        x = self.final_conv(x)
        x = x[..., -1]  # Take last time step [batch, channels, nodes]
        x = x.transpose(1, 2)  # [batch, nodes, channels]
        out = self.fc(x)  # [batch, nodes, prediction_steps]
        return out

