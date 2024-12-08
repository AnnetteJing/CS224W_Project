from turtle import forward
import torch
import torch.nn.functional as F

from CS224W_Project import *
from pytorch_geometric_temporal.torch_geometric_temporal.nn.attention import STConv


class STGCNModel(torch.nn.Module):
    """
    https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/main.py
    """
    def __init__(
        self, 
        num_nodes: int,
        num_features: int, 
        hidden_channels: int, 
        temporal_filter_size: int,
        graph_filter_size: int, 
    ):
        """
        num_nodes (V): 
        num_features (F): 
        hidden_channels (C): 
        temporal_filter_size (K_t): 
        graph_filter_size (K_g): 
        """
        super().__init__()
        self.st_conv1 = STConv(
            num_nodes=num_nodes, 
            in_channels=num_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=temporal_filter_size,
            K=graph_filter_size,
        )
        self.st_conv2 = STConv(
            num_nodes=num_nodes, 
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=temporal_filter_size,
            K=graph_filter_size,
        )
        # TODO: Compute W_2 based on temporal_filter_size K_t instead of hard coding it as 4
        self.linear = torch.nn.Linear(4 * hidden_channels, 12)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, (0, 3, 1, 2)) # [B, V, F, W] -> [B, W, V, F]
        h1 = self.st_conv1(x, edge_index=edge_index, edge_weight=edge_weight) # [B, W_1, V, C]
        h2 = self.st_conv2(h1, edge_index=edge_index, edge_weight=edge_weight) # [B, W_2, V, C]
        h2 = torch.permute(h2, (0, 2, 1, 3)).flatten(2, 3) # [B, V, W_2 * C]
        out = self.linear(h2) # [B, V, H]
        return out

