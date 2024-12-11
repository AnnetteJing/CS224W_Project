import torch
import torch.nn as nn
import torch.nn.functional as F

from CS224W_Project import *


class ApproxSVDGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(
        self, x: torch.Tensor, svecs_l: torch.Tensor, svecs_r: torch.Tensor, svals: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [B, V, in_channels]. Input features
        svecs_l: [V, K]. Top K left singular vectors of the graph Laplacian
        svecs_r: [V, K]. Top K right singular vectors of the graph Laplacian
        svals: [K,]. Top K singular values of the graph Laplacian
        ---
        out: [B, V, out_channels]. For each batch b = 0, ..., B - 1, 
            out[b] = svecs_l @ diag(svals) @ svecs_l.T @ x[b] + svecs_r @ diag(svals) @ svecs_r.T @ x[b]
        """
        svecs_l, svecs_r, svals = svecs_l.to(x.device), svecs_r.to(x.device), svals.to(x.device)
        batch_size, _, _ = x.size()
        outputs = []
        # Process each batch with SVD components
        for b in range(batch_size):
            # Transform signals to spectral domain
            x_spectral_l = torch.matmul(svecs_l.T, x[b]) # [K, in_channels]
            x_spectral_r = torch.matmul(svecs_r.T, x[b]) # [K, in_channels]
            # Apply spectral filtering
            x_filtered_l = torch.matmul(x_spectral_l, self.weight) # [K, out_channels]
            x_filtered_r = torch.matmul(x_spectral_r, self.weight) # [K, out_channels]
            # Scale by singular values
            x_filtered_l = x_filtered_l * svals.unsqueeze(-1) # [K, out_channels]
            x_filtered_r = x_filtered_r * svals.unsqueeze(-1) # [K, out_channels]
            # Transform back to spatial domain
            x_spatial_l = torch.matmul(svecs_l, x_filtered_l) # [V, out_channels]
            x_spatial_r = torch.matmul(svecs_r, x_filtered_r) # [V, out_channels]
            # Append output of current batch
            outputs.append(x_spatial_l + x_spatial_r)
        # Stack batch outputs
        out = torch.stack(outputs) # [B, V, out_channels]
        return out


class ApproxSVDSpectralGCN(nn.Module):
    def __init__(
        self, 
        num_features: int, 
        hidden_channels: int, 
        num_layers: int, 
        K: int,
    ):
        """
        num_features (F): Number of input features
        hidden_channels: Shared hidden dimension across layers
        num_layers: Number of ApproxSVDGraphConv layers
        K: Top K singular vectors & values would be used to form the graph convolution
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.K = K
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True
        )
        # Multiple spectral conv layers
        self.conv_layers = nn.ModuleList([
            ApproxSVDGraphConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
            ) for _ in range(num_layers)
        ])
        # Final prediction layer
        self.linear = nn.Linear(hidden_channels, 12)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        x: [B, V, F, W]. Input tensor of past observations
        edge_index: 
        edge_weight: 
        """
        batch_size, num_nodes, num_features, timesteps = x.size()
        
        # Process temporal dimension with GRU
        x_temporal = x.transpose(2, 3).contiguous()
        x_temporal = x_temporal.view(batch_size * num_nodes, timesteps, num_features)
        x_temporal, _ = self.gru(x_temporal)
        
        # Get final GRU hidden state
        x = x_temporal[:, -1, :].view(batch_size, num_nodes, self.hidden_channels)
        
        # Save Laplacian matrix SVD if not exists already
        if not hasattr(self, "svd"):
            # Get adjacency matrix without self-loop, A
            adj_mat = torch.sparse_coo_tensor(
                edge_index, edge_weight, (num_nodes, num_nodes)
            ).to_dense() - torch.eye(num_nodes) # [V, V]
            # Compute in-degrees, D = diag(d1, ..., d_V)
            in_deg = adj_mat.sum(axis=1) # [V,]
            # Compute normalized Laplacian with in-degrees & adjacency matrix, L = I - D^{-1/2} A D^{-1/2}
            inv_sqrt_in_deg = torch.where(in_deg > 0, 1 / torch.sqrt(in_deg), 0) # [V,]
            laplacian = torch.eye(num_nodes) - torch.outer(inv_sqrt_in_deg, inv_sqrt_in_deg) * adj_mat # [V, V]
            # SVD of the normalized Laplacian
            U, S, Vh = torch.linalg.svd(laplacian, full_matrices=False)
            self.svd = {
                "left_vecs": U[:, :self.K], 
                "right_vecs": Vh.T[:, :self.K], 
                "vals": S[:self.K]
            }
        
        # Apply multiple spectral conv layers with skip connections
        for conv in self.conv_layers:
            identity = x  # Save for skip connection
            x_conv = conv(x, self.svd["left_vecs"], self.svd["right_vecs"], self.svd["vals"])
            x_conv = F.relu(x_conv)
            x = x_conv + identity  # Skip connection
        
        # Final prediction
        out = self.linear(x)
        
        return out