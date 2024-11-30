import os
import sys

sys.path.append('/home/dlvm/CS224w/pytorch_geometric_temporal')
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_geometric_temporal.torch_geometric_temporal.dataset import PemsBayDatasetLoader, METRLADatasetLoader
from pytorch_geometric_temporal.torch_geometric_temporal.nn.recurrent import DCRNN
from tqdm import tqdm
import numpy as np
from src.utils.data_utils import TimeSeriesDataset
from src.utils.metrics import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Helper function for masked MAE loss
def masked_mae_loss(y_pred, y_true, null_val=np.nan):
    if torch.isnan(torch.tensor(null_val)):
        mask = ~torch.isnan(y_true)
    else:
        mask = (y_true != null_val)
    mask = mask.float()
    mask = mask.to(y_pred.device)
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    return torch.mean(loss)

def create_spectral_conv_svd(L):
    """
    Create the eigenvalues and eigenvectors of the graph Laplacian L using SVD.
    Handles batched input.
    
    Parameters:
    L (torch.Tensor): Batch of Graph Laplacian matrices [batch_size, num_nodes, num_nodes]
    
    Returns:
    eigenvals (torch.Tensor): Eigenvalues of the Laplacian [batch_size, num_nodes]
    eigenvecs (torch.Tensor): Eigenvectors of the Laplacian [batch_size, num_nodes, num_nodes]
    """
    batch_size = L.size(0)
    eigenvals = []
    eigenvecs = []
    
    # Process each batch item separately
    for i in range(batch_size):
        u, s, v = torch.linalg.svd(L[i], full_matrices=False)
        eigenvals.append(s)
        eigenvecs.append(u)
    
    return torch.stack(eigenvals), torch.stack(eigenvecs)

class SpectralGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, directed_type='svd'):
        super(SpectralGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.directed_type = directed_type
        
        self.theta = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta)

    def forward(self, x, L):
        """
        x: Input features [batch_size, num_nodes, in_channels]
        L: Batch of Graph Laplacians [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.size()
        
        # Initialize list to store Chebyshev polynomials for each batch
        Tk_batched = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Chebyshev polynomials for current batch
            Tk = [torch.eye(num_nodes, device=x.device), L[b]]
            
            for k in range(2, self.K):
                Tk.append(2 * L[b] @ Tk[-1] - Tk[-2])
            
            Tk_batched.append(torch.stack(Tk))  # [K, num_nodes, num_nodes]
        
        Tk_batched = torch.stack(Tk_batched)  # [batch_size, K, num_nodes, num_nodes]
        
        # Reshape x for batch multiplication
        x = x.unsqueeze(1)  # [batch_size, 1, num_nodes, in_channels]
        
        # Initialize output tensor
        out = torch.zeros(batch_size, num_nodes, self.out_channels, device=x.device)
        
        # Perform spectral convolution for each order k
        for k in range(self.K):
            # [batch_size, num_nodes, num_nodes] @ [batch_size, num_nodes, in_channels]
            conv = torch.matmul(Tk_batched[:, k], x.squeeze(1))
            # [batch_size, num_nodes, in_channels] @ [in_channels, out_channels]
            out += torch.matmul(conv, self.theta[k])
        
        return out

class SpectralGCN(torch.nn.Module):
    def __init__(self, node_features, num_timesteps, device, directed_type='svd'):
        super(SpectralGCN, self).__init__()
        self.node_features = node_features
        self.num_timesteps = num_timesteps
        self.spectral_conv = SpectralGraphConv(
            in_channels=node_features * num_timesteps,
            out_channels=32,
            K=3,
            directed_type=directed_type
        )
        self.device = device
        self.linear = torch.nn.Linear(32, 12)  # Output 12 timesteps

    def forward(self, x, edge_index, edge_weight):
        """
        x: Input tensor [batch_size, num_nodes, node_features, num_timesteps]
        edge_index: Edge indices [2, num_edges]
        edge_weight: Edge weights [num_edges]
        """
        batch_size, num_nodes, features, timesteps = x.size()
        
        # Reshape input to combine features and timesteps
        x = x.reshape(batch_size, num_nodes, features * timesteps)
        
        # Convert edge_index and weight to batched adjacency matrices
        A = torch.sparse_coo_tensor(
            edge_index, edge_weight, (num_nodes, num_nodes)
        ).to_dense().to(self.device)
        
        # Create batched Laplacian matrices
        A_batched = A.unsqueeze(0).expand(batch_size, -1, -1)
        D_batched = torch.diag_embed(A_batched.sum(dim=2))
        D_sqrt_inv = torch.linalg.inv(torch.sqrt(D_batched))
        L_batched = torch.eye(num_nodes, device=self.device).unsqueeze(0).expand(batch_size, -1, -1) - \
                   torch.matmul(torch.matmul(D_sqrt_inv, A_batched), D_sqrt_inv)
        
        # Apply spectral convolution
        h = self.spectral_conv(x, L_batched)  # [batch_size, num_nodes, 32]
        h = F.relu(h)
        h = self.linear(h)  # [batch_size, num_nodes, 12]
        
        return h
class Evaluator:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.total_samples = 0
        self.metrics = {
            'rmse': {'15': 0, '30': 0, '60': 0},
            'mae': {'15': 0, '30': 0, '60': 0},
            'mape': {'15': 0, '30': 0, '60': 0}
        }
    
    def update_metrics(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        self.total_samples += batch_size

        horizons = {'15': 2, '30': 5, '60': 11}
        
        for name, idx in horizons.items():
            y_true_t = y_true[:, :, 0, idx] if len(y_true.shape) == 4 else y_true[:, :, idx]
            y_pred_t = y_pred[:, :, idx]
            
            self.metrics['rmse'][name] += torch.sum((y_pred_t - y_true_t)**2)
            self.metrics['mae'][name] += torch.sum(torch.abs(y_pred_t - y_true_t))
            
            mask = y_true_t != 0
            self.metrics['mape'][name] += torch.sum(
                torch.abs((y_pred_t[mask] - y_true_t[mask]) / y_true_t[mask])
            )

    def get_final_metrics(self):
        total_points = self.total_samples * self.num_nodes
        results = {}
        
        for metric in self.metrics:
            results[metric] = {}
            for horizon in self.metrics[metric]:
                value = self.metrics[metric][horizon]
                if metric == 'rmse':
                    value = torch.sqrt(value / total_points)
                else:
                    value = value / total_points
                if metric == 'mape':
                    value = value * 100
                results[metric][horizon] = value.item()
                
        return results

    def print_metrics(self):
        results = self.get_final_metrics()
        
        print("--- RMSE ---")
        print(f"15-min ahead: {results['rmse']['15']:.4f}")
        print(f"30-min ahead: {results['rmse']['30']:.4f}")
        print(f"60-min ahead: {results['rmse']['60']:.4f}")
        print(f"Average across all horizons: {sum(results['rmse'].values())/3:.4f}")
        
        print("\n--- MAE ---")
        print(f"15-min ahead: {results['mae']['15']:.4f}")
        print(f"30-min ahead: {results['mae']['30']:.4f}")
        print(f"60-min ahead: {results['mae']['60']:.4f}")
        print(f"Average across all horizons: {sum(results['mae'].values())/3:.4f}")
        
        print("\n--- MAPE ---")
        print(f"15-min ahead: {results['mape']['15']:.4f}")
        print(f"30-min ahead: {results['mape']['30']:.4f}")
        print(f"60-min ahead: {results['mape']['60']:.4f}")
        print(f"Average across all horizons: {sum(results['mape'].values())/3:.4f}")

def main():
    # Initialize dataset and model
    df = TimeSeriesDataset(PemsBayDatasetLoader(), batch_size=16)
    epochs = 100
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    model = SpectralGCN(node_features=2, num_timesteps=12, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150], gamma=0.1
    )
    grad_accumulation_steps = 2

    # Training loop
    train_dataset = df.data_splits["train"].batches
    valid_dataset = df.data_splits["valid"].batches
    device_context = torch.amp.autocast("cuda", enabled=(device.type == "cuda"))

    for epoch in tqdm(range(epochs)):
        # Train
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for snapshot in tqdm(train_dataset):
            with device_context:
                y_hat = model(
                    x=snapshot.x.to(device),
                    edge_index=df.edge_index.to(device),
                    edge_weight=df.edge_attr.to(device)
                ).to(device)
                y_norm = df.scaler.normalize(snapshot.y, feature_idx=0).to(device)
                batch_train_loss = masked_mae_loss(y_pred=y_hat, y_true=y_norm)
                batch_train_loss = batch_train_loss / grad_accumulation_steps

            batch_train_loss.backward()
            train_loss += batch_train_loss.item()

        if (epoch + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad(), device_context:
            for snapshot in valid_dataset:
                y_hat = model(
                    x=snapshot.x.to(device),
                    edge_index=df.edge_index.to(device),
                    edge_weight=df.edge_attr.to(device)
                )
                y_norm = df.scaler.normalize(snapshot.y, feature_idx=0).to(device)
                batch_valid_loss = masked_mae_loss(y_pred=y_hat, y_true=y_norm)
                valid_loss += batch_valid_loss.item()

        valid_loss /= len(valid_dataset)
        print(f"Epoch {epoch}: Train loss = {train_loss:4f}, Valid loss = {valid_loss:4f}")

    # Evaluation
    model.eval()
    evaluator = Evaluator(num_nodes=df.num_nodes)
    test_dataset = df.data_splits["test"].batches
    test_loss = 0

    with torch.no_grad(), device_context:
        for x, y in tqdm(df.data_splits["test"].batches):
            y_hat = model(
                x=x.to(device),
                edge_index=df.edge_index.to(device),
                edge_weight=df.edge_attr.to(device)
            )
            y_norm = df.scaler.normalize(y.to(device), feature_idx=0).to(device)
            batch_test_loss = masked_mae_loss(y_hat, y_norm)
            test_loss += batch_test_loss.item()

            y_hat = df.scaler.unnormalize(y_hat, feature_idx=0).to(device)
            #print("eval y shape", y.shape)
            #print("eval y_hat shape", y_hat.shape)
            evaluator.update_metrics(y.to(device), y_hat)
            torch.cuda.empty_cache()

    test_loss /= len(test_dataset)
    print("\nTest Results:")
    print(f"Average Loss: {test_loss:.4f}")
    evaluator.print_metrics()

if __name__ == "__main__":
    main()
