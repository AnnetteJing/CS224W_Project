import os
import sys

sys.path.append('/home/dlvm/CS224w/pytorch_geometric_temporal')
import torch
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
    #print("init masked mae loss")
    mask = mask.float()
    mask = mask.to(y_pred.device)
    #print("mask created")
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    #print("returning masked mae loss")
    return torch.mean(loss)

# Model definition
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.dcrnn1 = DCRNN(in_channels=node_features, out_channels=64, K=2)
        self.dcrnn2 = DCRNN(in_channels=64, out_channels=64, K=2)
        self.linear = torch.nn.Linear(64, 12)

    def forward(self, x, edge_index, edge_weight):
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
    df = TimeSeriesDataset(PemsBayDatasetLoader(), batch_size=32)
    epochs = 1
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    model = RecurrentGCN(node_features=2).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        eps=1e-3
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 30, 40, 50], gamma=0.1
    )
    gradscaler = torch.amp.GradScaler()

    # Training loop
    train_batches = df.data_splits["train"].batches[:1]
    valid_batches = df.data_splits["valid"].batches[:1]
    
    device_context = torch.amp.autocast("cuda", enabled=(device.type == "cuda"))

    best_valid_loss = float('inf')
    best_epoch = 0
    for epoch in tqdm(range(epochs)):
        # Train
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for x, y in tqdm(train_batches):
            with device_context:
                y_hat = model(
                    x=x.to(device),
                    edge_index=df.edge_index.to(device),
                    edge_weight=df.edge_attr.to(device)
                ).to(device)
                y_norm = df.scaler.normalize(y, feature_idx=0).to(device)
                
                batch_train_loss = masked_mae_loss(y_pred=y_hat, y_true=y_norm)
                #print("done with loss")
        
            del y_hat, y_norm
            torch.cuda.empty_cache()
            #print("before scaling")
            #batch_train_loss.backward()
            gradscaler.scale(batch_train_loss).backward()
            #print("after scaling") 
            train_loss += batch_train_loss.item()
            #torch.cuda.empty_cache()
        
        gradscaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        gradscaler.step(optimizer)
        gradscaler.update()
        scheduler.step()
        train_loss /= len(train_batches)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad(), device_context:
            for x, y in valid_batches:
                y_hat = model(
                    x=x.to(device),
                    edge_index=df.edge_index.to(device),
                    edge_weight=df.edge_attr.to(device)
                )
                y_norm = df.scaler.normalize(y, feature_idx=0).to(device)
                batch_valid_loss = masked_mae_loss(y_pred=y_hat, y_true=y_norm)
                valid_loss += batch_valid_loss.item()
                torch.cuda.empty_cache()
                
        valid_loss /= len(valid_batches)
        print(f"Epoch {epoch}: Train loss = {train_loss:4f}, Valid loss = {valid_loss:4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
        elif epoch - best_epoch >= 10:
            print("Early stopping triggered.")
            break
    # Evaluation
    model.eval()
    evaluator = Evaluator(num_nodes=df.num_nodes)
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
            print("eval y shape", y.shape)
            print("eval y_hat shape", y_hat.shape)
            evaluator.update_metrics(y.to(device), y_hat)
            torch.cuda.empty_cache()

    test_loss /= len(df.data_splits["test"].batches)
    print("\nTest Results:")
    print(f"Average Loss: {test_loss:.4f}")
    evaluator.print_metrics()
    
if __name__ == "__main__":
    main()
