import os
import sys
print(sys.path)
import yaml
import torch
import torch.nn.functional as F
from pytorch_geometric_temporal.torch_geometric_temporal.nn.recurrent import DCRNN
from pytorch_geometric_temporal.torch_geometric_temporal.dataset import PemsBayDatasetLoader, METRLADatasetLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from CS224W_Project import *
from src.utils.trainer import *


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
    

def main():
    df = TimeSeriesDataset(PemsBayDatasetLoader(), batch_size=32)
    model = RecurrentGCN(node_features=2)
    with open(os.path.join(CONFIG_PATH, "train_config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    config["train"]["epochs"] = 1 # Reduce epochs until we want a full run
    trainer = ModelTrainer(model, df, **config)
    trainer.train(print_per_epoch=1, inner_loop_progress_bar=True)
    trainer.test(use_progress_bar=True)


if __name__ == "__main__":
    main()
