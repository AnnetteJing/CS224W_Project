import os
import yaml
import argparse
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from CS224W_Project import *
from src.models.dcrnn import DCRNNModel
from src.utils.trainer import *

from pytorch_geometric_temporal.torch_geometric_temporal.dataset import PemsBayDatasetLoader, METRLADatasetLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="One of the models defined under src.models")
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-d", "--data", type=str, default="both")
    parser.add_argument("--debug", action="store_true", help="Debug mode, reduces epoch to 1")
    args = parser.parse_args()
        
    # Read config file
    config_file_name = f"config_{args.model}.yaml" if args.config is None else args.config
    with open(os.path.join(CONFIG_PATH, config_file_name), "r") as f:
        config = yaml.safe_load(f)
    if args.debug:
        config["train"]["epochs"] = 1 # Reduce epochs to 1

    # Create model
    match args.model.lower():
        case "dcrnn":
            model = DCRNNModel(**config.get("model"))
        case _:
            raise ValueError(f"Model {args.model} not found")

    # Load datasets
    match args.data.lower():
        case "metr" | "metr-la" | "metr_la" | "metrla":
            dfs = {"metr": TimeSeriesDataset(METRLADatasetLoader(), batch_size=args.batch_size)}
        case "pems" | "pems-bay" | "pems_bay" | "pemsbay":
            dfs = {"pems": TimeSeriesDataset(PemsBayDatasetLoader(), batch_size=args.batch_size)}
        case _:
            dfs = {
                "metr": TimeSeriesDataset(METRLADatasetLoader(), batch_size=args.batch_size), 
                "pems": TimeSeriesDataset(PemsBayDatasetLoader(), batch_size=args.batch_size), 
            }

    # Train & test
    for df_name, df in dfs.items():
        print("###############################################")
        print(f"Training on {df_name.upper()} dataset...")
        trainer = ModelTrainer(model, df, **config)
        trainer.train(
            print_per_epoch=(1 if args.debug else 10), 
            inner_loop_progress_bar=args.debug
        )

        print(f"Testing on {df_name.upper()} dataset...")
        evaluator = trainer.test(use_progress_bar=True)

        print(f"Saving model state dict...")
        save_path = os.path.join(RESULTS_PATH, args.model, df_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(trainer.model.state_dict(), os.path.join(save_path, "state_dict.pt"))

        print(f"Saving test results...")
        metrics = {
            "num_samples": evaluator.num_samples.numpy(), 
            "num_samples_non_zero": evaluator.num_samples_non_zero.numpy(), 
            "sse": evaluator.sse.numpy(),
            "sae": evaluator.sae.numpy(), 
            "sape": evaluator.sape.numpy(),
        }
        np.savez(os.path.join(save_path, f"{df_name}_metrics.npz"), metrics)
        

if __name__ == "__main__":
    main()
