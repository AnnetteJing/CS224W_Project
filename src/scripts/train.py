import os
import yaml
import argparse
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from CS224W_Project import *
from src.models.dcrnn import DCRNNModel
from src.models.stgcn import STGCNModel
from src.utils.trainer import *

from pytorch_geometric_temporal.torch_geometric_temporal.dataset import PemsBayDatasetLoader, METRLADatasetLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="One of the models defined under src.models")
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-d", "--data", type=str, default="both")
    parser.add_argument("-s", "--save-folder", type=str, default=None)
    parser.add_argument("-f", "--forecast-file-name", type=str, default=None)
    parser.add_argument("--full", action="store_true", help="Full run, saves to designated path if save_path is unspecified")
    parser.add_argument("--debug", action="store_true", help="Debug mode, reduces epoch to 1")
    args = parser.parse_args()
        
    # Read config file
    config_file_name = f"config_{args.model}.yaml" if args.config is None else args.config
    with open(os.path.join(CONFIG_PATH, config_file_name), "r") as f:
        config = yaml.safe_load(f)
    if args.debug:
        config["train"]["epochs"] = 1 # Reduce epochs to 1

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
    model_config = config.pop("model")
    for df_name, df in dfs.items():
        print("###############################################")
        print(f"Creating model for {df_name.upper()} dataset...")
        match args.model.lower():
            case "dcrnn":
                model = DCRNNModel(**model_config)
            case "stgcn":
                model = STGCNModel(num_nodes=df.num_nodes, **model_config)
            case _:
                raise ValueError(f"Model {args.model} not found")
            
        print(f"Training on {df_name.upper()} dataset...")
        trainer = ModelTrainer(model, df, **config)
        trainer.train(
            print_per_epoch=(1 if args.debug else 5), 
            inner_loop_progress_bar=args.debug
        )

        print(f"Testing on {df_name.upper()} dataset...")
        evaluator, targets, forecasts = trainer.test(use_progress_bar=True)

        # Save results, including model state_dict, losses, and performance metrics
        print(f"Saving model state dict...")
        if args.save_folder is None:
            if args.full:
                save_folder = args.model
            else:
                save_folder = "test"
        else:
            save_folder = args.save_folder
        save_path = os.path.join(RESULTS_PATH, save_folder, df_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(trainer.model.state_dict(), os.path.join(save_path, "state_dict.pt"))

        print(f"Saving training & validation losses...")
        np.save(os.path.join(save_path, "loss_train.npy"), np.array(trainer.train_loss))
        np.save(os.path.join(save_path, "loss_valid.npy"), np.array(trainer.valid_loss))

        print(f"Saving test results...")
        np.savez_compressed(
            os.path.join(save_path, "metrics.npz"),
            num_samples=evaluator.num_samples.detach().cpu().numpy(), 
            num_samples_non_zero=evaluator.num_samples_non_zero.detach().cpu().numpy(), 
            sse=evaluator.sse.detach().cpu().numpy(),
            sae=evaluator.sae.detach().cpu().numpy(), 
            sape=evaluator.sape.detach().cpu().numpy(),
        )

        print(f"Results saved under {save_path}")

        # Save forecasts & targets separately under a folder ignored by git
        print(f"Saving forecasts & targets...")
        forecast_save_path = os.path.join(RESULTS_PATH, "forecasts")
        if not os.path.exists(forecast_save_path):
            os.makedirs(forecast_save_path)
        if args.forecast_file_name is None:
            forecast_file_name = f"{args.model}_{df_name}"
        else:
            forecast_file_name = args.forecast_file_name
        if args.debug:
            forecast_file_name += "_debug"
        np.savez_compressed(
            os.path.join(forecast_save_path, f"{forecast_file_name}.npz"),
            targets=targets, 
            forecasts=forecasts,
        )

        np.savez_compressed(
            os.path.join(forecast_save_path, f"{forecast_file_name}_example.npz"), 
            train_inputs=trainer.train_input_output_example["inputs"],
            train_outputs=trainer.train_input_output_example["outputs"],
            train_targets=trainer.train_input_output_example["targets"],
            valid_inputs=trainer.valid_input_output_example["inputs"],
            valid_outputs=trainer.valid_input_output_example["outputs"],
            valid_targets=trainer.valid_input_output_example["targets"],
        )

        print(f"Forecasts & examples saved under {forecast_save_path}")


if __name__ == "__main__":
    main()
