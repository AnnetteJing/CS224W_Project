import torch
import torch.nn as nn
from typing import Optional, Callable
from tqdm import tqdm
import time
import copy

from src.utils.data_utils import *
from src.utils.metrics import *


class ModelTrainer:
    def __init__(
        self, 
        model: type[nn.Module], 
        dataset: TimeSeriesDataset, 
        feature_idx: Optional[int] = 0,
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = masked_mae_loss, 
        device: Optional[str] = None,
        **kwargs
    ):
        """
        model: Model to train & evaluate
        dataset: TimeSeriesDataset used as input to the model
        feature_idx: Index of the feature that is being used as the target. Use all features if None
        loss_func: Loss function for training
        device: "cuda" or "cpu". If None, uses "cuda" whenever available 
        kwargs: Other configurations (see src/configs/train_config.yaml)
        """
        self.feature_idx = feature_idx
        self.loss_func = loss_func
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device)
        # Save dataset
        self.df = dataset.data_splits
        self.num_features = dataset.num_features # F
        self.num_nodes = dataset.num_nodes # V
        self.edge_index = dataset.edge_index.to(self.device)
        self.edge_attr = dataset.edge_attr.to(self.device)
        self.scaler = dataset.scaler # Performs normalization based on train data mean & std
        # Set up for automatic mixed precision training
        self.gradscaler = torch.amp.GradScaler(self.device)
        self.amp_context = torch.amp.autocast("cuda", enabled=(self.device.type == "cuda"))
        # Create self.optimizer & self.scheduler
        optim_config = kwargs.get("optim")
        self._create_optimizer_scheduler(optim_config)
        # Record training configs
        train_config = kwargs.get("train")
        self.epochs = train_config["epochs"]
        self.max_grad_norm = train_config["max_grad_norm"]

    def _create_optimizer_scheduler(self, optim_config: dict):
        """
        Creates an Adam optimizer & multistep learning rate scheduler based on optim_config
        """
        self.optimizer = torch.optim.Adam(
            filter(lambda p : p.requires_grad, self.model.parameters()), 
            lr=optim_config["initial_lr"], 
            eps=optim_config["epsilon"]
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=optim_config["lr_decay_milestones"], 
            gamma=optim_config["lr_decay_ratio"]
        )

    def get_preds(self, x: torch.Tensor, unnormalize: bool = False) -> torch.Tensor:
        """
        Computes model predictions of the targets

        x: [B, V, F, W]. Model inputs
        unnormalize: Whether to unnormalize the predictions
            If True, we are predicting the targets; otherwise we are predicting the normalized targets
        ---
        y_hat: [B, V, F, H] OR [B, V, H]. Model predictions
        """
        # Model prediction of normalized targets
        y_hat = self.model(
            x=x.to(self.device), 
            edge_index=self.edge_index, 
            edge_weight=self.edge_attr
        ) # [B, V, F, H] if self.feature_idx is None else [B, V, H]
        if unnormalize:
            # Model prediction of targets (not normalized)
            y_hat = self.scaler.unnormalize(y_hat, feature_idx=self.feature_idx)
        return y_hat

    def get_batch_loss(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss a data batch

        y: [B, V, F, H] OR [B, V, H]. Targets
        y_hat: [B, V, F, H] OR [B, V, H]. Model predictions
        ---
        batch_loss: [1]. Loss averaged across the batch
        """
        # Normalize targets
        y_norm = self.scaler.normalize(
            y.to(self.device), feature_idx=self.feature_idx
        ) # [B, V, F, H] -> [B, V, F, H] if self.feature_idx is None else [B, V, H]
        # Compute loss
        return self.loss_func(y=y_norm, y_hat=y_hat)

    def train_epoch(self, use_progress_bar: bool = False) -> float:
        """
        Train the model using self.df["train"]
        """
        self.model.train()
        train_loss = 0
        self.optimizer.zero_grad()
        iter_train_batches = tqdm(self.df["train"].batches) if use_progress_bar else self.df["train"].batches
        for x, y in iter_train_batches:
            x, y = x.to(self.device), y.to(self.device)
            with self.amp_context:
                y_hat = self.get_preds(x=x)
                train_batch_loss = self.get_batch_loss(y=y, y_hat=y_hat)
            self.gradscaler.scale(train_batch_loss).backward() # Scale loss before backprop
            train_loss += train_batch_loss.item()
            torch.cuda.empty_cache()
        self.gradscaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.gradscaler.step(self.optimizer)
        self.gradscaler.update()
        self.scheduler.step()
        train_loss /= self.df["train"].num_batches
        return train_loss

    def valid_epoch(self, use_progress_bar: bool = False) -> Optional[float]:
        """
        Evaluate the model using self.df["valid"]
        """
        if self.df["valid"] is None:
            return None
        self.model.eval()
        valid_loss = 0
        with torch.no_grad(), self.amp_context:
            iter_valid_batches = tqdm(self.df["valid"].batches) if use_progress_bar else self.df["valid"].batches
            for x, y in iter_valid_batches:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.get_preds(x=x)
                valid_loss += self.get_batch_loss(y=y, y_hat=y_hat).item()
                torch.cuda.empty_cache()
        valid_loss /= self.df["valid"].num_batches
        return valid_loss

    def train(
        self, 
        verbose: bool = True, 
        print_per_epoch: int = 10, 
        early_stopping: bool = True, 
        inner_loop_progress_bar: bool = False
    ) -> None:
        """
        Train & validate the model for multiple epochs, saving the best model based on validation loss 
        (use training loss if validation loss is not available).

        verbose: Whether to print training & validation loss every few epochs
        print_per_epoch: Number of epochs between printing losses if verbose=True
        early_stopping: Whether to perform early stopping when validation loss has been increasing for 15 epochs
            (use training loss if validation loss is not available)
        inner_loop_progress_bar: Whether to use progress bars when looping over the train & validation batches
        """
        self.train_loss = []
        self.valid_loss = []
        best_loss = float("inf")
        best_model_epoch = 0
        num_loss_increase, prev_loss = 0, float("inf") # Needed for early stopping
        epoch_loop = range(self.epochs) if verbose else tqdm(range(self.epochs))
        for epoch in epoch_loop:
            if verbose:
                start_time = time.time()
            train_loss = self.train_epoch(use_progress_bar=inner_loop_progress_bar)
            valid_loss = self.valid_epoch(use_progress_bar=inner_loop_progress_bar)
            self.train_loss.append(train_loss)
            if valid_loss is not None:
                self.valid_loss.append(valid_loss)
            # Save best model
            loss = valid_loss if valid_loss else train_loss
            if loss < best_loss:
                best_model_epoch = epoch
                best_model = copy.deepcopy(self.model)
                best_loss = loss
            # Print progress every `print_per_epoch` epochs if verbose
            if verbose and epoch % print_per_epoch == 0:
                end_time = time.time()
                print_statement = f"Epoch {epoch:2d} ({end_time - start_time:3.2f}s): Train loss = {train_loss:.4f}"
                if valid_loss is not None:
                    print_statement += f", Valid loss = {valid_loss:.4f}"
                print(print_statement)
            # Optional early stopping
            if early_stopping:
                if loss > prev_loss:
                    num_loss_increase += 1
                else:
                    num_loss_increase = 0
                if num_loss_increase == 15:
                    print("Loss has been increasing for 15 epochs, performing early stopping...")
                    break
                prev_loss = loss
        # Load best model
        if verbose:
            print(f"Saving best model from epoch {best_model_epoch} with loss {best_loss:4f}")
        self.model = best_model

    def test(self, test_data: Optional[BatchedData] = None, use_progress_bar: bool = False) -> Evaluator:
        """
        Test the model's performance

        test_data: Data used for testing. Defaults to self.df["test"]
        use_progress_bar: Whether to use a progress bar for looping over the test batches
        ---
        evaluator: Evaluator object that records performance details
        """
        test_data = self.df["test"] if test_data is None else test_data
        evaluator = Evaluator(
            num_nodes=self.num_nodes, 
            num_features=self.num_features,
            feature_idx=self.feature_idx
        )
        self.model.eval()
        test_loss = 0
        with torch.no_grad(), self.amp_context:
            iter_test_batches = tqdm(test_data.batches) if use_progress_bar else test_data.batches
            for x, y in iter_test_batches:
                x, y = x.to(self.device), y.to(self.device)
                # Compute test loss based on model prediction of normalized targets 
                y_hat = self.get_preds(x=x)
                test_loss += self.get_batch_loss(y=y, y_hat=y_hat).item()
                # Update evaluator based on model prediction of targets (not normalized)
                y_hat = self.scaler.unnormalize(y_hat, feature_idx=self.feature_idx)
                evaluator.update_metrics(y=y, y_hat=y_hat)
                # Free up cache
                torch.cuda.empty_cache()
        test_loss /= test_data.num_batches
        print(f"Test loss = {test_loss:.4f}")
        evaluator.print_metrics()
        return evaluator