import torch
from typing import Optional
from collections.abc import Iterable


def get_mask(
    y: torch.Tensor, 
    additional_val: Optional[float] = None
) -> torch.Tensor:
    """
    Returns a mask of Booleans based on whether values in y is np.nan or in additional_val
    
    y: Forecast actuals / targets. np.nan is understood as missing values, which should not be counted in loss functions
    additional_val: Value in addition to np.nan that should be masked
        e.g. Zeros need to be masked when computing MAPEs
    ---
    mask: Tensor of Booleans. False if the index is np.nan or additional_val
    """
    mask = ~torch.isnan(y)
    if additional_val is not None:
        mask = torch.logical_and(mask, y != additional_val)
    return mask


class Evaluator:
    def __init__(
        self, 
        num_nodes: int,
        num_features: Optional[int] = None,
        feature_idx: Optional[int] = 0,
        target_horizon_idx: Iterable[int] = (2, 5, 11), 
        max_horizon: int = 12
    ):
        """
        num_nodes (V): Number of nodes in the graph, i.e. number of individual time series
        num_features (F): Number of features.
        feature_idx: Index of the feature to evaluate. If None, evaluate all available features (F total).
            By default, we only evaluate the first feature.
        target_horizon_idx: List of indices of h-step ahead forecasts that should be evaluated. 
            By default we look 15, 30, and 60 minutes into the future, which translates to 3, 6, 12 steps 
            ahead because the timesteps are 5 minutes apart. Due to 0-indexing, this is 2, 5, and 11. 
        max_horizon (H): Maximum horizon of the targets. 
            Defaults to 12, which is 60 minutes into the future. 
        """
        assert num_features is not None or feature_idx is not None, "Must specify either num_features or feature_idx"
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.feature_idx = feature_idx
        self.target_horizon_idx = sorted(list(target_horizon_idx))
        self.max_horizon = max_horizon
        # Initialize everything to shape [V, f, H] or [V, H] of 0s (one 0 for each node, feature, & horizon)
        self.num_samples = self._init_zeros()
        self.num_samples_non_zero = self._init_zeros()
        self.sse = self._init_zeros() # Sum of squared errors
        self.sae = self._init_zeros() # Sum of absolute errors
        self.sape = self._init_zeros() # Sum of absolute percentage errors

    def _init_zeros(self) -> torch.Tensor:
        """
        Creates a zero tensor based on self.num_nodes (V) & optionally self.num_features (F)

        ---
        zeros: [V, F, H] if self.feature_idx is None, else [V, H]
        """
        if self.feature_idx is None:
            return torch.zeros(self.num_nodes, self.num_features, self.max_horizon) # [V, F, H]
        else:
            return torch.zeros(self.num_nodes, self.max_horizon) # [V, H]

    def update_metrics(self, y: torch.Tensor, y_hat: torch.Tensor) -> None:
        """
        Update components required to compute RMSE, MAE, and MAPE (see get_metrics)

        y: [B, V, F, H] OR [B, V, H]. Forecast targets / actuals
        y_hat: [B, V, F, H] if self.feature_idx is None, else [B, V, H]. Forecast values
        """
        if len(y.shape) == 4 and self.feature_idx is not None:
            y = y[:, :, self.feature_idx, :] # [B, V, F, H] -> [B, V, H]
        assert y.shape == y_hat.shape
        # Updates for MSE & MAE
        mask = get_mask(y) 
        self.num_samples += torch.sum(mask, dim=0)
        diff = torch.where(mask, y - y_hat, 0)
        self.sse += torch.sum(diff**2, dim=0)
        self.sae += torch.sum(torch.abs(diff), dim=0)
        # Updates for MAPE
        mask = get_mask(y, additional_val=0)
        self.num_samples_non_zero += torch.sum(mask, dim=0)
        diff = torch.where(mask, y - y_hat, 0)
        self.sape += torch.sum(torch.abs(diff / torch.where(mask, y, 1)), dim=0) * 100

    def get_metrics(self, agg_nodes: bool = True) -> dict[torch.Tensor]:
        """
        Returns a dictionary of evaluation metrics consisting of RMSE, MAE, and MAPE

        agg_nodes: Whether to aggregate across nodes / series
            - True: Aggregate across timesteps & nodes
            - False: Aggregate across timesteps only
        ---
        rmse, mae, mape: [F, H] OR [H,] if agg_nodes, else [V, F, H] OR [V, H]
            - rmse: Root mean square error
            - mae: Mean absolute error
            - mape: Mean absolute percentage error
        """
        if agg_nodes: # [F, H] if self.feature_idx is None, else [H,]
            rmse = torch.sqrt(torch.sum(self.sse, dim=0) / torch.sum(self.num_samples, dim=0))
            mae = torch.sum(self.sae, dim=0) / torch.sum(self.num_samples, dim=0)
            mape = torch.sum(self.sape, dim=0) / torch.sum(self.num_samples_non_zero, dim=0)
        else: # [V, F, H] if self.feature_idx is None, else [V, H]
            rmse = torch.sqrt(self.sse / self.num_samples)
            mae = self.sae / self.num_samples
            mape = self.sape / self.num_samples_non_zero
        return {"rmse": rmse, "mae": mae, "mape": mape}

    def print_metrics(self, metrics: Optional[dict[torch.Tensor]] = None) -> None:
        """
        Print out h-step ahead metrics for h specified by self.target_horizon_idx & 
        metrics averaged across all horizons (1, ..., H=self.max_horizon)

        metrics: Dict of [F, H] OR [H,] shaped tensors. Output of get_metrics(agg_nodes=True)
        """
        if metrics is None:
            metrics = self.get_metrics()
        for name, metric in metrics.items():
            print(f"--- {name.upper()} ---")
            for i in self.target_horizon_idx:
                vals_i = [metric[i]] if len(metric.shape) == 1 else metric[:, i]
                print(f"{(i + 1) * 5:2d}-min ahead:", *(f"{val:.4f}" for val in vals_i))
            avg_vals = [torch.mean(metric)] if len(metric.shape) == 1 else torch.mean(metric, dim=1)
            print(f"Average across all horizons:", *(f"{val:.4f}" for val in avg_vals))

