import numpy as np
import torch


class Scaler:
    def __init__(self, shift: np.ndarray, scale: np.ndarray):
        """
        shift: [F,] -> [1, F, 1]
        scale: [F,] -> [1, F, 1]
        """
        self.shift = shift.reshape(1, -1, 1)
        self.scale = scale.reshape(1, -1, 1)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [V, F, W + H]
        ---
        data_norm: [V, F, W + H]. data_norm = (data - shift) / scale
        """
        return (data - self.shift) / self.scale

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [V, F, W + H]
        ---
        data_unnorm: [V, F, W + H]. data_unnorm = shift + scale * data
        """
        return self.shift + self.scale * data
    

class TimeSeriesDataset:
    # TODO: This only works on PemsBayDatasetLoader & METRLADatasetLoader since their dataloaders are
    # modified to have indices and not be normalized using the entire dataset (preventing data leakage). 
    def __init__(
        self, 
        dataloader, 
        window: int = 12, 
        horizon: int = 12, 
        train: float = 0.7, 
        test: float = 0.2,
        shuffle_train: bool = True,
    ):
        """
        dataloader: One of the data loader objects defined under torch_geometric_temporal/dataset
        window (W): Number of timesteps the model is allowed to look-back
        horizon (H): Number of timesteps to predict ahead
        train: Ratio of data assigned to training. See split_data()
        test: Ratio of data assigned to testing. See split_data()
        shuffle_train: Whether to shuffle of the training data in the time dimension. See split_data()
        ---
        indices: List of (sample_start, sample_end) tuples that specifies the start (t - W + 1) 
            & end (t + H) indices of each data slice (t - W + 1, ..., t, t + 1, ..., t + H)
        X: np.array[V, F, T], V=num_nodes, F=num_features, T=num_timesteps. Raw dataset
        data: Graph signal object defined under torch_geometric_temporal/signal
        """
        self.data = dataloader.get_dataset(num_timesteps_in=window, num_timesteps_out=horizon)
        self.indices = dataloader.indices
        self.X = dataloader.X.detach().cpu().numpy().astype(np.float32)
        if len(self.X.shape) == 2: # [V, T]
            self.X = self.X.reshape(self.X.shape[0], 1, self.X.shape[1]) # [V, F=1, T]
        assert len(self.X.shape) == 3, "Missing dimension(s) in the raw dataset"
        self.split_data(train=train, test=test)

    def split_data(self, train: float = 0.7, test: float = 0.2, shuffle_train: bool = True):
        """
        train: Ratio of data assigned to training. num_train = round(train * len_of_data)
        test: Ratio of data assigned to testing. num_train_test = num_train + round(test * len_of_data)
        shuffle_train: Whether to shuffle of the training data in the time dimension
            - False: (t - W + 1, ..., t, t + 1, ..., t + H) is returned according to t = 0, 1, ...
            - True: (t - W + 1, ..., t, t + 1, ..., t + H) is returned based on a random order of t
        ---
        data_splits: Dict of Iterable[Data], where the Data object describes an homogeneous graph
            (see torch_geometric.data.data for the full definition)
            - train: data[:num_train]. 
            - test: data[num_train:num_train_test]
            - valid: data[num_train_test:]
        scaler: Scaler object with shift & scale set to the mean & std of the training samples 
            (across all nodes and training timesteps)
        """
        assert train > 0, "Train ratio must be positive"
        num_train = round(train * self.data.snapshot_count)
        num_train_test = num_train + round(test * self.data.snapshot_count)
        # Create data splits
        self.data_splits = {
            "train": self.data[:num_train], 
            "test": (
                self.data[num_train:num_train_test] 
                if num_train_test <= self.data.snapshot_count else None
            ), 
            "valid": (
                self.data[num_train_test:]
                if num_train_test < self.data.snapshot_count else None
            )
        }
        if shuffle_train:
            permutation = np.random.permutation(num_train)
            self.data_splits["train"] = [self.data_splits["train"][t] for t in permutation]
        # Create Scaler based on the mean & std of training data
        num_train_samp = self.indices[num_train - 1][-1] # Index of last sample in train split
        X_train = self.X[:, :, :num_train_samp] # [V, F, num_train_samp]
        self.scaler = Scaler(
            shift=np.mean(X_train, axis=(0, 2)), 
            scale=np.std(X_train, axis=(0, 2))
        )

