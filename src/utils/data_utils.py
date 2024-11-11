import numpy as np
import torch
from typing import Optional
from collections import namedtuple


Batch = namedtuple("Batch", ["x", "y"])
BatchedData = namedtuple("BatchedData", ["num_samples", "batches"])


class Scaler:
    def __init__(self, shift: np.ndarray, scale: np.ndarray):
        """
        shift: [F,] -> [1, F, 1]
        scale: [F,] -> [1, F, 1]
        """
        self.shift = torch.from_numpy(shift.reshape(1, 1, -1, 1))
        self.scale = torch.from_numpy(scale.reshape(1, 1, -1, 1))

    def normalize(self, data: torch.Tensor, feature_idx: Optional[int] = None) -> torch.Tensor:
        """
        data: [B, V, F, time_steps] OR [B, V, time_steps]. Must specify feature_idx for the latter
        feature_idx: Specified index of the input feature. If None, assume data has all features
        ---
        data_norm: [B, V, F, time_steps] OR [B, V, time_steps]. data_norm = (data - shift) / scale
        """
        if feature_idx is not None and len(data.shape) == 4:
            data = data[:, :, feature_idx, :]
        shift = self.shift if feature_idx is None else self.shift[:, :, feature_idx, :].squeeze()
        scale = self.scale if feature_idx is None else self.scale[:, :, feature_idx, :].squeeze()
        return (data - shift) / scale

    def unnormalize(self, data: torch.Tensor, feature_idx: Optional[int] = None) -> torch.Tensor:
        """
        data: [B, V, F, time_steps] OR [B, V, time_steps]. Must specify feature_idx for the latter
        feature_idx: Specified index of the input feature. If None, assume data has all features
        ---
        data_unnorm: [B, V, F, time_steps] OR [B, V, time_steps]. data_unnorm = shift + scale * data
        """
        if feature_idx is not None and len(data.shape) == 4:
            data = data[:, :, feature_idx, :]
        shift = self.shift if feature_idx is None else self.shift[:, :, feature_idx, :].squeeze()
        scale = self.scale if feature_idx is None else self.scale[:, :, feature_idx, :].squeeze()
        return shift + scale * data
    

class TimeSeriesDataset:
    # TODO: This only works on PemsBayDatasetLoader & METRLADatasetLoader since their dataloaders are
    # modified to have indices and not be normalized using the entire dataset (preventing data leakage). 
    # Further, it is assumed the given graph is static over time, i.e. we only observe 1 graph. 
    def __init__(
        self, 
        dataloader, 
        window: int = 12, 
        horizon: int = 12, 
        train: float = 0.7, 
        test: float = 0.2,
        batch_size: int = 64, 
        shuffle_train: bool = True,
    ):
        """
        dataloader: One of the data loader objects defined under torch_geometric_temporal/dataset
        window (W): Number of timesteps the model is allowed to look-back
        horizon (H): Number of timesteps to predict ahead
        train: Ratio of data assigned to training. See split_data()
        test: Ratio of data assigned to testing. See split_data()
        batch_size (B): Number of continuous timeslices to load. 
            Divides N (self.snapshot_count) into batches of shape [B, V, F, H]
        shuffle_train: Whether to shuffle of the training data in the time dimension. See split_data()
        ---
        self.snapshot_count (N): Number of input-target-pairs (input: [V, F, W], target: [V, F, H])
        self.num_nodes (V): Number of nodes in the graph, i.e. number of individual time series
        self.edge_index: [2, E]. Edges in the form of (start_node, end_node)
        self.edge_attr: [E,]. Edge attributes
        
        self._x: [N, V, F, W]. Stacked inputs
        self._y: [N, V, F, H]. Stacked targets
        self._indices: List of (sample_start, sample_end) tuples that specifies the start (t - W + 1) 
            & end (t + H) indices of each data slice (t - W + 1, ..., t, t + 1, ..., t + H)
        self._raw_data: np.array[V, F, T], V=num_nodes, F=num_features, T=num_timesteps. Raw dataset
        """
        # StaticGraphTemporalSignal object defined under torch_geometric_temporal/signal
        dataset = dataloader.get_dataset(num_timesteps_in=window, num_timesteps_out=horizon)
        # Inputs and targets before data-splitting
        self.snapshot_count = dataset.snapshot_count # N
        self._x = torch.stack([snapshot.x for snapshot in dataset]) # [N, V, F, W]
        self._y = torch.stack([snapshot.y for snapshot in dataset]) # [N, V, F, H]
        # Given the graph is static, edge_index & edge_attr are the same for all timestamps
        self.edge_index = dataset[0].edge_index # [2, E]
        self.edge_attr = dataset[0].edge_attr # [E,]
        # Info required to create the Scaler object given any training ratio
        self._indices = dataloader.indices
        self._raw_data = dataloader.X.detach().cpu().numpy().astype(np.float32)
        if len(self._raw_data.shape) == 2: # [V, T]
            self._raw_data = self._raw_data.reshape(self._raw_data.shape[0], 1, self._raw_data.shape[1]) # [V, F=1, T]
        assert len(self._raw_data.shape) == 3, "Missing dimension(s) in the raw dataset"
        # Number of nodes in the graph
        self.num_nodes = self._raw_data.shape[0] # V
        # Split the dataset into train, test, valid batches
        self.split_data(train=train, test=test, shuffle_train=shuffle_train, batch_size=batch_size)

    def _create_batches(
        self,
        batch_size: int, 
        start: int = 0,
        end: Optional[int] = None,
        shuffle: bool = False, 
        fill_last_batch: bool = False
    ) -> Optional[BatchedData]:
        """
        Create batches with Batch.x of shape [B, V, F, W] and Batch.y of shape [B, V, F, H].
        If the last batch has less than B (batch_size) samples, optionally fill it with random samples.
        """
        end = self.snapshot_count if end is None else end
        if not (0 <= start < end <= self.snapshot_count):
            return None
        x, y = self._x[start:end], self._y[start:end]
        num_samples = len(x)
        # Optionally append random samples so that num_samples is a multiple of B (batch_size)
        if fill_last_batch and num_samples % batch_size != 0:
            extra_sample_idx = np.random.choice(
                range(num_samples), 
                size=batch_size - num_samples % batch_size, 
                replace=False
            )
            x = torch.cat(x, x[extra_sample_idx])
            y = torch.cat(y, y[extra_sample_idx])
            num_samples = len(x)
        # Optionally shuffle the samples
        if shuffle:
            permutation = np.random.permutation(num_samples)
            x, y = x[permutation], y[permutation]
        # Generate batches
        batches = []
        for b in range(0, num_samples, batch_size):
            batches.append(Batch(x[b:b + batch_size], y[b:b + batch_size]))
        batched_data = BatchedData(num_samples, batches)
        return batched_data
            
    def split_data(
        self, train: float = 0.7, test: float = 0.2, shuffle_train: bool = True, batch_size: int = 64
    ) -> None:
        """
        train: Ratio of data assigned to training. num_train = round(train * len_of_data)
        test: Ratio of data assigned to testing. num_train_test = num_train + round(test * len_of_data)
        shuffle_train: Whether to shuffle of the training data in the time dimension
            - False: (t - W + 1, ..., t, t + 1, ..., t + H) is returned according to t = 0, 1, ...
            - True: (t - W + 1, ..., t, t + 1, ..., t + H) is returned based on a random order of t
        ---
        self.data_splits: Dict of BatchedData. BatchedData.batches is a list of batches assigned to the split, 
        each batch is a namedtuple with batch.x being shape [B, V, F, W] and batch.y being shape [B, V, F, H], 
        and BatchedData.num_samples is the total number of samples (sum of batch lengths). 
            - train: data[:num_train]
            - test: data[num_train:num_train_test]
            - valid: data[num_train_test:]
        self.scaler: Scaler object with shift & scale set to the mean & std of the training samples 
            (across all nodes and training timesteps)
        """
        assert train > 0, "Train ratio must be positive"
        num_train = round(train * self.snapshot_count)
        num_train_test = num_train + round(test * self.snapshot_count)
        # Create data splits
        self.data_splits = {
            "train": self._create_batches(
                batch_size=batch_size, end=num_train, shuffle=shuffle_train
            ), 
            "test": self._create_batches(
                batch_size=batch_size, start=num_train, end=num_train_test
            ), 
            "valid": self._create_batches(batch_size=batch_size, start=num_train_test)
        }
        # Create Scaler based on the mean & std of training data
        num_train_samp = self._indices[num_train - 1][-1] # Index of last sample in train split
        raw_data_train = self._raw_data[:, :, :num_train_samp] # [V, F, num_train_samp]
        self.scaler = Scaler(
            shift=np.mean(raw_data_train, axis=(0, 2)), 
            scale=np.std(raw_data_train, axis=(0, 2))
        )

