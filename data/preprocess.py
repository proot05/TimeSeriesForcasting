from torch.utils.data import Dataset
import pickle
import os
from typing import Tuple, List, Any
import numpy as np
from scipy.interpolate import interp1d
import torch
from typing import Sequence, Optional, Union
from data.normalize import Normalizer_ts
import matplotlib.pyplot as plt
from pathlib import Path


class TimeSeriesPreprocessor:
    def __init__(
        self,
        train_size: int,
        seq_len: int,
        norm_method: str = 'ms',
        norm_dim: int = 0,
        output_dir: Union[str, Path] = "."
    ):
        """
        Parameters
        ----------
        train_size : int
            Number of most‐recent points to use for training.
        seq_len : int
            Number of RNN inputs
        norm_method : str
            Method passed to Normalizer_ts (e.g. 'ms').
        norm_dim : int
            Dimension along which to normalize.
        output_dir : str or Path
            Base folder where plots & data subfolder will be created.
        """
        self.train_size = train_size
        self.seq_len = seq_len
        self.norm_method = norm_method
        self.norm_dim = norm_dim
        self.output_dir = Path(output_dir)

        # Will be set after calling .process()
        self.dt_new: Optional[float] = None
        self.normalizer: Optional[Normalizer_ts] = None
        self.dataset_inputs_n: Optional[torch.Tensor] = None

    def process(
        self,
        indtime_history: Sequence[float],
        ind_history: Sequence[float],
    ) -> torch.Tensor:
        """
        Slice out the last `train_size` points, interpolate them to
        uniform times, compute dt_new, normalize, and store results.

        Parameters
        ----------
        indtime_history : Sequence[float]
            Historical time stamps (non‐uniform).
        ind_history : Sequence[float]
            Historical data values at those time stamps.

        Returns
        -------
        torch.Tensor
            The normalized, interpolated dataset inputs of shape (train_size, 1).
        """
        latest_ind = len(indtime_history)
        if latest_ind < self.train_size:
            raise ValueError(
                f"Not enough data: got {latest_ind}, need {self.train_size}"
            )

        # interpolate to uniform time grid
        interp_func = interp1d(indtime_history, ind_history, kind='linear', fill_value='extrapolate')

        ind_times_new = np.linspace(indtime_history[0], indtime_history[-1], num=self.train_size)

        # new uniform timestep
        self.dt_new = float(ind_times_new[1] - ind_times_new[0])

        # generate new data points
        ind_acquired_new = interp_func(ind_times_new)

        # plot out how the interpolated data looks compared to the original data.
        data_dir = self.output_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        lw = 3
        ax.plot(indtime_history, ind_history, color='C0', linestyle='solid', linewidth=lw, alpha=1,
                label='Original Data')
        ax.plot(ind_times_new, ind_acquired_new, color='C1', linestyle='dashed', linewidth=lw, alpha=1,
                label='Processed Data')
        leg = ax.legend(loc='lower right', frameon=True)
        leg.get_frame().set_edgecolor('black')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Index')
        fig.savefig(data_dir / 'interpolated_data.png', bbox_inches='tight')
        plt.close()

        # to tensor of shape (train_size, 1)
        dataset_inputs = torch.tensor(ind_acquired_new, dtype=torch.float32).unsqueeze(-1)

        # normalize
        self.normalizer = Normalizer_ts(method=self.norm_method, dim=self.norm_dim)

        self.dataset_inputs_n = self.normalizer.fit_normalize(dataset_inputs)

        # plot out how the normalized data looks
        data_dir = self.output_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        lw = 3
        ax.plot(self.dataset_inputs_n, color='C1', linestyle='solid', linewidth=lw, alpha=1,)
        leg.get_frame().set_edgecolor('black')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Index')
        fig.savefig(data_dir / 'normalized_data.png', bbox_inches='tight')
        plt.close()

        return self.dataset_inputs_n

    def save_state(self, path: Union[str, Path]):
        """
        Save dt_new and the fitted normalizer together.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'dt_new': self.dt_new,
            'seq_len': self.seq_len,
            'normalizer': self.normalizer
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @staticmethod
    def load_state(path: Union[str, Path]) -> dict:
        """
        Load dt_new and normalizer from disk.
        Returns a dict with keys 'dt_new' (float) and 'normalizer' (Normalizer_ts).
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        return state

class MyDatasetAutoregress(Dataset):
    def __init__(self, inputs, n_seq1, forward):
        """
        Args:
            inputs (tensor): The input time series data.
            n_seq1 (int): Length of input sequence for the model.
            forward_steps (int): Number of steps ahead to predict.
        """
        self.inputs = inputs
        self.n_seq1 = n_seq1
        self.forward = forward

    def __len__(self):
        # The dataset length will be reduced by the number of steps ahead we are predicting
        return len(self.inputs) - self.n_seq1 - self.forward

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index to retrieve the sequence.

        Returns:
            dict: {'input': input sequence, 'output': target sequence (steps ahead)}
        """
        # The input sequence starts at idx and ends at idx + n_seq1
        input_seq = self.inputs[idx:int(idx + self.n_seq1)]

        # The output sequence starts at idx + forward_steps and ends at idx + n_seq1 + forward_steps
        output_seq = self.inputs[idx:int(idx + self.n_seq1 + self.forward)]

        return {'input': input_seq, 'output': output_seq}


# OLD CLASS
class MyDataset(Dataset):
    def __init__(self, inputs, n_seq1):
        self.inputs = inputs
        self.n_seq1 = n_seq1

    def __len__(self):
        return len(self.inputs)-self.n_seq1

    def __getitem__(self, idx):
        return {'input':self.inputs[idx:int(idx+self.n_seq1)],
                'output':self.inputs[idx+1:int(idx+self.n_seq1)+1]}


class MembraneDataLoader:
    def __init__(
            self,
            date: str,
            frequency: float,
            number: int,
            base_dir: str = "datasets"
    ):
        """
        Parameters
        ----------
        date : str
            The folder name for the date, e.g. "4-17-25".
        frequency : float
            The recording frequency, e.g. 0.67 (for "0.67 Hz membrane").
        number : int
            The sub‑folder recording number, e.g. 1.
        base_dir : str, optional
            Root directory where your "datasets" folder lives.
        """
        self.date = date
        self.frequency = frequency
        self.number = number
        self.base_dir = base_dir

    def _get_filepath(self) -> str:
        """
        Construct the full path to the lists.pkl file.
        """
        freq_folder = f"{self.frequency} Hz membrane"
        return os.path.join(
            self.base_dir,
            freq_folder,
            self.date,
            str(self.number),
            "lists.pkl"
        )

    def load_data(self) -> Tuple[List[float], Any]:
        """
        Load the data file, extract 'indtime' and 'ind' fields,
        and zero-reference the time vector.

        Returns
        -------
        mem_time : List[float]
            Time vector with first time subtracted (zero‑referenced).
        mem_data : Any
            The raw membrane data array (whatever type was stored).
        """
        path = self._get_filepath()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No data file found at {path!r}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        # assume data is a dict with keys 'indtime' and 'ind'
        raw_time = data["indtime"]
        mem_data = data["ind"]

        # zero‑reference the time vector
        origin = raw_time[0]
        mem_time = [t - origin for t in raw_time]

        return mem_time, mem_data

