"""PyTorch Dataset wrappers for synthetic EEG data."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from brainnn.eeg.preprocessing import bandpass_filter, normalize
from brainnn.eeg.synthetic import SyntheticEEGConfig, SyntheticMotorImagery


class SyntheticMotorImageryDataset(Dataset):
    """Motor imagery dataset from synthetic EEG generator.

    Args:
        n_trials: number of trials to generate
        config: SyntheticEEGConfig (defaults to 22ch/4class if None)
        low_freq: bandpass low cutoff (Hz)
        high_freq: bandpass high cutoff (Hz)
    """

    def __init__(
        self,
        n_trials: int = 500,
        config: SyntheticEEGConfig | None = None,
        low_freq: float = 4.0,
        high_freq: float = 40.0,
    ) -> None:
        config = config or SyntheticEEGConfig()
        gen = SyntheticMotorImagery(config)
        X, y = gen.generate(n_trials)

        X = bandpass_filter(X, low_freq, high_freq, config.sfreq)
        X = normalize(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.config = config

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_motor_imagery_loaders(
    n_trials: int = 500,
    config: SyntheticEEGConfig | None = None,
    val_split: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for synthetic motor imagery."""
    dataset = SyntheticMotorImageryDataset(n_trials=n_trials, config=config)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader
