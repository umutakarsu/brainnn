"""EEG data acquisition, generation, and preprocessing."""

from brainnn.eeg.channels import MONTAGES, get_montage, channel_distances
from brainnn.eeg.preprocessing import bandpass_filter, notch_filter, normalize, epoch
from brainnn.eeg.synthetic import SyntheticEEGConfig, SyntheticMotorImagery

__all__ = [
    "MONTAGES",
    "SyntheticEEGConfig",
    "SyntheticMotorImagery",
    "bandpass_filter",
    "channel_distances",
    "epoch",
    "get_montage",
    "normalize",
    "notch_filter",
]
