"""EEG preprocessing utilities — filtering, normalization, epoching."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def bandpass_filter(
    data: np.ndarray,
    low: float,
    high: float,
    sfreq: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Args:
        data: (..., n_samples) — works with any leading dimensions
        low: low cutoff frequency in Hz
        high: high cutoff frequency in Hz
        sfreq: sampling frequency in Hz
        order: filter order
    """
    nyq = sfreq / 2.0
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, data, axis=-1)


def notch_filter(
    data: np.ndarray,
    freq: float,
    sfreq: float,
    quality: float = 30.0,
) -> np.ndarray:
    """Remove powerline noise (50 or 60 Hz)."""
    nyq = sfreq / 2.0
    b, a = sp_signal.iirnotch(freq / nyq, quality)
    return sp_signal.filtfilt(b, a, data, axis=-1)


def normalize(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize EEG data along the time axis (last dimension).

    Args:
        data: (..., n_samples)
        method: 'zscore' or 'minmax'
    """
    if method == "zscore":
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (data - mean) / std
    if method == "minmax":
        lo = data.min(axis=-1, keepdims=True)
        hi = data.max(axis=-1, keepdims=True)
        rng = np.where(hi == lo, 1.0, hi - lo)
        return (data - lo) / rng
    raise ValueError(f"Unknown normalization method: {method}")


def epoch(
    continuous: np.ndarray,
    events: np.ndarray,
    sfreq: float,
    tmin: float = 0.0,
    tmax: float = 2.0,
) -> np.ndarray:
    """Segment continuous EEG into epochs around event onsets.

    Args:
        continuous: (n_channels, n_total_samples)
        events: (n_events,) sample indices of event onsets
        sfreq: sampling frequency
        tmin: start time relative to event (seconds)
        tmax: end time relative to event (seconds)

    Returns:
        (n_events, n_channels, epoch_samples)
    """
    start_offset = int(tmin * sfreq)
    end_offset = int(tmax * sfreq)
    epoch_len = end_offset - start_offset

    epochs = []
    for onset in events:
        s = int(onset) + start_offset
        e = s + epoch_len
        if s >= 0 and e <= continuous.shape[-1]:
            epochs.append(continuous[..., s:e])
    return np.stack(epochs)
