"""Feature extraction from EEG signals — band power, CSP, spectral features."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

STANDARD_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


def band_power(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    method: str = "welch",
) -> np.ndarray:
    """Extract average power in a frequency band.

    Args:
        data: (..., n_samples)
        sfreq: sampling frequency
        band: (low_freq, high_freq) in Hz
        method: 'welch' or 'fft'

    Returns:
        Power array with shape data.shape[:-1]
    """
    if method == "welch":
        nperseg = min(256, data.shape[-1])
        freqs, psd = sp_signal.welch(data, sfreq, nperseg=nperseg, axis=-1)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return psd[..., mask].mean(axis=-1)

    if method == "fft":
        n = data.shape[-1]
        fft_vals = np.fft.rfft(data, axis=-1)
        fft_power = np.abs(fft_vals) ** 2 / n
        freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return fft_power[..., mask].mean(axis=-1)

    raise ValueError(f"Unknown method: {method}")


def all_band_powers(
    data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Extract power for all standard frequency bands."""
    if bands is None:
        bands = STANDARD_BANDS
    return {name: band_power(data, sfreq, freq_range) for name, freq_range in bands.items()}


def compute_csp(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 4,
) -> np.ndarray:
    """Common Spatial Patterns for 2-class motor imagery.

    Args:
        X: (n_trials, n_channels, n_samples)
        y: (n_trials,) binary labels
        n_components: total number of spatial filters (takes top + bottom)

    Returns:
        W: (n_components, n_channels) spatial filter matrix
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}")

    X0 = X[y == classes[0]]
    X1 = X[y == classes[1]]

    C0 = np.mean([np.cov(trial) for trial in X0], axis=0)
    C1 = np.mean([np.cov(trial) for trial in X1], axis=0)

    Cc = C0 + C1

    eigvals, eigvecs = np.linalg.eigh(Cc)
    eigvals = np.maximum(eigvals, 1e-10)
    P = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    S0 = P @ C0 @ P.T
    _, W = np.linalg.eigh(S0)
    W = W.T

    n_half = n_components // 2
    selected = np.concatenate([W[:n_half], W[-n_half:]], axis=0)
    return selected @ P


def csp_features(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply CSP spatial filters and extract log-variance features.

    Args:
        X: (n_trials, n_channels, n_samples)
        W: (n_components, n_channels) from compute_csp

    Returns:
        (n_trials, n_components) log-variance features
    """
    Z = np.array([W @ trial for trial in X])
    return np.log(np.var(Z, axis=-1) + 1e-10)
