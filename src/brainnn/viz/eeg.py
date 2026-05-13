"""EEG visualization — signal plots, topographic maps, power spectra."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import signal as sp_signal
from scipy.interpolate import griddata

from brainnn.eeg.channels import get_montage


def plot_eeg(
    data: np.ndarray,
    sfreq: float,
    channel_names: list[str] | None = None,
    title: str | None = None,
    offset: float = 50.0,
) -> Figure:
    """Plot multi-channel EEG as stacked traces.

    Args:
        data: (n_channels, n_samples)
        sfreq: sampling frequency
        channel_names: list of channel labels
        title: plot title
        offset: vertical spacing between channels
    """
    n_ch, n_samples = data.shape
    t = np.arange(n_samples) / sfreq

    fig, ax = plt.subplots(figsize=(14, max(3, 0.4 * n_ch + 1)))
    for i in range(n_ch):
        ax.plot(t, data[i] + i * offset, "k-", linewidth=0.4, alpha=0.8)

    if channel_names:
        ax.set_yticks(np.arange(n_ch) * offset)
        ax.set_yticklabels(channel_names, fontsize=7)
    else:
        ax.set_yticks([])

    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_psd(
    data: np.ndarray,
    sfreq: float,
    channel_names: list[str] | None = None,
    fmax: float = 50.0,
) -> Figure:
    """Plot power spectral density."""
    nperseg = min(256, data.shape[-1])
    freqs, psd = sp_signal.welch(data, sfreq, nperseg=nperseg, axis=-1)
    mask = freqs <= fmax

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(psd.shape[0]):
        label = channel_names[i] if channel_names else f"Ch {i}"
        ax.semilogy(freqs[mask], psd[i, mask], alpha=0.6, linewidth=0.8, label=label)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.set_title("Power Spectral Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_topomap(
    values: np.ndarray,
    n_channels: int,
    title: str | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "RdBu_r",
) -> plt.Axes:
    """Plot topographic map of scalp values.

    Args:
        values: (n_channels,) values to map
        n_channels: channel count (must match a known montage)
        title: plot title
        ax: matplotlib axes (creates one if None)
        cmap: colormap
    """
    montage = get_montage(n_channels)
    positions = montage["positions"]
    names = montage["names"]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    xi = np.linspace(-1.1, 1.1, 150)
    yi = np.linspace(-1.1, 1.1, 150)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(positions, values, (xi, yi), method="cubic", fill_value=np.nan)
    mask = xi ** 2 + yi ** 2 > 1.0
    zi[mask] = np.nan

    ax.contourf(xi, yi, zi, levels=25, cmap=cmap, extend="both")

    # Head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)

    # Nose
    ax.plot([-0.08, 0, 0.08], [1.0, 1.12, 1.0], "k-", linewidth=2)

    # Ears
    for sign in [-1, 1]:
        ax.plot(
            [sign * 1.0, sign * 1.08, sign * 1.08, sign * 1.0],
            [0.1, 0.05, -0.05, -0.1],
            "k-", linewidth=1.5,
        )

    ax.scatter(positions[:, 0], positions[:, 1], c="k", s=15, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, positions[i], fontsize=5, ha="center", va="bottom")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    return ax


def plot_erp(
    epochs: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    channel_idx: int = 0,
    channel_name: str | None = None,
    class_names: list[str] | None = None,
) -> Figure:
    """Plot event-related potentials (grand average per class).

    Args:
        epochs: (n_trials, n_channels, n_samples)
        labels: (n_trials,)
        sfreq: sampling frequency
        channel_idx: which channel to plot
        channel_name: label for the channel
        class_names: labels for each class
    """
    classes = np.unique(labels)
    n_samples = epochs.shape[-1]
    t = np.arange(n_samples) / sfreq

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in classes:
        mask = labels == c
        mean = epochs[mask, channel_idx, :].mean(axis=0)
        std = epochs[mask, channel_idx, :].std(axis=0)
        label = class_names[int(c)] if class_names else f"Class {int(c)}"
        ax.plot(t, mean, linewidth=1.5, label=label)
        ax.fill_between(t, mean - std, mean + std, alpha=0.15)

    ch_label = channel_name or f"Channel {channel_idx}"
    ax.set_title(f"Event-Related Potential — {ch_label}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
