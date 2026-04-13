"""Activation heatmaps — visualize what each modality encoder is responding to."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.gridspec import GridSpec

import torch

from brainnn.core.config import ModalityType
from brainnn.core.brain import BrainOutput


def plot_modality_activations(
    output: BrainOutput,
    title: str = "Modality Activations",
) -> matplotlib.figure.Figure:
    """Plot activation heatmaps for each modality's latent representation.

    Args:
        output: BrainOutput from a simulation step
        title: plot title

    Returns:
        matplotlib Figure
    """
    modalities = list(output.modality_latents.keys())
    n_mod = len(modalities)

    fig, axes = plt.subplots(1, n_mod + 1, figsize=(4 * (n_mod + 1), 4))
    if n_mod + 1 == 1:
        axes = [axes]

    for i, mod in enumerate(modalities):
        latent = output.modality_latents[mod].detach().cpu().numpy()
        # Take first sample in batch
        if latent.ndim > 1:
            latent = latent[0]
        # Reshape to square-ish for visualization
        side = int(np.ceil(np.sqrt(len(latent))))
        padded = np.zeros(side * side)
        padded[:len(latent)] = latent
        grid = padded.reshape(side, side)

        im = axes[i].imshow(grid, cmap="viridis", aspect="auto")
        axes[i].set_title(f"{mod.value.capitalize()}")
        axes[i].set_xlabel("Latent dim")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    # Fused representation
    fused = output.fused_latent.detach().cpu().numpy()
    if fused.ndim > 1:
        fused = fused[0]
    side = int(np.ceil(np.sqrt(len(fused))))
    padded = np.zeros(side * side)
    padded[:len(fused)] = fused
    grid = padded.reshape(side, side)

    im = axes[-1].imshow(grid, cmap="magma", aspect="auto")
    axes[-1].set_title("Fused")
    plt.colorbar(im, ax=axes[-1], fraction=0.046)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_attention_weights(
    output: BrainOutput,
    title: str = "ADHD Attention Weights",
) -> matplotlib.figure.Figure:
    """Plot attention weight distribution across heads and modalities."""
    weights = output.attention_weights.detach().cpu().numpy()
    # (batch, heads, ...) — take first batch
    if weights.ndim > 2:
        weights = weights[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(weights, cmap="hot", aspect="auto")
    ax.set_ylabel("Attention Head")
    ax.set_xlabel("Modality / Position")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
