"""Cross-modal connectivity graphs — visualize lateral connection strengths."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

import torch

from brainnn.core.config import ModalityType
from brainnn.synesthesia.network import SynestheticNet


def plot_connectivity_matrix(
    net: SynestheticNet,
    title: str = "Cross-Modal Connection Strengths",
) -> matplotlib.figure.Figure:
    """Plot a heatmap of connection strengths between all modality pairs at each layer.

    Extracts learned gate values and projection norms to estimate how strongly
    each modality pair is connected.
    """
    modalities = net.modality_types
    n_mod = len(modalities)
    layers = sorted(int(k) for k in net.projection_banks.keys())
    n_layers = len(layers)

    # Build connectivity matrices per layer
    matrices = []
    for layer_idx in layers:
        bank = net.projection_banks[str(layer_idx)]
        matrix = np.zeros((n_mod, n_mod))

        for i, source in enumerate(modalities):
            for j, target in enumerate(modalities):
                if source == target:
                    matrix[i, j] = 1.0  # self-connection
                    continue
                key = f"{source.value}_to_{target.value}"
                if key in bank.connections:
                    conn = bank.connections[key]
                    # Use projection weight norm as strength proxy
                    weight_norm = sum(
                        p.data.norm().item()
                        for p in conn.projection.parameters()
                    )
                    matrix[i, j] = weight_norm
        matrices.append(matrix)

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    mod_labels = [m.value.capitalize() for m in modalities]

    for idx, (layer_idx, matrix) in enumerate(zip(layers, matrices)):
        im = axes[idx].imshow(matrix, cmap="YlOrRd", aspect="auto")
        axes[idx].set_xticks(range(n_mod))
        axes[idx].set_yticks(range(n_mod))
        axes[idx].set_xticklabels(mod_labels, rotation=45)
        axes[idx].set_yticklabels(mod_labels)
        axes[idx].set_title(f"Layer {layer_idx}")
        axes[idx].set_xlabel("Target")
        axes[idx].set_ylabel("Source")
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
