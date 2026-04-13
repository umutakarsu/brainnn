"""Unified dashboard — combine brain state timeline, activations, and connectivity."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.gridspec import GridSpec

from brainnn.core.state import BrainState
from brainnn.core.brain import BrainOutput, NeuroDivergentBrainSimulator
from brainnn.viz.activations import plot_modality_activations, plot_attention_weights


def plot_brain_state_timeline(
    state: BrainState,
    title: str = "Brain State Over Time",
) -> matplotlib.figure.Figure:
    """Plot the evolution of brain state variables over simulation steps."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    colors = {
        "dopamine": "#e74c3c",
        "focus": "#3498db",
        "arousal": "#2ecc71",
        "fatigue": "#95a5a6",
    }

    for ax, (key, color) in zip(axes.flat, colors.items()):
        values = state.history.get(key, [])
        if not values:
            ax.set_title(key.capitalize())
            continue
        steps = list(range(len(values)))
        ax.plot(steps, values, color=color, linewidth=2)
        ax.fill_between(steps, values, alpha=0.2, color=color)
        ax.set_title(key.capitalize(), fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # Mark hyperfocus zones for focus plot
        if key == "focus":
            for i, v in enumerate(values):
                if v > 0.8:
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="gold")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_simulation_dashboard(
    brain: NeuroDivergentBrainSimulator,
    outputs: list[BrainOutput],
    step_idx: int = -1,
) -> matplotlib.figure.Figure:
    """Create a comprehensive dashboard for a simulation run.

    Shows:
    - Brain state timeline (top row)
    - Modality activations at selected step (bottom left)
    - Attention weights at selected step (bottom right)

    Args:
        brain: the simulator instance (for state history)
        outputs: list of BrainOutput from simulation
        step_idx: which step to show detailed view for (-1 = last)
    """
    if step_idx < 0:
        step_idx = len(outputs) + step_idx

    output = outputs[step_idx]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

    # Row 1: Brain state timeline
    colors = {
        "dopamine": "#e74c3c",
        "focus": "#3498db",
        "arousal": "#2ecc71",
        "fatigue": "#95a5a6",
    }
    ax_state = fig.add_subplot(gs[0, :])
    for key, color in colors.items():
        values = brain.state.history.get(key, [])
        if values:
            ax_state.plot(range(len(values)), values, color=color, linewidth=2, label=key.capitalize())
    ax_state.axvline(x=step_idx, color="black", linestyle="--", alpha=0.5, label=f"Step {step_idx}")
    ax_state.set_ylim(-0.05, 1.05)
    ax_state.set_xlabel("Simulation Step")
    ax_state.set_ylabel("Value")
    ax_state.set_title("Brain State Timeline")
    ax_state.legend(loc="upper right")
    ax_state.grid(True, alpha=0.3)

    # Row 2: Per-modality activations
    modalities = list(output.modality_latents.keys())
    for i, mod in enumerate(modalities):
        ax = fig.add_subplot(gs[1, 0] if i == 0 else gs[1, 1])
        latent = output.modality_latents[mod].detach().cpu().numpy()
        if latent.ndim > 1:
            latent = latent[0]
        side = int(np.ceil(np.sqrt(len(latent))))
        padded = np.zeros(side * side)
        padded[:len(latent)] = latent
        grid = padded.reshape(side, side)
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_title(f"{mod.value.capitalize()} Activation")
        plt.colorbar(im, ax=ax, fraction=0.046)
        if i >= 1:
            break  # Show max 2 in this row

    # Row 3: Attention weights + state snapshot
    ax_attn = fig.add_subplot(gs[2, 0])
    weights = output.attention_weights.detach().cpu().numpy()
    if weights.ndim > 2:
        weights = weights[0]
    im = ax_attn.imshow(weights, cmap="hot", aspect="auto")
    ax_attn.set_ylabel("Attention Head")
    ax_attn.set_xlabel("Modality")
    ax_attn.set_title("Attention Weights")
    plt.colorbar(im, ax=ax_attn, fraction=0.046)

    # State snapshot as text
    ax_text = fig.add_subplot(gs[2, 1])
    ax_text.axis("off")
    snapshot = output.brain_state_snapshot
    text = "\n".join([
        f"Step: {int(snapshot.get('step', step_idx))}",
        f"Dopamine: {snapshot.get('dopamine', 0):.3f}",
        f"Focus: {snapshot.get('focus', 0):.3f}",
        f"Arousal: {snapshot.get('arousal', 0):.3f}",
        f"Fatigue: {snapshot.get('fatigue', 0):.3f}",
        f"SNR: {snapshot.get('snr', 0):.3f}",
        f"Hyperfocused: {'YES' if snapshot.get('is_hyperfocused', 0) > 0.5 else 'No'}",
    ])
    ax_text.text(0.1, 0.5, text, fontsize=14, fontfamily="monospace",
                 verticalalignment="center", transform=ax_text.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax_text.set_title("Brain State Snapshot")

    fig.suptitle("Neurodivergent Brain Simulator — Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig
