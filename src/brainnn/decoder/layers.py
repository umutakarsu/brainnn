"""Neuromorphic building blocks for BCI decoding.

Each layer implements a computational principle observed in biological
neural circuits, providing neuroscience-grounded inductive biases for
EEG decoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralInhibition(nn.Module):
    """Center-surround spatial sharpening inspired by cortical receptive fields.

    In biological neural circuits, neurons inhibit their neighbors to enhance
    contrast and sharpen spatial selectivity. Applied to EEG: each channel is
    sharpened by subtracting a weighted combination of neighboring channels.

    Unlike CSP (which learns global spatial rotations), lateral inhibition
    operates locally and preserves electrode topology.

    Args:
        n_channels: number of EEG channels
        init_radius: initial inhibition radius (channels within this distance
                     receive negative weights)
    """

    def __init__(self, n_channels: int, init_radius: float = 0.5) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.eye(n_channels))
        with torch.no_grad():
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        dist = abs(i - j) / n_channels
                        if dist < init_radius:
                            self.kernel[i, j] = -0.2 * (1 - dist / init_radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, C, T)"""
        return torch.einsum("ij,bjt->bit", self.kernel, x)


class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution capturing oscillatory dynamics.

    Different neural oscillations operate at different timescales — delta waves
    cycle over hundreds of milliseconds while gamma bursts last ~20ms. Instead of
    fixed filter banks, learnable convolutions at multiple temporal scales
    capture dynamics at each frequency resolution simultaneously.

    Args:
        in_channels: number of input channels
        hidden_dim: output channels per scale
        scales: list of kernel sizes (larger = slower oscillations)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 16,
        scales: list[int] | None = None,
    ) -> None:
        super().__init__()
        if scales is None:
            scales = [3, 7, 15, 31, 63]

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, k, padding="same", bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ELU(),
            )
            for k in scales
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(B, C, T) → list of (B, H, T), one per scale."""
        return [branch(x) for branch in self.branches]


class CrossFrequencyCoupling(nn.Module):
    """Differentiable phase-amplitude coupling between oscillatory scales.

    Phase-amplitude coupling (PAC) is a fundamental mechanism for neural
    communication: low-frequency phase organizes high-frequency amplitude,
    coordinating information transfer across cortical areas.

    This layer approximates PAC by modeling interactions between the outputs
    of slow (large-kernel) and fast (small-kernel) temporal convolutions.

    Args:
        n_scales: total number of temporal scales
        hidden_dim: channels per scale from MultiScaleTemporalConv
        coupling_dim: output dimension
    """

    def __init__(self, n_scales: int, hidden_dim: int, coupling_dim: int = 32) -> None:
        super().__init__()
        self.n_slow = max(1, n_scales // 2)
        n_fast = n_scales - self.n_slow
        n_pairs = self.n_slow * n_fast

        self.projection = nn.Sequential(
            nn.Conv1d(n_pairs * hidden_dim, coupling_dim * 2, 1, bias=False),
            nn.BatchNorm1d(coupling_dim * 2),
            nn.ELU(),
            nn.Conv1d(coupling_dim * 2, coupling_dim, 1),
        )

    def forward(self, scale_outputs: list[torch.Tensor]) -> torch.Tensor:
        """list of (B, H, T) → (B, coupling_dim, T)

        Slow scales (large kernels, last in list) provide phase proxy.
        Fast scales (small kernels, first in list) provide amplitude proxy.
        """
        fast = scale_outputs[: len(scale_outputs) - self.n_slow]
        slow = scale_outputs[len(scale_outputs) - self.n_slow :]

        couplings = []
        for s in slow:
            phase_proxy = torch.tanh(s)
            for f in fast:
                couplings.append(phase_proxy * f.abs())

        return self.projection(torch.cat(couplings, dim=1))


class PredictiveCodingBlock(nn.Module):
    """Prediction error extraction inspired by cortical predictive coding.

    The neocortex operates as a hierarchical prediction machine: each level
    generates top-down predictions of its input. Only the prediction error
    (surprise) propagates upward. This compresses redundant information and
    highlights informationally rich signal components.

    Args:
        in_dim: input feature dimension
        hidden_dim: output dimension
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.predictor = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, bias=False)
        self.error_transform = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, T) → (B, hidden, T-1)

        Computes prediction error between predicted and actual next timestep.
        """
        predicted = self.predictor(x)
        error = x[:, :, 1:] - predicted[:, :, :-1]
        return self.error_transform(error)


class NeuralOscillationAttention(nn.Module):
    """Temporal attention modulated by oscillatory dynamics.

    In the brain, alpha oscillations (8-13 Hz) gate information flow — high
    alpha power suppresses cortical processing while desynchronization
    releases processing resources. This layer learns to attend to
    task-relevant temporal segments using a self-attention mechanism.

    Args:
        hidden_dim: feature dimension
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, T) → (B, D)  via attention-weighted temporal pooling."""
        x_t = x.transpose(1, 2)  # (B, T, D)

        q = self.query(x_t.mean(dim=1, keepdim=True))  # (B, 1, D)
        k = self.key(x_t)
        v = self.value(x_t)

        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).squeeze(1)
        return self.norm(out)
