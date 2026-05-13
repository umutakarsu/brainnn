"""SynapseFlow — neuromorphic EEG decoder.

Core thesis: the brain is the best decoder of itself. Instead of applying
generic machine learning to EEG, SynapseFlow uses computational principles
from neuroscience as architectural priors:

    1. Lateral Inhibition     — center-surround spatial sharpening (V1)
    2. Multi-scale Oscillatory Decomposition — frequency-band analysis
    3. Cross-Frequency Coupling — phase-amplitude interactions (hippocampus)
    4. Predictive Coding      — prediction error signals (hierarchical cortex)
    5. Neural Oscillation Attention — oscillation-gated temporal selection

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Raw EEG (B, C, T)                                     │
    │       ↓                                                 │
    │  [Lateral Inhibition] ── spatial sharpening             │
    │       ↓                                                 │
    │  [Multi-Scale Temporal Conv]                            │
    │     ↓    ↓    ↓    ↓    ↓   ── oscillatory streams     │
    │    δ    θ    α    β    γ                                │
    │     ↓    ↓    ↓    ↓    ↓                               │
    │  [Cross-Frequency Coupling] ── PAC features             │
    │       ↓                                                 │
    │  [Predictive Coding] ── prediction error signals        │
    │       ↓                                                 │
    │  [Neural Oscillation Attention] ── temporal pooling     │
    │       ↓                                                 │
    │  [Classification Head]                                  │
    └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import torch
import torch.nn as nn

from brainnn.decoder.layers import (
    CrossFrequencyCoupling,
    LateralInhibition,
    MultiScaleTemporalConv,
    NeuralOscillationAttention,
    PredictiveCodingBlock,
)


class SynapseFlowDecoder(nn.Module):
    """Neuromorphic EEG decoder with biologically-grounded inductive biases.

    Args:
        n_channels: number of EEG channels
        n_classes: number of output classes
        n_samples: time samples per trial
        hidden_dim: feature dimension per oscillatory scale
        n_scales: number of temporal scales (frequency resolutions)
        coupling_dim: cross-frequency coupling output dimension
        dropout: dropout rate
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_samples: int = 256,
        hidden_dim: int = 16,
        n_scales: int = 5,
        coupling_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Stage 1: spatial sharpening
        self.lateral_inhibition = LateralInhibition(n_channels)

        # Stage 2: multi-scale oscillatory decomposition
        self.temporal_conv = MultiScaleTemporalConv(
            n_channels, hidden_dim, scales=[3, 7, 15, 31, 63][:n_scales]
        )

        # Stage 3: cross-frequency coupling
        self.coupling = CrossFrequencyCoupling(n_scales, hidden_dim, coupling_dim)

        # Stage 4: predictive coding
        total_temporal = n_scales * hidden_dim + coupling_dim
        self.predictive_coding = PredictiveCodingBlock(total_temporal, hidden_dim * 2)

        # Stage 5: oscillation-gated temporal attention
        self.attention = NeuralOscillationAttention(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_channels, n_samples) raw EEG

        Returns:
            (batch, n_classes) logits
        """
        x = self.lateral_inhibition(x)

        scale_outputs = self.temporal_conv(x)

        coupling = self.coupling(scale_outputs)
        temporal = torch.cat(scale_outputs, dim=1)
        features = torch.cat([temporal, coupling], dim=1)

        features = self.predictive_coding(features)

        pooled = self.attention(features)

        return self.head(self.dropout(pooled))
