"""Dopaminergic modulation — simulates dopamine dynamics affecting attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from brainnn.core.config import ADHDConfig
from brainnn.core.state import BrainState


class DopaminergicModulator(nn.Module):
    """Modulates attention head activity based on dopamine levels.

    Simulates how dopamine affects attention in ADHD:
    - Low dopamine → some heads randomly "go dark" (distraction)
    - High dopamine → heads amplify (hyperfocus)
    - Dopamine crash → sudden loss of all focus
    """

    def __init__(self, config: ADHDConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.hyperfocus_threshold = config.hyperfocus_threshold
        self.head_dropout_prob = config.head_dropout_prob

        # Learnable per-head dopamine sensitivity
        self.head_sensitivity = nn.Parameter(torch.ones(config.num_heads))

    def forward(
        self,
        attention_output: torch.Tensor,
        brain_state: BrainState,
    ) -> torch.Tensor:
        """Modulate multi-head attention output based on dopamine state.

        Args:
            attention_output: (batch, num_heads, seq_len, head_dim) or (batch, num_heads, dim)
            brain_state: current BrainState

        Returns:
            Modulated attention output with same shape
        """
        dopamine = brain_state.dopamine
        batch_size = attention_output.shape[0]

        if brain_state.is_hyperfocused:
            return self._hyperfocus_modulation(attention_output, dopamine)
        else:
            return self._distracted_modulation(attention_output, dopamine, batch_size)

    def _hyperfocus_modulation(
        self, attention_output: torch.Tensor, dopamine: float
    ) -> torch.Tensor:
        """In hyperfocus: amplify top heads, suppress others."""
        # Amplification factor based on how far above threshold
        amp_factor = 1.0 + (dopamine - self.hyperfocus_threshold) * 3.0

        # Top ~25% of heads get amplified, rest get suppressed
        sensitivity = self.head_sensitivity.abs()
        threshold = sensitivity.quantile(0.75)

        mask = (sensitivity >= threshold).float()
        # Amplify focused heads, suppress unfocused
        scale = mask * amp_factor + (1 - mask) * 0.1

        # Reshape scale to broadcast: (num_heads,) → (1, num_heads, 1, ...)
        for _ in range(attention_output.ndim - 2):
            scale = scale.unsqueeze(-1)
        scale = scale.unsqueeze(0)  # batch dim

        return attention_output * scale

    def _distracted_modulation(
        self, attention_output: torch.Tensor, dopamine: float, batch_size: int
    ) -> torch.Tensor:
        """In distracted state: randomly drop heads, scale by dopamine."""
        if self.training:
            # Dropout probability increases as dopamine decreases
            effective_dropout = self.head_dropout_prob * (1.0 - dopamine)
            # Per-batch, per-head dropout mask
            mask = torch.bernoulli(
                torch.full((batch_size, self.num_heads), 1.0 - effective_dropout,
                           device=attention_output.device)
            )
            # Reshape for broadcasting
            for _ in range(attention_output.ndim - 2):
                mask = mask.unsqueeze(-1)

            # Scale by dopamine level
            scale = dopamine * self.head_sensitivity.abs()
            for _ in range(attention_output.ndim - 2):
                scale = scale.unsqueeze(-1)
            scale = scale.unsqueeze(0)

            return attention_output * mask * scale
        else:
            # At eval, use expected value
            scale = dopamine * (1.0 - self.head_dropout_prob * (1.0 - dopamine))
            return attention_output * scale
