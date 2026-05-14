"""Dopaminergic value-of-attention gating.

Architectural note (Dayan-correction):
    This module previously was described as "dopamine controls precision/SNR."
    That conflated two distinct neuromodulatory roles. Per Yu & Dayan (2005)
    and subsequent value-of-attention literature (Westbrook & Braver; Frank et al.):

        Dopamine is NOT a precision controller. Precision lives in ACh / NE.

        Dopamine implements the COST-BENEFIT GATING of attention:
            "Is this head worth attending to?"
            → High DA = strong selection (lock onto valuable heads, suppress others)
            → Low DA = weak gating, uniform / scattered attention

    The class name `DopaminergicModulator` is retained for backward compatibility,
    but the mechanism is now interpreted as **value-gating** rather than precision
    control or noise modulation. Head sensitivity weights (`head_sensitivity`)
    represent learned expected-value-of-attending per head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from brainnn.core.config import ADHDConfig
from brainnn.core.state import BrainState


class DopaminergicModulator(nn.Module):
    """Value-of-attention gate driven by dopamine.

    Computes a multiplicative gain per attention head as a function of:
      1. Learned head value-of-attention (head_sensitivity) — what the head is "worth"
      2. Current dopaminergic gain (brain_state.value_gate) — how strongly to act on value

    Behaviour map:
        - High DA + high focus  → hyperfocus mode: amplify top-value heads,
                                  suppress low-value heads. (Strong cost-benefit gating.)
        - Low DA                → weak gating: heads are not strongly selected;
                                  output is uniformly attenuated and stochastic
                                  head dropout simulates "should I bother?" failures.

    Note: head amplification here is NOT a precision/SNR change. Precision noise
    is injected separately by `GaussianNoiseInjector` upstream, driven by ACh and NE.
    """

    def __init__(self, config: ADHDConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.hyperfocus_threshold = config.hyperfocus_threshold
        self.head_dropout_prob = config.head_dropout_prob

        # Learnable per-head value-of-attention weights.
        # (Previously named "dopamine sensitivity"; that was confusing — these
        # are expected-value-of-attending parameters that DA acts on.)
        self.head_sensitivity = nn.Parameter(torch.ones(config.num_heads))

    def forward(
        self,
        attention_output: torch.Tensor,
        brain_state: BrainState,
    ) -> torch.Tensor:
        """Apply value-of-attention gating to per-head outputs.

        Args:
            attention_output: (batch, num_heads, seq_len, head_dim) or (batch, num_heads, dim)
            brain_state: current BrainState. Uses `value_gate` (DA-derived) — not `precision`.

        Returns:
            Gated attention output with same shape.
        """
        # Use value_gate property — explicitly the dopaminergic gain, not raw dopamine.
        # This keeps the semantic that DA drives value gating, with mild fatigue penalty.
        value_gain = brain_state.value_gate
        batch_size = attention_output.shape[0]

        if brain_state.is_hyperfocused:
            return self._hyperfocus_modulation(attention_output, value_gain)
        else:
            return self._distracted_modulation(attention_output, value_gain, batch_size)

    def _hyperfocus_modulation(
        self, attention_output: torch.Tensor, value_gain: float
    ) -> torch.Tensor:
        """Strong value-gating mode: amplify high-value heads, suppress low-value heads.

        Args:
            value_gain: DA-derived gating gain in [0, 1].
                        High value_gain → strong "lock onto best" selection.
        """
        # Amplification factor scales with how far above hyperfocus threshold
        amp_factor = 1.0 + (value_gain - self.hyperfocus_threshold) * 3.0

        # Top ~25% of heads (by learned value-of-attention) get amplified, rest suppressed
        sensitivity = self.head_sensitivity.abs()  # head value weights
        threshold = sensitivity.quantile(0.75)

        mask = (sensitivity >= threshold).float()
        # Amplify high-value heads, suppress low-value heads
        scale = mask * amp_factor + (1 - mask) * 0.1

        # Reshape scale to broadcast: (num_heads,) → (1, num_heads, 1, ...)
        for _ in range(attention_output.ndim - 2):
            scale = scale.unsqueeze(-1)
        scale = scale.unsqueeze(0)  # batch dim

        return attention_output * scale

    def _distracted_modulation(
        self, attention_output: torch.Tensor, value_gain: float, batch_size: int
    ) -> torch.Tensor:
        """Weak value-gating mode: stochastic head dropout + uniform attenuation.

        Models "why bother?" failures of cost-benefit attention allocation when
        DA-driven value gain is low. NOT a precision/SNR effect — precision noise
        is handled separately by the cholinergic/noradrenergic noise injector.

        Args:
            value_gain: DA-derived gating gain in [0, 1].
                        Low value_gain → more frequent head dropout, lower overall scale.
        """
        if self.training:
            # Dropout probability rises as the value-of-attending signal falls
            effective_dropout = self.head_dropout_prob * (1.0 - value_gain)
            # Per-batch, per-head dropout mask
            mask = torch.bernoulli(
                torch.full((batch_size, self.num_heads), 1.0 - effective_dropout,
                           device=attention_output.device)
            )
            # Reshape for broadcasting
            for _ in range(attention_output.ndim - 2):
                mask = mask.unsqueeze(-1)

            # Scale by value-gain × per-head learned value weights
            scale = value_gain * self.head_sensitivity.abs()
            for _ in range(attention_output.ndim - 2):
                scale = scale.unsqueeze(-1)
            scale = scale.unsqueeze(0)

            return attention_output * mask * scale
        else:
            # At eval, use expected value
            scale = value_gain * (1.0 - self.head_dropout_prob * (1.0 - value_gain))
            return attention_output * scale
