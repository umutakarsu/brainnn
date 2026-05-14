"""Cholinergic / noradrenergic precision control of attention scores.

Architectural note (Dayan-correction):
    Earlier this module was framed as "ADHD attention noise" driven by SNR derived
    from dopamine. Following Yu & Dayan (2005), precision (the inverse of attention
    noise) is more accurately attributed to the cholinergic/noradrenergic system:

        - Acetylcholine encodes *expected* uncertainty → sets the noise floor.
        - Norepinephrine encodes *unexpected* uncertainty / arousal → modulates
          volatility (Yerkes-Dodson-shaped: too low or too high both degrade precision).

    The numerical interface (`forward(scores, snr)`) is preserved for backward
    compatibility, but the `snr` argument is now computed from `BrainState.precision`
    (which derives from ACh and NE) — NOT from dopamine.

    Dopamine's role is value-of-attention gating, handled separately in modulation.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianNoiseInjector(nn.Module):
    """Injects Gaussian noise into attention scores, controlled by precision.

    High precision (high ACh + balanced NE) → little noise → crisp attention.
    Low precision (low ACh or extreme NE) → high noise → scattered attention.

    The `snr` argument is a [0, 1] precision value; it is sourced from
    `BrainState.precision`, which is computed from acetylcholine and
    norepinephrine — not from dopamine.
    """

    def __init__(self, max_noise_std: float = 0.5) -> None:
        super().__init__()
        self.max_noise_std = max_noise_std

    def forward(self, attention_scores: torch.Tensor, snr: float) -> torch.Tensor:
        """Add noise to attention scores based on current precision.

        Args:
            attention_scores: (batch, heads, seq_len, seq_len) or (batch, heads, dim)
            snr: precision in [0, 1]. 0 = maximum noise, 1 = no noise.
                 Source: `BrainState.precision` (derived from ACh + NE).
        """
        if not self.training and snr >= 0.99:
            return attention_scores

        # Noise std is inversely proportional to precision
        noise_std = self.max_noise_std * (1.0 - snr)

        if noise_std < 1e-6:
            return attention_scores

        noise = torch.randn_like(attention_scores) * noise_std
        return attention_scores + noise

    def forward_from_neuromodulators(
        self,
        attention_scores: torch.Tensor,
        acetylcholine: float,
        norepinephrine: float,
    ) -> torch.Tensor:
        """Explicit ACh+NE entry point — bypasses `BrainState.precision`.

        Useful for ablation studies that pharmacologically dissociate ACh and NE.

        Args:
            acetylcholine: [0, 1] — drives base precision.
            norepinephrine: [0, 1] — Yerkes-Dodson, optimal near 0.5.
        """
        # Reconstruct the same precision formula used in BrainState.precision
        ne_stability = max(0.0, 1.0 - abs(norepinephrine - 0.5) * 2.0)
        precision = max(0.0, min(1.0, acetylcholine * 0.7 + ne_stability * 0.3))
        return self.forward(attention_scores, precision)
