"""Noise injection for simulating low signal-to-noise ratio in ADHD attention."""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianNoiseInjector(nn.Module):
    """Injects Gaussian noise into attention scores, controlled by SNR.

    Higher SNR → less noise → more focused attention (neurotypical or hyperfocus).
    Lower SNR → more noise → scattered attention (ADHD distraction).
    """

    def __init__(self, max_noise_std: float = 0.5) -> None:
        super().__init__()
        self.max_noise_std = max_noise_std

    def forward(self, attention_scores: torch.Tensor, snr: float) -> torch.Tensor:
        """Add noise to attention scores based on current SNR.

        Args:
            attention_scores: (batch, heads, seq_len, seq_len) or (batch, heads, dim)
            snr: signal-to-noise ratio [0, 1]. 0 = maximum noise, 1 = no noise
        """
        if not self.training and snr >= 0.99:
            return attention_scores

        # Noise std is inversely proportional to SNR
        noise_std = self.max_noise_std * (1.0 - snr)

        if noise_std < 1e-6:
            return attention_scores

        noise = torch.randn_like(attention_scores) * noise_std
        return attention_scores + noise
