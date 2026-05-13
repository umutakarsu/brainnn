"""Transformer-based EEG decoder.

Treats EEG as a sequence of time-window patches, similar to Vision Transformer
but adapted for multi-channel time series.

Architecture:
    Patch Embedding → Positional Encoding → Transformer Encoder → CLS Head
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGTransformer(nn.Module):
    """Transformer decoder for EEG classification.

    Splits EEG into non-overlapping time windows, embeds each as a token,
    and classifies using a [CLS] token.

    Args:
        n_channels: number of EEG channels
        n_classes: number of output classes
        n_samples: number of time samples per trial
        d_model: transformer embedding dimension
        nhead: number of attention heads
        num_layers: number of transformer encoder layers
        window_size: samples per patch/token
        dropout: dropout rate
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
        n_samples: int = 256,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        window_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        n_tokens = n_samples // window_size

        self.patch_embed = nn.Linear(n_channels * window_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_channels, n_samples) raw EEG

        Returns:
            (batch, n_classes) logits
        """
        B, C, T = x.shape
        n_patches = T // self.window_size

        # (B, C, n_patches, window_size) → (B, n_patches, C * window_size)
        x = x.unfold(2, self.window_size, self.window_size)
        x = x.permute(0, 2, 1, 3).reshape(B, n_patches, -1)

        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]

        x = self.norm(self.transformer(x))

        return self.head(x[:, 0])
