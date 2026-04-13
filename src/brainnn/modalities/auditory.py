"""Auditory modality encoder/decoder — 1D-Conv for spectrograms / waveforms."""

from __future__ import annotations

import torch
import torch.nn as nn

from brainnn.core.config import ModalityConfig
from brainnn.modalities.base import BaseModalityEncoder, BaseModalityDecoder


class AuditoryEncoder(BaseModalityEncoder):
    """1D convolutional encoder for auditory input.

    Input:  (batch, channels, time_steps)  — e.g. (B, 1, 16000) for mono audio
    Output: (batch, latent_dim)
    Intermediates: list of (batch, latent_dim) per layer
    """

    def __init__(self, config: ModalityConfig) -> None:
        super().__init__(config)
        hidden = config.hidden_dim

        layers: list[nn.Module] = []
        in_ch = config.input_channels
        for i in range(config.num_layers):
            out_ch = hidden // (2 ** (config.num_layers - 1 - i))
            out_ch = max(out_ch, 32)
            layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch
        self.conv_layers = nn.ModuleList(layers)

        self.layer_projections = nn.ModuleList([
            nn.LazyLinear(config.latent_dim) for _ in range(config.num_layers)
        ])
        self.final_proj = nn.LazyLinear(config.latent_dim)
        self._lateral_buffers: dict[int, torch.Tensor | None] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_intermediates(x)
        return out

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        intermediates = []
        h = x
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            flat = h.flatten(1)
            projected = self.layer_projections[i](flat)

            lateral = self._lateral_buffers.pop(i, None)
            if lateral is not None:
                projected = projected + lateral

            intermediates.append(projected)

        out = self.final_proj(h.flatten(1))
        self._lateral_buffers.clear()
        return out, intermediates

    def inject_lateral(self, layer_idx: int, lateral_input: torch.Tensor) -> None:
        self._lateral_buffers[layer_idx] = lateral_input


class AuditoryDecoder(BaseModalityDecoder):
    """Decoder that reconstructs auditory output from latent representation.

    Input:  (batch, latent_dim)
    Output: (batch, channels, time_steps)
    """

    def __init__(self, config: ModalityConfig, output_length: int = 16000) -> None:
        super().__init__(config)
        self.output_length = output_length
        hidden = config.hidden_dim

        init_length = output_length // (2 ** config.num_layers)
        init_length = max(init_length, 4)
        self.init_channels = hidden
        self.init_length = init_length
        self.fc = nn.Linear(config.latent_dim, hidden * init_length)

        layers: list[nn.Module] = []
        in_ch = hidden
        for i in range(config.num_layers):
            is_last = i == config.num_layers - 1
            out_ch = config.input_channels if is_last else in_ch // 2
            out_ch = max(out_ch, config.input_channels)
            layers.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=8, stride=2, padding=3),
                nn.Identity() if is_last else nn.BatchNorm1d(out_ch),
                nn.Tanh() if is_last else nn.GELU(),
            ))
            in_ch = out_ch
        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.init_channels, self.init_length)
        return self.deconv_layers(h)
