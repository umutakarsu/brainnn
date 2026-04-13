"""Visual modality encoder/decoder — CNN-based image processing."""

from __future__ import annotations

import torch
import torch.nn as nn

from brainnn.core.config import ModalityConfig
from brainnn.modalities.base import BaseModalityEncoder, BaseModalityDecoder


class VisualEncoder(BaseModalityEncoder):
    """CNN encoder for visual input (images / visual patterns).

    Input:  (batch, channels, H, W)  — e.g. (B, 3, 64, 64) for RGB
    Output: (batch, latent_dim)
    Intermediates: list of (batch, latent_dim) per layer
    """

    def __init__(self, config: ModalityConfig) -> None:
        super().__init__(config)
        hidden = config.hidden_dim

        # Build conv layers — each doubles channels, halves spatial dims
        layers: list[nn.Module] = []
        in_ch = config.input_channels
        for i in range(config.num_layers):
            out_ch = hidden // (2 ** (config.num_layers - 1 - i))
            out_ch = max(out_ch, 32)
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch
        self.conv_layers = nn.ModuleList(layers)

        # Projection from flattened conv output to latent_dim (per layer)
        self.layer_projections = nn.ModuleList([
            nn.LazyLinear(config.latent_dim) for _ in range(config.num_layers)
        ])
        # Final projection
        self.final_proj = nn.LazyLinear(config.latent_dim)

        # Lateral injection buffers
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
            # Flatten for projection
            flat = h.flatten(1)
            projected = self.layer_projections[i](flat)

            # Inject lateral input if available
            lateral = self._lateral_buffers.pop(i, None)
            if lateral is not None:
                projected = projected + lateral

            intermediates.append(projected)

        out = self.final_proj(h.flatten(1))
        self._lateral_buffers.clear()
        return out, intermediates

    def inject_lateral(self, layer_idx: int, lateral_input: torch.Tensor) -> None:
        self._lateral_buffers[layer_idx] = lateral_input


class VisualDecoder(BaseModalityDecoder):
    """Decoder that reconstructs visual output from latent representation.

    Input:  (batch, latent_dim)
    Output: (batch, channels, H, W)  — reconstructed image
    """

    def __init__(self, config: ModalityConfig, output_size: int = 64) -> None:
        super().__init__(config)
        self.output_size = output_size
        hidden = config.hidden_dim

        # Initial projection to spatial feature map
        init_size = output_size // (2 ** config.num_layers)
        init_size = max(init_size, 2)
        self.init_channels = hidden
        self.init_size = init_size
        self.fc = nn.Linear(config.latent_dim, hidden * init_size * init_size)

        # Build transposed conv layers
        layers: list[nn.Module] = []
        in_ch = hidden
        for i in range(config.num_layers):
            is_last = i == config.num_layers - 1
            out_ch = config.input_channels if is_last else in_ch // 2
            out_ch = max(out_ch, config.input_channels)
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.Identity() if is_last else nn.BatchNorm2d(out_ch),
                nn.Sigmoid() if is_last else nn.GELU(),
            ))
            in_ch = out_ch
        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.init_channels, self.init_size, self.init_size)
        return self.deconv_layers(h)
