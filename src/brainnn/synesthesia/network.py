"""SynestheticNet — the core network that orchestrates cross-modal processing."""

from __future__ import annotations

import torch
import torch.nn as nn

from brainnn.core.config import BrainConfig, ModalityType
from brainnn.modalities.base import BaseModalityEncoder
from brainnn.modalities.visual import VisualEncoder
from brainnn.modalities.auditory import AuditoryEncoder
from brainnn.modalities.tactile import TactileEncoder
from brainnn.synesthesia.cross_modal import CrossModalProjectionBank

_ENCODER_MAP = {
    ModalityType.VISUAL: VisualEncoder,
    ModalityType.AUDITORY: AuditoryEncoder,
    ModalityType.TACTILE: TactileEncoder,
}


class SynestheticNet(nn.Module):
    """Multi-modal network with lateral cross-modal connections at each layer.

    Unlike standard multimodal models that fuse at the end (late fusion),
    SynestheticNet allows information to flow between modalities at every
    encoder layer — simulating synesthetic cross-modal perception.

    Architecture:
        Görsel:    [V1] → [V2] → [V3] → [V4]
                     ↕       ↕       ↕
        İşitsel:   [A1] → [A2] → [A3] → [A4]
                     ↕       ↕       ↕
        Dokunsal:  [T1] → [T2] → [T3] → [T4]
    """

    def __init__(self, config: BrainConfig) -> None:
        super().__init__()
        self.config = config
        self.modality_types = list(config.modalities.keys())

        # Create encoders for each modality
        self.encoders = nn.ModuleDict()
        for mod_type, mod_config in config.modalities.items():
            encoder_cls = _ENCODER_MAP[mod_type]
            self.encoders[mod_type.value] = encoder_cls(mod_config)

        # Create cross-modal projection banks for each connected layer
        self.projection_banks = nn.ModuleDict()
        for layer_idx in config.synesthesia.connection_layers:
            self.projection_banks[str(layer_idx)] = CrossModalProjectionBank(
                modality_types=self.modality_types,
                latent_dim=config.latent_dim,
                config=config.synesthesia,
            )

        # Fusion layer — combines all modality outputs into a single representation
        num_modalities = len(config.modalities)
        self.fusion = nn.Sequential(
            nn.Linear(config.latent_dim * num_modalities, config.latent_dim * 2),
            nn.GELU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )

    def forward(
        self,
        inputs: dict[ModalityType, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[ModalityType, torch.Tensor]]:
        """Process multi-modal inputs with synesthetic lateral connections.

        Args:
            inputs: {modality_type: raw_input_tensor}

        Returns:
            (fused_representation, {modality_type: per_modality_output})
        """
        return self._forward_synesthetic(inputs)

    def _forward_synesthetic(
        self,
        inputs: dict[ModalityType, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[ModalityType, torch.Tensor]]:
        """Layer-by-layer forward pass with lateral injections."""
        active_modalities = [m for m in self.modality_types if m in inputs]

        # Step 1: Run each encoder layer-by-layer, injecting laterals at each step
        # We need to manually step through layers to synchronize lateral connections
        num_layers = self.config.modalities[active_modalities[0]].num_layers

        # Initialize hidden states
        current: dict[ModalityType, torch.Tensor] = {m: inputs[m] for m in active_modalities}
        all_intermediates: dict[ModalityType, list[torch.Tensor]] = {m: [] for m in active_modalities}

        for layer_idx in range(num_layers):
            # Run each encoder's single layer
            layer_outputs: dict[ModalityType, torch.Tensor] = {}
            for mod in active_modalities:
                encoder: BaseModalityEncoder = self.encoders[mod.value]
                # Run the specific layer
                h = encoder.conv_layers[layer_idx](current[mod])
                flat = h.flatten(1)
                projected = encoder.layer_projections[layer_idx](flat)
                layer_outputs[mod] = projected
                current[mod] = h  # Keep spatial representation for next conv layer

            # Apply lateral connections at this layer if configured
            if str(layer_idx) in self.projection_banks:
                lateral_signals = self.projection_banks[str(layer_idx)](layer_outputs)
                for mod in active_modalities:
                    if mod in lateral_signals:
                        layer_outputs[mod] = layer_outputs[mod] + lateral_signals[mod]

            for mod in active_modalities:
                all_intermediates[mod].append(layer_outputs[mod])

        # Step 2: Final projection per modality
        per_modality_outputs: dict[ModalityType, torch.Tensor] = {}
        for mod in active_modalities:
            encoder = self.encoders[mod.value]
            final = encoder.final_proj(current[mod].flatten(1))
            per_modality_outputs[mod] = final

        # Step 3: Fuse all modality outputs
        # Pad missing modalities with zeros
        batch_size = next(iter(per_modality_outputs.values())).shape[0]
        fused_parts = []
        for mod in self.modality_types:
            if mod in per_modality_outputs:
                fused_parts.append(per_modality_outputs[mod])
            else:
                fused_parts.append(torch.zeros(batch_size, self.config.latent_dim,
                                               device=next(iter(per_modality_outputs.values())).device))

        fused = self.fusion(torch.cat(fused_parts, dim=-1))

        return fused, per_modality_outputs
