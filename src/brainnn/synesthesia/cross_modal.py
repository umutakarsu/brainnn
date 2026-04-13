"""Cross-modal projection matrices for mapping between different feature spaces."""

from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn

from brainnn.core.config import ModalityType, SynesthesiaConfig
from brainnn.synesthesia.lateral import LateralConnection


class CrossModalProjectionBank(nn.Module):
    """Manages all lateral connections between modality pairs at a given layer.

    For N modalities and bidirectional connections, creates N*(N-1) connections.
    For unidirectional, creates N*(N-1)/2 connections.
    """

    def __init__(
        self,
        modality_types: list[ModalityType],
        latent_dim: int,
        config: SynesthesiaConfig,
    ) -> None:
        super().__init__()
        self.modality_types = modality_types
        self.config = config

        # Create connections for all modality pairs
        self.connections = nn.ModuleDict()
        pairs = list(combinations(modality_types, 2))

        for source, target in pairs:
            key = f"{source.value}_to_{target.value}"
            self.connections[key] = LateralConnection(
                latent_dim=latent_dim,
                synesthesia_strength=config.synesthesia_strength,
                learnable_gating=config.learnable_gating,
            )
            if config.bidirectional:
                reverse_key = f"{target.value}_to_{source.value}"
                self.connections[reverse_key] = LateralConnection(
                    latent_dim=latent_dim,
                    synesthesia_strength=config.synesthesia_strength,
                    learnable_gating=config.learnable_gating,
                )

    def forward(
        self, features: dict[ModalityType, torch.Tensor]
    ) -> dict[ModalityType, torch.Tensor]:
        """Compute cross-modal lateral signals for each modality.

        Args:
            features: {modality_type: (batch, latent_dim)} intermediate features

        Returns:
            {modality_type: (batch, latent_dim)} accumulated lateral signals
        """
        lateral_signals: dict[ModalityType, torch.Tensor] = {}

        for target in self.modality_types:
            if target not in features:
                continue

            accumulated = torch.zeros_like(features[target])
            for source in self.modality_types:
                if source == target or source not in features:
                    continue

                key = f"{source.value}_to_{target.value}"
                if key in self.connections:
                    lateral = self.connections[key](features[source])
                    accumulated = accumulated + lateral

            lateral_signals[target] = accumulated

        return lateral_signals
