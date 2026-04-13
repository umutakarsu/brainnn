"""Lateral connections between modality encoder layers — the core of synesthesia."""

from __future__ import annotations

import torch
import torch.nn as nn


class LateralConnection(nn.Module):
    """Learnable connection between two modality encoder layers.

    Models the cross-modal "bleed" that characterizes synesthesia.
    A gating mechanism controls how much information flows between modalities.

    Input:  source tensor (batch, latent_dim)
    Output: projected tensor (batch, latent_dim) to be injected into target modality
    """

    def __init__(
        self,
        latent_dim: int,
        synesthesia_strength: float = 0.5,
        learnable_gating: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base_strength = synesthesia_strength

        # Linear projection from source to target feature space
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Learnable gate that controls connection strength
        if learnable_gating:
            self.gate = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Project source modality features for injection into target modality."""
        projected = self.projection(source)

        if self.gate is not None:
            gate_values = self.gate(source)
            projected = projected * gate_values

        return projected * self.base_strength
