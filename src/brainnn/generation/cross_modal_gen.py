"""Cross-modal generation — produce one modality's output from another's latent."""

from __future__ import annotations

import torch
import torch.nn as nn

from brainnn.core.config import BrainConfig, ModalityType
from brainnn.modalities.visual import VisualDecoder
from brainnn.modalities.auditory import AuditoryDecoder
from brainnn.modalities.tactile import TactileDecoder
from brainnn.modalities.base import BaseModalityDecoder

_DECODER_MAP: dict[ModalityType, type[BaseModalityDecoder]] = {
    ModalityType.VISUAL: VisualDecoder,
    ModalityType.AUDITORY: AuditoryDecoder,
    ModalityType.TACTILE: TactileDecoder,
}


class CrossModalGenerator(nn.Module):
    """Generates output in any target modality from a latent representation.

    This enables synesthetic experiences like:
    - Sound → Color (chromesthesia)
    - Visual → Sound (auditory synesthesia)
    - Touch → Color / Sound

    Uses a shared latent space + modality-specific decoders.
    """

    def __init__(self, config: BrainConfig) -> None:
        super().__init__()
        self.config = config

        # Create decoders for each modality
        self.decoders = nn.ModuleDict()
        for mod_type, mod_config in config.modalities.items():
            decoder_cls = _DECODER_MAP[mod_type]
            self.decoders[mod_type.value] = decoder_cls(mod_config)

        # Modality-specific latent transformations
        # Maps from shared latent space to modality-specific decoder input
        self.latent_transforms = nn.ModuleDict()
        for mod_type in config.modalities:
            self.latent_transforms[mod_type.value] = nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim),
                nn.GELU(),
                nn.Linear(config.latent_dim, config.latent_dim),
            )

    def generate(
        self,
        latent: torch.Tensor,
        target_modality: ModalityType,
    ) -> torch.Tensor:
        """Generate output in the target modality from a latent representation.

        Args:
            latent: (batch, latent_dim) shared latent representation
            target_modality: which modality to generate

        Returns:
            Generated output in the target modality's native format
        """
        transformed = self.latent_transforms[target_modality.value](latent)
        return self.decoders[target_modality.value](transformed)

    def generate_all(
        self,
        latent: torch.Tensor,
    ) -> dict[ModalityType, torch.Tensor]:
        """Generate output in all configured modalities.

        Args:
            latent: (batch, latent_dim) shared latent representation

        Returns:
            {modality_type: generated_output}
        """
        outputs = {}
        for mod_type in self.config.modalities:
            outputs[mod_type] = self.generate(latent, mod_type)
        return outputs

    def forward(
        self,
        latent: torch.Tensor,
        target_modality: ModalityType | None = None,
    ) -> dict[ModalityType, torch.Tensor] | torch.Tensor:
        if target_modality is not None:
            return self.generate(latent, target_modality)
        return self.generate_all(latent)
