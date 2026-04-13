"""Abstract base classes for modality encoders and decoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from brainnn.core.config import ModalityConfig


class BaseModalityEncoder(nn.Module, ABC):
    """Abstract encoder that maps raw sensory input to a latent representation.

    Subclasses must expose intermediate layer outputs for lateral connections.
    """

    def __init__(self, config: ModalityConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_layers = config.num_layers

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        ...

    @abstractmethod
    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode input and return (final_output, [intermediate_outputs_per_layer]).

        The intermediate outputs are used by SynestheticNet for lateral connections.
        Each intermediate tensor should have shape (batch, latent_dim).
        """
        ...

    @abstractmethod
    def inject_lateral(self, layer_idx: int, lateral_input: torch.Tensor) -> None:
        """Buffer a lateral input to be added at a specific layer during the next forward pass."""
        ...


class BaseModalityDecoder(nn.Module, ABC):
    """Abstract decoder that maps latent representation back to sensory space."""

    def __init__(self, config: ModalityConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to sensory output."""
        ...
