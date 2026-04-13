"""Configuration dataclasses for the neurodivergent brain simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ModalityType(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"


@dataclass
class ModalityConfig:
    """Configuration for a single sensory modality encoder/decoder."""

    modality_type: ModalityType
    input_channels: int
    latent_dim: int = 128
    num_layers: int = 4
    hidden_dim: int = 256


@dataclass
class SynesthesiaConfig:
    """Configuration for cross-modal lateral connections."""

    # Which layer indices have lateral connections (0-indexed)
    connection_layers: list[int] = field(default_factory=lambda: [0, 1, 2])
    # Base strength of cross-modal connections [0, 1]
    synesthesia_strength: float = 0.5
    # Whether connections are bidirectional
    bidirectional: bool = True
    # Learnable gating on/off
    learnable_gating: bool = True


@dataclass
class ADHDConfig:
    """Configuration for ADHD attention simulation."""

    # Number of attention heads
    num_heads: int = 8
    # Baseline dopamine level [0, 1]
    baseline_dopamine: float = 0.3
    # Dopamine decay rate per timestep
    dopamine_decay: float = 0.02
    # Threshold for hyperfocus activation
    hyperfocus_threshold: float = 0.8
    # Base signal-to-noise ratio (higher = more focused)
    base_snr: float = 1.0
    # Maximum noise magnitude
    max_noise_std: float = 0.5
    # Probability of random head dropout per step
    head_dropout_prob: float = 0.3
    # Fatigue accumulation rate
    fatigue_rate: float = 0.01


@dataclass
class BrainConfig:
    """Top-level configuration for the neurodivergent brain simulator."""

    # Modality configs — at least one required
    modalities: dict[ModalityType, ModalityConfig] = field(default_factory=dict)
    synesthesia: SynesthesiaConfig = field(default_factory=SynesthesiaConfig)
    adhd: ADHDConfig = field(default_factory=ADHDConfig)

    # Shared latent dimension across modalities
    latent_dim: int = 128
    # Simulation timestep duration (abstract units)
    dt: float = 0.1

    @classmethod
    def default_three_modality(cls) -> BrainConfig:
        """Create a default config with visual, auditory, and tactile modalities."""
        modalities = {
            ModalityType.VISUAL: ModalityConfig(
                modality_type=ModalityType.VISUAL,
                input_channels=3,  # RGB
                latent_dim=128,
                num_layers=3,
                hidden_dim=128,
            ),
            ModalityType.AUDITORY: ModalityConfig(
                modality_type=ModalityType.AUDITORY,
                input_channels=1,  # mono spectrogram
                latent_dim=128,
                num_layers=3,
                hidden_dim=128,
            ),
            ModalityType.TACTILE: ModalityConfig(
                modality_type=ModalityType.TACTILE,
                input_channels=1,  # pressure map
                latent_dim=128,
                num_layers=3,
                hidden_dim=128,
            ),
        }
        return cls(modalities=modalities, latent_dim=128)
