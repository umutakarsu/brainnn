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
    """Configuration for ADHD attention simulation.

    Neuromodulator architecture (Yu & Dayan 2005 framework):
        Dopamine     -> value-of-attention gating (which heads to amplify)
        Acetylcholine -> expected uncertainty / precision (noise floor)
        Norepinephrine -> unexpected uncertainty / arousal (Yerkes-Dodson)
    """

    # Number of attention heads
    num_heads: int = 8

    # Dopamine (value axis)
    baseline_dopamine: float = 0.3
    dopamine_decay: float = 0.02
    hyperfocus_threshold: float = 0.8
    head_dropout_prob: float = 0.3

    # Acetylcholine / Norepinephrine (precision axis)
    baseline_acetylcholine: float = 0.5
    baseline_norepinephrine: float = 0.5
    max_noise_std: float = 0.5
    base_snr: float = 1.0  # retained for backward compat; unused in new code

    # General
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
