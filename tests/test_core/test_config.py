"""Tests for core configuration."""

from brainnn.core.config import BrainConfig, ModalityType, ModalityConfig, SynesthesiaConfig, ADHDConfig


def test_default_three_modality():
    config = BrainConfig.default_three_modality()
    assert len(config.modalities) == 3
    assert ModalityType.VISUAL in config.modalities
    assert ModalityType.AUDITORY in config.modalities
    assert ModalityType.TACTILE in config.modalities
    assert config.latent_dim == 128


def test_modality_config():
    mc = ModalityConfig(modality_type=ModalityType.VISUAL, input_channels=3)
    assert mc.latent_dim == 128
    assert mc.num_layers == 4


def test_adhd_config_defaults():
    cfg = ADHDConfig()
    assert 0 < cfg.baseline_dopamine < 1
    assert cfg.num_heads == 8
    assert cfg.hyperfocus_threshold == 0.8


def test_synesthesia_config_defaults():
    cfg = SynesthesiaConfig()
    assert cfg.bidirectional is True
    assert cfg.learnable_gating is True
    assert len(cfg.connection_layers) == 3
