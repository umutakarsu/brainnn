"""Tests for modality encoders — tensor shapes and lateral injection."""

import torch
import pytest

from brainnn.core.config import ModalityConfig, ModalityType
from brainnn.modalities.visual import VisualEncoder, VisualDecoder
from brainnn.modalities.auditory import AuditoryEncoder, AuditoryDecoder
from brainnn.modalities.tactile import TactileEncoder, TactileDecoder


BATCH = 2
LATENT = 64


@pytest.fixture
def visual_config():
    return ModalityConfig(modality_type=ModalityType.VISUAL, input_channels=3, latent_dim=LATENT, num_layers=3, hidden_dim=128)


@pytest.fixture
def auditory_config():
    return ModalityConfig(modality_type=ModalityType.AUDITORY, input_channels=1, latent_dim=LATENT, num_layers=3, hidden_dim=128)


@pytest.fixture
def tactile_config():
    return ModalityConfig(modality_type=ModalityType.TACTILE, input_channels=1, latent_dim=LATENT, num_layers=3, hidden_dim=128)


class TestVisualEncoder:
    def test_forward_shape(self, visual_config):
        enc = VisualEncoder(visual_config)
        x = torch.randn(BATCH, 3, 32, 32)
        out = enc(x)
        assert out.shape == (BATCH, LATENT)

    def test_intermediates(self, visual_config):
        enc = VisualEncoder(visual_config)
        x = torch.randn(BATCH, 3, 32, 32)
        out, intermediates = enc.forward_with_intermediates(x)
        assert len(intermediates) == visual_config.num_layers
        for inter in intermediates:
            assert inter.shape == (BATCH, LATENT)

    def test_lateral_injection(self, visual_config):
        enc = VisualEncoder(visual_config)
        x = torch.randn(BATCH, 3, 32, 32)
        lateral = torch.randn(BATCH, LATENT)
        enc.inject_lateral(0, lateral)
        out, intermediates = enc.forward_with_intermediates(x)
        assert out.shape == (BATCH, LATENT)


class TestVisualDecoder:
    def test_forward_shape(self, visual_config):
        dec = VisualDecoder(visual_config, output_size=32)
        z = torch.randn(BATCH, LATENT)
        out = dec(z)
        assert out.shape[0] == BATCH
        assert out.shape[1] == 3  # RGB


class TestAuditoryEncoder:
    def test_forward_shape(self, auditory_config):
        enc = AuditoryEncoder(auditory_config)
        x = torch.randn(BATCH, 1, 1024)
        out = enc(x)
        assert out.shape == (BATCH, LATENT)

    def test_intermediates(self, auditory_config):
        enc = AuditoryEncoder(auditory_config)
        x = torch.randn(BATCH, 1, 1024)
        out, intermediates = enc.forward_with_intermediates(x)
        assert len(intermediates) == auditory_config.num_layers


class TestAuditoryDecoder:
    def test_forward_shape(self, auditory_config):
        dec = AuditoryDecoder(auditory_config, output_length=1024)
        z = torch.randn(BATCH, LATENT)
        out = dec(z)
        assert out.shape[0] == BATCH
        assert out.shape[1] == 1


class TestTactileEncoder:
    def test_forward_shape(self, tactile_config):
        enc = TactileEncoder(tactile_config)
        x = torch.randn(BATCH, 1, 16, 16)
        out = enc(x)
        assert out.shape == (BATCH, LATENT)

    def test_intermediates(self, tactile_config):
        enc = TactileEncoder(tactile_config)
        x = torch.randn(BATCH, 1, 16, 16)
        out, intermediates = enc.forward_with_intermediates(x)
        assert len(intermediates) == tactile_config.num_layers


class TestTactileDecoder:
    def test_forward_shape(self, tactile_config):
        dec = TactileDecoder(tactile_config, output_size=16)
        z = torch.randn(BATCH, LATENT)
        out = dec(z)
        assert out.shape[0] == BATCH
        assert out.shape[1] == 1
