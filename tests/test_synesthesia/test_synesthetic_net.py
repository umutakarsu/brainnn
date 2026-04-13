"""Tests for SynestheticNet and lateral connections."""

import torch
import pytest

from brainnn.core.config import BrainConfig, ModalityType, ModalityConfig, SynesthesiaConfig
from brainnn.synesthesia.lateral import LateralConnection
from brainnn.synesthesia.cross_modal import CrossModalProjectionBank
from brainnn.synesthesia.network import SynestheticNet


BATCH = 2
LATENT = 64


def _make_config(num_layers=3):
    modalities = {
        ModalityType.VISUAL: ModalityConfig(
            modality_type=ModalityType.VISUAL, input_channels=3,
            latent_dim=LATENT, num_layers=num_layers, hidden_dim=128,
        ),
        ModalityType.AUDITORY: ModalityConfig(
            modality_type=ModalityType.AUDITORY, input_channels=1,
            latent_dim=LATENT, num_layers=num_layers, hidden_dim=128,
        ),
        ModalityType.TACTILE: ModalityConfig(
            modality_type=ModalityType.TACTILE, input_channels=1,
            latent_dim=LATENT, num_layers=num_layers, hidden_dim=128,
        ),
    }
    return BrainConfig(
        modalities=modalities,
        latent_dim=LATENT,
        synesthesia=SynesthesiaConfig(
            connection_layers=[0, 1, 2][:num_layers],
            synesthesia_strength=0.5,
        ),
    )


class TestLateralConnection:
    def test_output_shape(self):
        conn = LateralConnection(latent_dim=LATENT, synesthesia_strength=0.5)
        x = torch.randn(BATCH, LATENT)
        out = conn(x)
        assert out.shape == (BATCH, LATENT)

    def test_strength_zero_gives_zero(self):
        conn = LateralConnection(latent_dim=LATENT, synesthesia_strength=0.0, learnable_gating=False)
        x = torch.randn(BATCH, LATENT)
        out = conn(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_gradient_flow(self):
        conn = LateralConnection(latent_dim=LATENT)
        x = torch.randn(BATCH, LATENT, requires_grad=True)
        out = conn(x)
        out.sum().backward()
        assert x.grad is not None


class TestCrossModalProjectionBank:
    def test_all_modalities_get_lateral(self):
        mods = [ModalityType.VISUAL, ModalityType.AUDITORY, ModalityType.TACTILE]
        bank = CrossModalProjectionBank(mods, LATENT, SynesthesiaConfig())
        features = {m: torch.randn(BATCH, LATENT) for m in mods}
        laterals = bank(features)
        assert set(laterals.keys()) == set(mods)
        for v in laterals.values():
            assert v.shape == (BATCH, LATENT)


class TestSynestheticNet:
    def test_forward_all_modalities(self):
        config = _make_config()
        net = SynestheticNet(config)
        inputs = {
            ModalityType.VISUAL: torch.randn(BATCH, 3, 32, 32),
            ModalityType.AUDITORY: torch.randn(BATCH, 1, 1024),
            ModalityType.TACTILE: torch.randn(BATCH, 1, 16, 16),
        }
        fused, per_mod = net(inputs)
        assert fused.shape == (BATCH, LATENT)
        assert len(per_mod) == 3

    def test_forward_subset_modalities(self):
        config = _make_config()
        net = SynestheticNet(config)
        inputs = {
            ModalityType.VISUAL: torch.randn(BATCH, 3, 32, 32),
            ModalityType.AUDITORY: torch.randn(BATCH, 1, 1024),
        }
        fused, per_mod = net(inputs)
        assert fused.shape == (BATCH, LATENT)
        assert len(per_mod) == 2

    def test_gradient_flows_through_laterals(self):
        config = _make_config()
        net = SynestheticNet(config)
        inputs = {
            ModalityType.VISUAL: torch.randn(BATCH, 3, 32, 32, requires_grad=True),
            ModalityType.AUDITORY: torch.randn(BATCH, 1, 1024, requires_grad=True),
            ModalityType.TACTILE: torch.randn(BATCH, 1, 16, 16, requires_grad=True),
        }
        fused, _ = net(inputs)
        fused.sum().backward()
        for v in inputs.values():
            assert v.grad is not None
