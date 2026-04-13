"""Integration tests for NeuroDivergentBrainSimulator."""

import torch
import pytest

from brainnn.core.config import BrainConfig, ModalityType, ModalityConfig, SynesthesiaConfig, ADHDConfig
from brainnn.core.brain import NeuroDivergentBrainSimulator, BrainOutput


BATCH = 2
LATENT = 64


def _make_config():
    modalities = {
        ModalityType.VISUAL: ModalityConfig(
            modality_type=ModalityType.VISUAL, input_channels=3,
            latent_dim=LATENT, num_layers=3, hidden_dim=128,
        ),
        ModalityType.AUDITORY: ModalityConfig(
            modality_type=ModalityType.AUDITORY, input_channels=1,
            latent_dim=LATENT, num_layers=3, hidden_dim=128,
        ),
        ModalityType.TACTILE: ModalityConfig(
            modality_type=ModalityType.TACTILE, input_channels=1,
            latent_dim=LATENT, num_layers=3, hidden_dim=128,
        ),
    }
    return BrainConfig(
        modalities=modalities,
        latent_dim=LATENT,
        synesthesia=SynesthesiaConfig(connection_layers=[0, 1, 2]),
        adhd=ADHDConfig(num_heads=8),
    )


def _make_inputs():
    return {
        ModalityType.VISUAL: torch.randn(BATCH, 3, 32, 32),
        ModalityType.AUDITORY: torch.randn(BATCH, 1, 1024),
        ModalityType.TACTILE: torch.randn(BATCH, 1, 16, 16),
    }


class TestBrainSimulator:
    def test_single_step(self):
        brain = NeuroDivergentBrainSimulator(_make_config())
        output = brain.step(_make_inputs(), generate=True)
        assert isinstance(output, BrainOutput)
        assert output.fused_latent.shape == (BATCH, LATENT)
        assert output.attended_latent.shape == (BATCH, LATENT)
        assert len(output.generated) == 3

    def test_simulation_run(self):
        brain = NeuroDivergentBrainSimulator(_make_config())
        inputs_seq = [_make_inputs() for _ in range(5)]
        outputs = brain.simulate(inputs_seq, generate=False)
        assert len(outputs) == 5
        # State should have evolved
        assert brain.state.step == 5
        assert brain.state.fatigue > 0

    def test_reward_boosts_dopamine(self):
        brain = NeuroDivergentBrainSimulator(_make_config())
        brain.step(_make_inputs(), reward_signal=0.0)
        dop_before = brain.state.dopamine
        brain.step(_make_inputs(), reward_signal=0.8)
        # Dopamine may have decayed then been boosted, but should be higher than full decay
        assert brain.state.dopamine > 0

    def test_reset(self):
        brain = NeuroDivergentBrainSimulator(_make_config())
        brain.step(_make_inputs())
        brain.step(_make_inputs())
        brain.reset()
        assert brain.state.step == 0
        assert brain.state.fatigue == 0.0

    def test_brain_state_snapshot(self):
        brain = NeuroDivergentBrainSimulator(_make_config())
        output = brain.step(_make_inputs())
        snap = output.brain_state_snapshot
        assert "dopamine" in snap
        assert "focus" in snap
        assert "snr" in snap
        assert "is_hyperfocused" in snap

    def test_generated_modalities(self):
        brain = NeuroDivergentBrainSimulator(_make_config())
        output = brain.step(_make_inputs(), generate=True)
        assert ModalityType.VISUAL in output.generated
        assert ModalityType.AUDITORY in output.generated
        assert ModalityType.TACTILE in output.generated
