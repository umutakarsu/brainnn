"""Tests for BrainState dynamics."""

import torch

from brainnn.core.state import BrainState


def test_initial_state():
    state = BrainState()
    assert state.arousal == 0.5
    assert state.focus == 0.5
    assert state.fatigue == 0.0
    assert state.step == 0


def test_update_increases_fatigue():
    state = BrainState()
    initial_fatigue = state.fatigue
    state.update(dt=1.0, dopamine_decay=0.02, fatigue_rate=0.1)
    assert state.fatigue > initial_fatigue


def test_update_decays_dopamine():
    state = BrainState(dopamine=0.5)
    state.update(dt=1.0, dopamine_decay=0.1, fatigue_rate=0.01)
    assert state.dopamine < 0.5


def test_stimulus_boosts_dopamine():
    state = BrainState(dopamine=0.3)
    state.receive_stimulus(0.5)
    assert state.dopamine > 0.3


def test_precision_property():
    """Precision is driven by ACh and NE (Yu & Dayan 2005), NOT by dopamine.

    This test verifies the v0.2 Dayan-correction architecture:
    high ACh + balanced NE -> high precision; fatigue caps precision;
    dopamine does NOT enter the precision computation.
    """
    # Max precision: ACh=1.0, NE balanced at 0.5, no fatigue
    state = BrainState(acetylcholine=1.0, norepinephrine=0.5, fatigue=0.0)
    assert state.precision == 1.0
    # Backward-compat alias should match
    assert state.snr == state.precision

    # Fatigue caps precision
    state.fatigue = 1.0
    assert state.precision < 1.0

    # Dopamine should NOT affect precision (pharmacological dissociation)
    state.fatigue = 0.0
    state.dopamine = 0.1
    p_low_da = state.precision
    state.dopamine = 0.9
    p_high_da = state.precision
    assert abs(p_low_da - p_high_da) < 1e-9


def test_value_gate_property():
    """Value-of-attention gate is driven by dopamine, capped by fatigue."""
    state = BrainState(dopamine=1.0, fatigue=0.0)
    assert state.value_gate == 1.0
    state.fatigue = 1.0
    assert state.value_gate < 1.0


def test_norepinephrine_yerkes_dodson():
    """NE has an inverted-U effect on precision (Yerkes-Dodson)."""
    base = BrainState(acetylcholine=0.5, fatigue=0.0)
    base.norepinephrine = 0.5
    p_balanced = base.precision
    base.norepinephrine = 0.0
    p_low = base.precision
    base.norepinephrine = 1.0
    p_high = base.precision
    assert p_balanced > p_low
    assert p_balanced > p_high


def test_hyperfocus_detection():
    state = BrainState(focus=0.9)
    assert state.is_hyperfocused
    state.focus = 0.3
    assert not state.is_hyperfocused


def test_to_tensor():
    state = BrainState(
        arousal=0.1, focus=0.2, fatigue=0.3,
        dopamine=0.4, acetylcholine=0.5, norepinephrine=0.6,
    )
    t = state.to_tensor()
    assert t.shape == (6,)
    assert torch.allclose(t, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))


def test_reset():
    state = BrainState(dopamine=0.9, fatigue=0.8, step=100)
    state._record()
    state.reset(baseline_dopamine=0.3)
    assert state.dopamine == 0.3
    assert state.fatigue == 0.0
    assert state.step == 0
    assert len(state.history["dopamine"]) == 0


def test_history_recording():
    state = BrainState()
    for _ in range(5):
        state.update(dt=0.1, dopamine_decay=0.02, fatigue_rate=0.01)
    assert len(state.history["dopamine"]) == 5
    assert len(state.history["focus"]) == 5


def test_values_stay_clamped():
    state = BrainState(dopamine=0.01)
    for _ in range(100):
        state.update(dt=1.0, dopamine_decay=0.5, fatigue_rate=0.5)
    assert 0.0 <= state.dopamine <= 1.0
    assert 0.0 <= state.fatigue <= 1.0
    assert 0.0 <= state.focus <= 1.0
    assert 0.0 <= state.arousal <= 1.0
