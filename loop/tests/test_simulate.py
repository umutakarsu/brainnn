"""Task 4 required tests: every preset produces a peak above baseline, an
undershoot below baseline, and monotonic baseline erosion with repetition.
"""

import numpy as np
import pytest

from loop.presets import PRESETS, GENERIC_PRESET
from loop import simulate

ALL_PRESETS = {**PRESETS, "generic": GENERIC_PRESET}


@pytest.mark.parametrize("key", list(ALL_PRESETS))
def test_peak_above_baseline(key):
    preset = ALL_PRESETS[key]
    out = simulate.simulate_timeline(preset, intensity=5.0, n_episodes=0.0)
    dopamine = out["dopamine"]
    baseline = out["effective_baseline"]
    assert dopamine.max() > baseline + 1e-6, f"{key}: no peak above baseline"


@pytest.mark.parametrize("key", list(ALL_PRESETS))
def test_undershoot_below_baseline(key):
    preset = ALL_PRESETS[key]
    out = simulate.simulate_timeline(preset, intensity=5.0, n_episodes=0.0)
    dopamine = out["dopamine"]
    baseline = out["effective_baseline"]
    assert dopamine.min() < baseline - 1e-6, f"{key}: no undershoot below baseline"


@pytest.mark.parametrize("key", list(ALL_PRESETS))
def test_monotonic_baseline_erosion(key):
    preset = ALL_PRESETS[key]
    episodes = np.array([0, 10, 25, 50, 100, 200, 500])
    drifts = np.array([simulate.baseline_drift(preset, n) for n in episodes])
    baselines = simulate.BASELINE + drifts
    diffs = np.diff(baselines)
    assert np.all(diffs < 0), f"{key}: baseline not monotonically eroding"
    assert baselines[0] == pytest.approx(simulate.BASELINE)


@pytest.mark.parametrize("key", list(ALL_PRESETS))
def test_erosion_saturates_above_floor(key):
    preset = ALL_PRESETS[key]
    drift = simulate.baseline_drift(preset, 1_000_000)
    assert drift > -simulate.MAX_EROSION - 1e-9
    assert drift <= 0.0


def test_wanting_exceeds_liking_for_binge():
    """Binge-eating key insight: anticipation peak > consumption peak."""
    preset = PRESETS["binge_eating"]
    t = np.linspace(0, 3, 2000)
    scale = simulate.intensity_scale(5.0)
    a_ant = (preset["anticipation_spike"] - 1.0) * scale
    a_con = (preset["consumption_spike"] - 1.0) * scale
    assert a_ant > a_con


def test_intensity_scales_amplitude():
    preset = PRESETS["nicotine"]
    low = simulate.simulate_timeline(preset, intensity=1.0)["dopamine"].max()
    high = simulate.simulate_timeline(preset, intensity=10.0)["dopamine"].max()
    assert high > low


def test_recovery_moves_baseline_up():
    preset = PRESETS["porn"]
    n = simulate.episodes_from_frequency_duration("daily", "months")
    rec = simulate.simulate_recovery(preset, n, days=30)
    assert rec["baseline"][-1] > rec["baseline"][0]
