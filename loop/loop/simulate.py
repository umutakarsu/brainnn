"""Continuous pharmacokinetic / opponent-process curve model (§5).

This is deliberately NOT a learned model and NOT the discrete-timestep update
loop from the research engine — it is a continuous curve model, so the whole
product deploys with numpy alone (no torch).

Theoretical frame (cited in the About section):
    - Solomon & Corbit (1974), opponent-process theory of motivation:
      every primary affective response (the "a-process", here the dopamine
      spike) triggers a slower, opposite "b-process" that outlasts it — the
      dip below baseline. Repetition strengthens the b-process.
    - Koob & Le Moal (2001), allostatic model: with repetition the reward
      set-point drifts DOWN, so more is needed to feel normal. That is the
      `baseline_drift` term.
    - Berridge & Robinson: "wanting" (dopamine anticipation) is dissociable
      from "liking" (opioid relief). The anticipation peak can exceed the
      consumption peak — this is the binge-eating key insight.

A single episode's dopamine trace (relative to the current baseline) is the sum
of four deviations:
    1. anticipation  — a hump peaking BEFORE consumption ("wanting")
    2. consumption   — a sharp peak with exponential clearance ("liking")
    3. undershoot    — the slow opponent b-process dipping BELOW baseline
    4. (baseline)    — shifted down by accumulated tolerance across episodes
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Frequency / duration → episode-count mapping (drives SCREEN 2 tolerance)
# ---------------------------------------------------------------------------

FREQUENCY_EPISODES_PER_DAY: dict[str, float] = {
    "several_per_day": 4.0,
    "daily": 1.0,
    "few_per_week": 3.0 / 7.0,
    "bursts": 1.5,
}

DURATION_DAYS: dict[str, float] = {
    "weeks": 21.0,
    "months": 120.0,
    "years": 730.0,
}

# Allostatic erosion cannot take the baseline to zero — there is a floor.
MAX_EROSION = 0.6

# Dopamine baseline is normalised to 1.0; spikes are multiples of it.
BASELINE = 1.0
CORTISOL_BASELINE = 0.2


# ---------------------------------------------------------------------------
# Preset unit helpers (presets mix sec / min / hours)
# ---------------------------------------------------------------------------

def peak_time_hours(preset: dict) -> float:
    if "time_to_peak_sec" in preset:
        return preset["time_to_peak_sec"] / 3600.0
    if "time_to_peak_min" in preset:
        return preset["time_to_peak_min"] / 60.0
    return 0.25


def half_life_hours(preset: dict) -> float:
    return float(preset.get("half_life_hours", 1.0))


def crash_onset_hours(preset: dict) -> float:
    if "crash_onset_hours" in preset:
        return float(preset["crash_onset_hours"])
    if "crash_onset_min" in preset:
        return preset["crash_onset_min"] / 60.0
    # Fall back to a little after the peak.
    return peak_time_hours(preset) + 0.75


def recovery_hours(preset: dict) -> float:
    return float(preset.get("recovery_hours", 24.0))


def intensity_scale(intensity: float) -> float:
    """Map the 1-10 'pull on a bad day' slider to an amplitude multiplier.

    intensity 1 -> 0.7, intensity 5 -> ~1.0, intensity 10 -> 1.3.
    """
    intensity = max(1.0, min(10.0, float(intensity)))
    return 0.7 + (intensity - 1.0) / 9.0 * 0.6


# ---------------------------------------------------------------------------
# Baseline drift (allostatic set-point erosion) — Koob & Le Moal 2001
# ---------------------------------------------------------------------------

def baseline_drift(preset: dict, n_episodes: float) -> float:
    """Signed baseline shift after `n_episodes` repetitions (<= 0).

    Monotonically decreasing in n_episodes, saturating at -MAX_EROSION. The
    per-episode rate is the preset's `tolerance_per_episode`.
    """
    rate = float(preset.get("tolerance_per_episode", 0.02))
    n = max(0.0, float(n_episodes))
    return -MAX_EROSION * (1.0 - math.exp(-rate * n))


def episodes_from_frequency_duration(frequency_key: str, duration_key: str) -> float:
    """Total episodes implied by SCREEN 2's frequency + duration answers."""
    epd = FREQUENCY_EPISODES_PER_DAY.get(frequency_key, 1.0)
    days = DURATION_DAYS.get(duration_key, 120.0)
    return epd * days


# ---------------------------------------------------------------------------
# Single-episode curves
# ---------------------------------------------------------------------------

def episode_curve(
    t: np.ndarray,
    preset: dict,
    intensity: float = 5.0,
    baseline: float = BASELINE,
) -> np.ndarray:
    """Dopamine trace for one episode, onset at t=0. `t` is hours (>=0 region used).

    Returns an array the same shape as `t`. Guaranteed to peak above `baseline`
    and to dip below `baseline` somewhere on a sufficiently long window.
    """
    t = np.asarray(t, dtype=float)
    scale = intensity_scale(intensity)

    p_peak = peak_time_hours(preset)
    hl = half_life_hours(preset)
    onset_crash = crash_onset_hours(preset)
    rec = recovery_hours(preset)

    a_ant = (preset.get("anticipation_spike", 1.0) - 1.0) * scale
    a_con = (preset.get("consumption_spike", 1.0) - 1.0) * scale
    depth = preset.get("crash_depth", 0.5)

    # 1. Anticipation — "wanting" hump peaking before consumption.
    center_ant = p_peak * 0.55
    sigma_ant = max(p_peak * 0.5, 1e-3)
    anticipation = a_ant * np.exp(-((t - center_ant) ** 2) / (2.0 * sigma_ant ** 2))

    # 2. Consumption — gaussian rise to the peak, exponential clearance after.
    sigma_rise = max(p_peak / 2.5, 1e-3)
    rise = np.exp(-((t - p_peak) ** 2) / (2.0 * sigma_rise ** 2))
    decay = np.exp(-np.maximum(t - p_peak, 0.0) * math.log(2.0) / hl)
    consumption = a_con * np.where(t <= p_peak, rise, decay)

    # 3. Undershoot — slow opponent b-process, dips below baseline.
    #    Asymmetric bump: gaussian rise into the trough, exponential recovery out.
    tau_rec = max(rec / 3.0, 1e-3)
    sigma_down = max(onset_crash / 2.0, 1e-3)
    below_rise = np.exp(-((t - onset_crash) ** 2) / (2.0 * sigma_down ** 2))
    below_decay = np.exp(-np.maximum(t - onset_crash, 0.0) / tau_rec)
    bump = np.where(t <= onset_crash, below_rise, below_decay)
    bump = np.where(t >= 0.0, bump, 0.0)
    undershoot = -depth * baseline * bump

    return baseline + anticipation + consumption + undershoot


def cortisol_curve(t: np.ndarray, preset: dict, intensity: float = 5.0) -> np.ndarray:
    """Cortisol (stress) trace, onset at t=0.

    Cortisol rises as dopamine crashes — the post-episode stress spike. Amplitude
    is the preset's `cortisol_rebound` (modest default when absent, e.g. nicotine
    and caffeine which are not primarily a stress-rebound story).
    """
    t = np.asarray(t, dtype=float)
    scale = intensity_scale(intensity)
    rebound = preset.get("cortisol_rebound", 0.3) * scale

    onset_crash = crash_onset_hours(preset)
    rec = recovery_hours(preset)
    # Cortisol peaks a little after the dopamine trough begins.
    peak_c = onset_crash * 1.1
    tau_rec = max(rec / 3.0, 1e-3)
    sigma_down = max(peak_c / 2.0, 1e-3)
    rise = np.exp(-((t - peak_c) ** 2) / (2.0 * sigma_down ** 2))
    fall = np.exp(-np.maximum(t - peak_c, 0.0) / tau_rec)
    bump = np.where(t <= peak_c, rise, fall)
    bump = np.where(t >= 0.0, bump, 0.0)
    return CORTISOL_BASELINE + rebound * bump


# ---------------------------------------------------------------------------
# Timeline assembly (for SCREEN 3)
# ---------------------------------------------------------------------------

def timeline_window_hours(preset: dict) -> float:
    """A window long enough to show peak, crash, and recovery."""
    w = crash_onset_hours(preset) + recovery_hours(preset)
    return float(min(max(w, 6.0), 96.0))


def simulate_timeline(
    preset: dict,
    intensity: float = 5.0,
    n_episodes: float = 0.0,
    n_points: int = 600,
) -> dict:
    """One representative episode on the CURRENT (drifted) baseline.

    Returns arrays for plotting: time (hours), dopamine, cortisol, and the
    effective baseline line the dopamine curve is measured against.
    """
    window = timeline_window_hours(preset)
    t = np.linspace(0.0, window, n_points)

    drift = baseline_drift(preset, n_episodes)
    eff_baseline = BASELINE + drift

    dopamine = episode_curve(t, preset, intensity=intensity, baseline=eff_baseline)
    cortisol = cortisol_curve(t, preset, intensity=intensity)
    baseline_line = np.full_like(t, eff_baseline)

    return {
        "t": t,
        "dopamine": dopamine,
        "cortisol": cortisol,
        "baseline": baseline_line,
        "effective_baseline": eff_baseline,
        "drift": drift,
        "window_hours": window,
    }


# ---------------------------------------------------------------------------
# "What if?" comparisons (for SCREEN 4)
# ---------------------------------------------------------------------------

def simulate_recovery(
    preset: dict,
    n_episodes: float,
    days: int = 30,
    n_points: int = 240,
) -> dict:
    """Baseline recovery over `days` of no repetition (the most motivating chart).

    The eroded baseline relaxes back toward intact (1.0) with a time constant
    derived from the preset's recovery_hours (allostatic recovery is slower than
    a single episode's, so it is scaled into days).
    """
    d = np.linspace(0.0, float(days), n_points)
    start_drift = baseline_drift(preset, n_episodes)
    tau_days = min(max(recovery_hours(preset) / 24.0 * 4.0, 3.0), 20.0)
    baseline = BASELINE + start_drift * np.exp(-d / tau_days)
    return {
        "days": d,
        "baseline": baseline,
        "start_baseline": BASELINE + start_drift,
        "tau_days": tau_days,
    }


def compare_frequency(
    preset: dict,
    intensity: float,
    n_episodes_current: float,
    factor: float = 0.5,
) -> dict:
    """Baseline now vs. at a fraction (`factor`) of the current cumulative load."""
    now = BASELINE + baseline_drift(preset, n_episodes_current)
    reduced = BASELINE + baseline_drift(preset, n_episodes_current * factor)
    return {"current_baseline": now, "reduced_baseline": reduced, "factor": factor}


def compare_delay(
    preset: dict,
    intensity: float,
    n_episodes: float,
    delay_hours: float = 2.0,
    n_points: int = 600,
) -> dict:
    """Same episode, onset now vs. delayed — shows the curve shifted, not erased.

    The point is honest: delaying does not remove the loop, it moves the crash.
    Understanding the shape is the product, not a willpower prescription.
    """
    window = timeline_window_hours(preset) + delay_hours
    t = np.linspace(0.0, window, n_points)
    drift = baseline_drift(preset, n_episodes)
    eff = BASELINE + drift

    now = episode_curve(t, preset, intensity=intensity, baseline=eff)
    delayed = episode_curve(t - delay_hours, preset, intensity=intensity, baseline=eff)
    return {
        "t": t,
        "now": now,
        "delayed": delayed,
        "baseline": eff,
        "delay_hours": delay_hours,
    }
