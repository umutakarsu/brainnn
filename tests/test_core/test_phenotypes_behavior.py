"""Tests for the v0.3 ADHD model: phenotypes, tonic/phasic dynamics, lapses,
and the behavioural read-out.

These check *structural* properties the model must have for the ADHD account
to hold — directions, orderings, and the specific bugs fixed in v0.3 — rather
than exact parameter values, which are tuned and may drift.
"""

import numpy as np
import pytest

from brainnn.core.state import BrainState
from brainnn.core.phenotypes import (
    get_phenotype, PHENOTYPES, MEDICATIONS, NEUROTYPICAL, ADHD_COMBINED,
)
from brainnn.behavior import (
    simulate_session, simulate_group, fit_exgaussian, compare_groups,
    cohens_d, RTParams,
)


# ---------------------------------------------------------------------------
# Regressions for the two v0.2 bugs
# ---------------------------------------------------------------------------

def test_phasic_ne_burst_persists_across_updates():
    """v0.2 bug: update() overwrote NE, so a surprise burst survived 0 steps."""
    s = BrainState(_rng=np.random.default_rng(0))
    baseline = s.norepinephrine
    s.receive_stimulus(0.8)
    bumped = s.norepinephrine
    assert bumped > baseline
    s.update(dt=1.0, dopamine_decay=0.02)
    after_one = s.norepinephrine
    # Must decay gradually, not snap back below the pre-stimulus baseline.
    assert after_one > baseline
    assert after_one < bumped


def test_focus_weights_are_normalised():
    """v0.2 bug: precision*0.6 + value_gate*0.6 peaked at 1.2 and saturated."""
    s = BrainState(acetylcholine=1.0, norepinephrine=0.5, dopamine=1.0, fatigue=0.0)
    assert s.precision == pytest.approx(1.0)
    assert s.value_gate == pytest.approx(1.0)
    assert 0.5 * s.precision + 0.5 * s.value_gate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Dynamics must be stationary, not monotonic collapse
# ---------------------------------------------------------------------------

def test_fatigue_plateaus_below_one():
    s = BrainState(phenotype=NEUROTYPICAL, _rng=np.random.default_rng(0))
    s.reset()
    for _ in range(500):
        s.update(dt=1.0, dopamine_decay=0.012)
    assert s.fatigue < 1.0
    assert s.fatigue == pytest.approx(NEUROTYPICAL.fatigue_ceiling, abs=0.05)


def test_tonic_dopamine_is_homeostatic_not_draining():
    s = BrainState(phenotype=NEUROTYPICAL, _rng=np.random.default_rng(0))
    s.reset()
    for _ in range(500):
        s.update(dt=1.0, dopamine_decay=0.012)
    assert s.tonic_dopamine > 0.1
    assert s.tonic_dopamine == pytest.approx(NEUROTYPICAL.tonic_dopamine, abs=0.05)


# ---------------------------------------------------------------------------
# Grace tonic/phasic account
# ---------------------------------------------------------------------------

def test_lower_tonic_dopamine_gives_larger_phasic_response():
    """Core ADHD assumption: low tonic tone disinhibits phasic bursts."""
    nt = BrainState(phenotype=NEUROTYPICAL, _rng=np.random.default_rng(0)); nt.reset()
    ad = BrainState(phenotype=ADHD_COMBINED, _rng=np.random.default_rng(0)); ad.reset()
    assert ad.tonic_dopamine < nt.tonic_dopamine
    nt.receive_stimulus(0.4)
    ad.receive_stimulus(0.4)
    assert ad.phasic_dopamine > nt.phasic_dopamine
    assert ad.phasic_ratio > nt.phasic_ratio


def test_phenotype_phasic_gain_inverse_to_tonic():
    for key, pheno in PHENOTYPES.items():
        if key == "neurotypical":
            continue
        assert pheno.tonic_dopamine < NEUROTYPICAL.tonic_dopamine
        assert pheno.phasic_gain > NEUROTYPICAL.phasic_gain


# ---------------------------------------------------------------------------
# Lapses and the RT tail
# ---------------------------------------------------------------------------

def test_lapse_collapses_precision():
    s = BrainState(phenotype=ADHD_COMBINED, _rng=np.random.default_rng(0)); s.reset()
    s.lapse_active = False
    p_normal = s.precision
    s.lapse_active = True
    p_lapse = s.precision
    assert p_lapse < p_normal
    assert p_lapse == pytest.approx(p_normal * (1 - ADHD_COMBINED.lapse_depth), rel=1e-6)


def test_adhd_phenotypes_lapse_more_than_neurotypical():
    for key in ("adhd_combined", "adhd_inattentive"):
        assert PHENOTYPES[key].lapse_rate > NEUROTYPICAL.lapse_rate


def test_session_rt_is_right_skewed():
    """Lapses must produce a heavy right tail, not a symmetric or left-skewed
    distribution. This failed in the first implementation because fatigue
    marched to 1.0 and turned the session into a monotonic slowdown."""
    s = simulate_session(get_phenotype("adhd_combined"), n_trials=400, seed=0)
    rt = s["rt"]
    skew = ((rt - rt.mean()) ** 3).mean()
    assert skew > 0


def test_session_is_stationary():
    """First and last quarter of a session should have similar mean RT."""
    s = simulate_session(get_phenotype("neurotypical"), n_trials=400, seed=0)
    rt = s["rt"]
    q = len(rt) // 4
    assert abs(rt[:q].mean() - rt[-q:].mean()) < 0.25 * rt.mean()


# ---------------------------------------------------------------------------
# Ex-Gaussian fit
# ---------------------------------------------------------------------------

def test_fit_recovers_known_exgaussian():
    rng = np.random.default_rng(0)
    mu, sigma, tau = 400.0, 40.0, 80.0
    x = rng.normal(mu, sigma, 5000) + rng.exponential(tau, 5000)
    f = fit_exgaussian(x)
    assert f.mu == pytest.approx(mu, abs=15)
    assert f.sigma == pytest.approx(sigma, abs=10)
    assert f.tau == pytest.approx(tau, abs=15)


def test_fit_rejects_tiny_samples():
    with pytest.raises(ValueError):
        fit_exgaussian(np.array([1.0, 2.0, 3.0]))


def test_fit_parameters_are_sane():
    for key in PHENOTYPES:
        s = simulate_session(get_phenotype(key), n_trials=300, seed=1)
        f = fit_exgaussian(s["rt"])
        assert f.sigma > 0
        assert f.tau >= 0
        assert 100 < f.mu < 2000


# ---------------------------------------------------------------------------
# Group-level prediction: the published ADHD signature
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def groups():
    return {
        "nt": simulate_group("neurotypical", n_participants=40, n_trials=250, seed=1),
        "comb": simulate_group("adhd_combined", n_participants=40, n_trials=250, seed=2),
    }


def test_adhd_has_larger_tau_than_control(groups):
    assert groups["comb"]["tau"].mean() > groups["nt"]["tau"].mean()


def test_adhd_has_larger_sigma_than_control(groups):
    assert groups["comb"]["sigma"].mean() > groups["nt"]["sigma"].mean()


def test_tau_effect_exceeds_sigma_effect(groups):
    """The headline literature pattern: the slow tail carries the difference."""
    e = compare_groups(groups["comb"], groups["nt"])
    assert e["tau"] > e["sigma"]


def test_effect_sizes_are_positive_and_bounded(groups):
    e = compare_groups(groups["comb"], groups["nt"])
    for k in ("tau", "sigma"):
        assert 0 < e[k] < 6, f"{k} d={e[k]} implausible"


def test_heterogeneity_zero_inflates_effect_size():
    """Sanity check on the heterogeneity mechanism: removing individual
    differences should make groups separate far more sharply."""
    nt_h = simulate_group("neurotypical", 20, 200, seed=5, heterogeneity=1.0)
    ad_h = simulate_group("adhd_combined", 20, 200, seed=6, heterogeneity=1.0)
    nt_0 = simulate_group("neurotypical", 20, 200, seed=5, heterogeneity=0.0)
    ad_0 = simulate_group("adhd_combined", 20, 200, seed=6, heterogeneity=0.0)
    d_h = cohens_d(ad_h["tau"], nt_h["tau"])
    d_0 = cohens_d(ad_0["tau"], nt_0["tau"])
    assert d_0 > d_h


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------

def test_all_medications_resolve():
    for med in MEDICATIONS:
        p = get_phenotype("adhd_combined", med)
        assert 0 <= p.tonic_dopamine <= 1
        assert p.lapse_rate >= 0


def test_stimulant_raises_tonic_and_lowers_phasic_gain():
    base = get_phenotype("adhd_combined")
    mph = get_phenotype("adhd_combined", "methylphenidate")
    assert mph.tonic_dopamine > base.tonic_dopamine
    assert mph.phasic_gain < base.phasic_gain
    assert mph.lapse_rate < base.lapse_rate


def test_unknown_phenotype_raises():
    with pytest.raises(KeyError):
        get_phenotype("not_a_thing")
