"""Behavioural read-out: reaction times, ex-Gaussian fitting, and validation.

Why this module exists
----------------------
A simulation of a condition is only worth anything if it makes a prediction you
could be wrong about. This module turns the internal state trajectory into
**reaction times**, fits the standard ex-Gaussian description to them, and
compares the result against the published ADHD literature.

The target finding
------------------
Increased intra-individual reaction-time variability is the most robustly
replicated cognitive signature of ADHD. When RT distributions are decomposed
into an ex-Gaussian — a Gaussian of mean ``mu`` and SD ``sigma`` convolved with
an exponential of mean ``tau`` — the group difference is carried
disproportionately by **tau**, the slow tail:

    tau  :  large ADHD-vs-control difference
    sigma:  moderate
    mu   :  small

See Leth-Steensen, Elbaz & Douglas (2000, *Acta Psychologica* 104:167-190) for
the original decomposition in ADHD, and Kofler et al. (2013, *Clinical
Psychology Review* 33:795-811) for the meta-analysis. The usual interpretation
is that the tail reflects occasional **attentional lapses** rather than a
uniform slowing (Sonuga-Barke & Castellanos 2007, *Neurosci Biobehav Rev*
31:977-986).

That ordering — ``tau > sigma > mu`` — is the prediction this model is tested
against. The lapse mechanism in :class:`brainnn.core.state.BrainState` is the
only thing in the model that could produce it, so the test is informative.

Honest scope
------------
The phenotype parameters in :mod:`brainnn.core.phenotypes` were **not fitted**
to reaction-time data. The claim being checked is qualitative: does the
mechanism reproduce the *ordering and rough relative magnitude* of the three
effect sizes? Absolute effect sizes should not be read as predictions, and a
match on ordering is weak evidence — it is one of a small number of possible
orderings. It is, however, a result the model could have failed to produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from brainnn.core.state import BrainState
from dataclasses import replace

from brainnn.core.phenotypes import Phenotype, get_phenotype


# ---------------------------------------------------------------------------
# Reaction-time generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RTParams:
    """Parameters mapping internal state onto a reaction time.

    The formulation is a simplified accumulator: evidence rises at a rate set
    by current attentional precision and value gating, and a response is made
    when it reaches a bound. Decision time is therefore ``boundary / drift``,
    which is hyperbolic in drift — a modest drop in precision produces a
    disproportionately long RT. That non-linearity, combined with stochastic
    lapses, is what generates a heavy right tail.
    """

    non_decision_ms: float = 180.0   # sensory encoding + motor execution
    boundary: float = 1.0            # evidence required for a response
    drift_scale: float = 8.5         # converts precision into drift rate
    drift_floor: float = 0.35        # prevents divide-by-zero on total collapse
    motor_noise_ms: float = 22.0     # residual Gaussian variability
    max_rt_ms: float = 4000.0        # responses beyond this are treated as misses


def trial_rt(state: BrainState, params: RTParams, rng: np.random.Generator) -> float:
    """Generate a single reaction time, in milliseconds, from the current state.

    Drift is driven mainly by precision (ACh/NE) and modulated by value gating
    (DA): you respond fast when you are both sharp and motivated.

    Precision is jittered trial to trial. Attention does not sit at a constant
    level punctuated by discrete lapses — it fluctuates continuously, and that
    continuous fluctuation is what populates the Gaussian (``sigma``) component
    of the distribution. Without it the fit degenerates, with ``tau`` absorbing
    essentially all the variance and ``sigma`` collapsing to zero.
    """
    jittered_precision = state.precision * (
        1.0 + rng.normal(0.0, state.phenotype.precision_jitter)
    )
    jittered_precision = float(np.clip(jittered_precision, 0.02, 1.0))

    # Dopaminergic gating affects engagement more than speed-once-engaged, so
    # its coupling to drift is deliberately weaker than precision's.
    motivation = 0.70 + 0.30 * state.value_gate
    drift = params.drift_scale * jittered_precision * motivation
    drift = max(drift, params.drift_floor)

    decision_ms = (params.boundary / drift) * 1000.0
    rt = params.non_decision_ms + decision_ms
    rt += rng.normal(0.0, params.motor_noise_ms)
    return float(np.clip(rt, 1.0, params.max_rt_ms))


def simulate_session(
    phenotype: Phenotype,
    n_trials: int = 300,
    seed: int = 0,
    params: Optional[RTParams] = None,
    dt: float = 1.0,
    dopamine_decay: float = 0.012,
    task_demand: float = 0.6,
    reward_every: int = 25,
    reward_size: float = 0.35,
) -> dict:
    """Run one simulated participant through a sustained-attention task.

    The task is deliberately boring — intermittent small rewards, constant
    demand — because that is the condition under which ADHD RT differences are
    largest. Returns the RT vector plus the state trajectory.
    """
    if params is None:
        params = RTParams()

    rng = np.random.default_rng(seed)
    state = BrainState(phenotype=phenotype, _rng=rng)
    state.reset()

    rts = np.empty(n_trials)
    lapses = np.zeros(n_trials, dtype=bool)
    precision = np.empty(n_trials)

    for i in range(n_trials):
        state.update(dt=dt, dopamine_decay=dopamine_decay, task_demand=task_demand)
        if reward_every and i > 0 and i % reward_every == 0:
            state.receive_stimulus(reward_size)

        rts[i] = trial_rt(state, params, rng)
        lapses[i] = state.lapse_active
        precision[i] = state.precision

    return {
        "rt": rts,
        "lapse": lapses,
        "precision": precision,
        "phenotype": phenotype.name,
        "state": state,
    }


# ---------------------------------------------------------------------------
# Ex-Gaussian fitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExGaussian:
    """Ex-Gaussian parameters: Normal(mu, sigma) convolved with Exponential(tau)."""

    mu: float
    sigma: float
    tau: float

    @property
    def mean(self) -> float:
        return self.mu + self.tau

    def __str__(self) -> str:
        return f"mu={self.mu:.1f} sigma={self.sigma:.1f} tau={self.tau:.1f}"


def _fit_exgaussian_moments(rt: np.ndarray) -> ExGaussian:
    """Closed-form method-of-moments fit. Used to initialise MLE and as a
    fallback if the optimiser fails.

    For ``X = N(mu, sigma) + Exp(tau)``:

        mean               = mu + tau
        variance           = sigma^2 + tau^2
        third central mom. = 2 * tau^3
    """
    m1 = float(rt.mean())
    m2 = float(rt.var(ddof=1))
    m3 = float(((rt - m1) ** 3).mean())
    tau = (m3 / 2.0) ** (1.0 / 3.0) if m3 > 0 else 0.0
    sigma_sq = m2 - tau ** 2
    sigma = float(np.sqrt(sigma_sq)) if sigma_sq > 0 else 0.0
    return ExGaussian(mu=float(m1 - tau), sigma=sigma, tau=float(tau))


def fit_exgaussian(rt: np.ndarray) -> ExGaussian:
    """Fit an ex-Gaussian by maximum likelihood.

    Uses ``scipy.stats.exponnorm`` (whose shape parameter is ``K = tau/sigma``)
    with a method-of-moments initial guess. MLE is what the ADHD reaction-time
    literature uses, and it matters here: when the exponential tail is large,
    method of moments squeezes ``sigma`` toward zero because it assigns all
    excess variance to ``tau``. MLE separates the two components properly.

    Falls back to the moments estimate if optimisation fails or returns a
    degenerate solution.

    Raises:
        ValueError: if fewer than 10 trials are supplied.
    """
    rt = np.asarray(rt, dtype=float)
    if rt.size < 10:
        raise ValueError(f"need at least 10 trials to fit, got {rt.size}")

    init = _fit_exgaussian_moments(rt)
    sigma0 = max(init.sigma, 1.0)
    tau0 = max(init.tau, 1.0)

    try:
        from scipy.stats import exponnorm
        K, loc, scale = exponnorm.fit(rt, tau0 / sigma0, loc=init.mu, scale=sigma0)
        mu, sigma, tau = float(loc), float(scale), float(K * scale)
        finite = all(np.isfinite(v) for v in (mu, sigma, tau))
        if finite and sigma > 0 and tau >= 0:
            return ExGaussian(mu=mu, sigma=sigma, tau=tau)
    except Exception:
        pass
    return init


def iiv_metrics(rt: np.ndarray) -> dict:
    """Conventional intra-individual variability measures.

    Reported alongside the ex-Gaussian because the older ADHD literature uses
    SD and coefficient of variation, and they are easier to interpret.
    """
    rt = np.asarray(rt, dtype=float)
    mean = float(rt.mean())
    sd = float(rt.std(ddof=1))
    return {
        "mean_rt": mean,
        "sd_rt": sd,
        "cv": sd / mean if mean > 0 else float("nan"),
        "p90_minus_p50": float(np.percentile(rt, 90) - np.percentile(rt, 50)),
    }


# ---------------------------------------------------------------------------
# Group comparison
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples, pooled SD."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def jitter_phenotype(
    pheno: Phenotype, rng: np.random.Generator, scale: float = 1.0
) -> Phenotype:
    """Perturb a phenotype to represent one individual.

    A phenotype is a group *centre*, not a person. Real participants differ
    substantially within a diagnostic category, and without that heterogeneity
    simulated groups separate perfectly and effect sizes become meaningless —
    every "participant" would be the same person under a different random seed.

    Perturbations are multiplicative and modest, applied to the parameters that
    individual differences plausibly load on.
    """
    def jit(value: float, sd: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(np.clip(value * (1.0 + rng.normal(0.0, sd * scale)), lo, hi))

    return replace(
        pheno,
        tonic_dopamine=jit(pheno.tonic_dopamine, 0.22),
        acetylcholine_baseline=jit(pheno.acetylcholine_baseline, 0.18),
        ne_tonic=jit(pheno.ne_tonic, 0.12),
        lapse_rate=jit(pheno.lapse_rate, 0.50, hi=0.5),
        lapse_depth=jit(pheno.lapse_depth, 0.25),
        precision_jitter=jit(pheno.precision_jitter, 0.20, hi=0.6),
        fatigue_ceiling=jit(pheno.fatigue_ceiling, 0.20),
    )


def simulate_group(
    phenotype_key: str,
    n_participants: int = 40,
    n_trials: int = 300,
    medication: str = "none",
    seed: int = 0,
    heterogeneity: float = 1.0,
) -> dict:
    """Simulate a group of participants and fit each one individually.

    Fitting per participant and then comparing parameters across groups is the
    standard procedure in this literature; pooling trials across people would
    confound within- and between-person variability.

    Args:
        heterogeneity: scales between-participant parameter spread. Set to 0 to
            disable individual differences (useful for isolating mechanism, but
            it inflates effect sizes enormously).
    """
    pheno = get_phenotype(phenotype_key, medication)
    mus, sigmas, taus, means, sds, lapse_rates = [], [], [], [], [], []

    for p in range(n_participants):
        p_rng = np.random.default_rng(seed * 10_000 + p + 7_777)
        indiv = (
            jitter_phenotype(pheno, p_rng, heterogeneity)
            if heterogeneity > 0 else pheno
        )
        session = simulate_session(indiv, n_trials=n_trials, seed=seed * 10_000 + p)
        fit = fit_exgaussian(session["rt"])
        iiv = iiv_metrics(session["rt"])
        mus.append(fit.mu)
        sigmas.append(fit.sigma)
        taus.append(fit.tau)
        means.append(iiv["mean_rt"])
        sds.append(iiv["sd_rt"])
        lapse_rates.append(float(session["lapse"].mean()))

    return {
        "phenotype": pheno.name,
        "mu": np.array(mus),
        "sigma": np.array(sigmas),
        "tau": np.array(taus),
        "mean_rt": np.array(means),
        "sd_rt": np.array(sds),
        "observed_lapse_rate": np.array(lapse_rates),
    }


def compare_groups(clinical: dict, control: dict) -> dict:
    """Effect sizes for each ex-Gaussian parameter, clinical minus control."""
    return {
        "mu": cohens_d(clinical["mu"], control["mu"]),
        "sigma": cohens_d(clinical["sigma"], control["sigma"]),
        "tau": cohens_d(clinical["tau"], control["tau"]),
        "mean_rt": cohens_d(clinical["mean_rt"], control["mean_rt"]),
        "sd_rt": cohens_d(clinical["sd_rt"], control["sd_rt"]),
    }


# Qualitative target from the meta-analytic literature. Stated as an ordering
# plus loose magnitude bands rather than point estimates, because the published
# values vary substantially with task and sample.
LITERATURE_PATTERN = {
    "ordering": ("tau", "sigma", "mu"),
    "bands": {
        "tau": (0.60, 1.30),    # large
        "sigma": (0.30, 0.90),  # moderate
        "mu": (0.00, 0.60),     # small
    },
    "source": (
        "Kofler et al. 2013, Clin Psychol Rev 33:795-811 (meta-analysis); "
        "Leth-Steensen et al. 2000, Acta Psychol 104:167-190"
    ),
    # Stated up front so nobody reads the ordering result as more than it is.
    "known_limitations": [
        "tau is overproduced: simulated ADHD/control tau ratios run ~3.5-5x "
        "against ~1.5-2.5x in published data, so the tau effect size lands "
        "roughly twice the meta-analytic band. The ordering is reproduced; "
        "the magnitude is not. Likely cause: decision time is hyperbolic in "
        "precision (boundary/drift), which amplifies lapses non-linearly. A "
        "shallower RT-precision coupling is the obvious thing to try next.",
        "Phenotype parameters were hand-set to reproduce the qualitative "
        "pattern, not fitted to participant data. Reproducing an ordering is "
        "weak evidence on its own; it is one of six possible orderings.",
        "The task is a single sustained-attention paradigm. The literature "
        "effect sizes vary considerably with task, so the bands are loose.",
    ],
}


def check_against_literature(effects: dict) -> dict:
    """Check whether simulated effect sizes reproduce the published pattern.

    Returns a dict with the ordering test, per-parameter band membership, and
    an overall pass/fail. The ordering test is the substantive one.
    """
    tau, sigma, mu = effects["tau"], effects["sigma"], effects["mu"]
    ordering_ok = tau > sigma > mu

    bands = LITERATURE_PATTERN["bands"]
    in_band = {k: bool(bands[k][0] <= effects[k] <= bands[k][1]) for k in bands}

    return {
        "ordering_expected": "tau > sigma > mu",
        "ordering_observed": " > ".join(
            k for k, _ in sorted(
                (("tau", tau), ("sigma", sigma), ("mu", mu)),
                key=lambda kv: kv[1], reverse=True,
            )
        ),
        "ordering_ok": bool(ordering_ok),
        "in_literature_band": in_band,
        "all_in_band": all(in_band.values()),
        "source": LITERATURE_PATTERN["source"],
    }
