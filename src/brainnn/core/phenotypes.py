"""Attentional phenotypes — parameter sets for neurotypical and ADHD profiles.

Why this module exists
----------------------
Earlier versions of this simulator had a single parameter set, which meant the
"ADHD simulation" could not actually contrast ADHD against anything. A model of
a condition has to be able to produce both the condition and its absence from
the same machinery, differing only in parameters. That is what this provides.

Theoretical basis
-----------------
The central computational account used here is the **tonic/phasic dopamine**
model (Grace 1991, *Neuroscience* 41:1-24; applied to ADHD in Grace 2001 and
Tripp & Wickens 2008, *J Child Psychol Psychiatry* 49:691-704):

    Tonic dopamine sets the background level and, via autoreceptor feedback,
    *suppresses* the size of phasic bursts. Lower tonic DA therefore
    disinhibits phasic responses: each immediate reward produces a relatively
    larger burst. The behavioural consequence is a steeper pull toward
    immediate reward and poorer tolerance of delay.

Supporting evidence for reduced dopaminergic tone in ADHD comes from receptor
and transporter imaging (Volkow et al. 2009, *JAMA* 302:1084-1091, reduced D2/D3
availability in the reward pathway; Volkow et al. 2007, *Arch Gen Psychiatry*).

The noradrenergic parameters follow adaptive gain theory (Aston-Jones & Cohen
2005, *Annu Rev Neurosci* 28:403-450): tonic NE sets arousal on an inverted-U,
while phasic NE bursts to task-relevant and unexpected events. Altered
phasic/tonic balance is one proposed mechanism for the attentional lapses that
dominate ADHD reaction-time distributions.

The cholinergic parameter follows Yu & Dayan (2005, *Neuron* 46:681-692), where
ACh encodes expected uncertainty and therefore sets attentional precision.

Honest scope
------------
These parameter values are **not fitted to individual-participant data**. They
are hand-set so that the model reproduces the *qualitative* reaction-time
signature reported in the ADHD literature (see `brainnn.behavior`), and their
relative ordering is what carries the scientific content — not their absolute
magnitudes. Treat a phenotype as a hypothesis about mechanism, not as a
measurement of any person.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Phenotype:
    """A parameter set describing one attentional profile.

    All neuromodulator levels are dimensionless and live in [0, 1].
    """

    name: str
    description: str

    # --- Dopaminergic (value / reward) ---
    # Background dopaminergic tone. Low tone is the core ADHD assumption.
    tonic_dopamine: float = 0.45
    # Multiplier on phasic bursts. Grace's model: low tonic disinhibits phasic,
    # so this should move *inversely* to tonic_dopamine across phenotypes.
    phasic_gain: float = 1.0
    # How fast a phasic burst decays back toward tonic, per unit dt.
    phasic_decay: float = 0.35

    # --- Cholinergic (expected uncertainty / precision) ---
    acetylcholine_baseline: float = 0.60
    # How strongly ACh is recruited by task demand.
    acetylcholine_gain: float = 0.30

    # --- Noradrenergic (unexpected uncertainty / arousal) ---
    ne_tonic: float = 0.50
    # Size of the phasic NE burst to a surprising event.
    ne_phasic_gain: float = 0.40
    ne_phasic_decay: float = 0.30

    # --- Attentional lapses ---
    # Per-step probability that precision transiently collapses. This is the
    # mechanism that produces the heavy right tail of the RT distribution.
    lapse_rate: float = 0.02
    # How far precision drops during a lapse (fraction of current precision).
    lapse_depth: float = 0.55
    # Continuous trial-to-trial fluctuation of precision between lapses, as a
    # fraction of its level. Lapses populate the exponential tail of the RT
    # distribution; this parameter populates the Gaussian spread (sigma).
    # Elevated in ADHD, consistent with attention fluctuating moment to moment
    # rather than failing only in discrete episodes.
    precision_jitter: float = 0.14

    # --- Effort / fatigue ---
    fatigue_rate: float = 0.010
    # Asymptote that fatigue approaches. The vigilance decrement in sustained
    # attention plateaus; it does not march to total exhaustion. Without a
    # ceiling the whole session is dominated by monotonic slowing, which shifts
    # the RT distribution instead of giving it a tail.
    fatigue_ceiling: float = 0.35
    # How strongly fatigue erodes precision.
    fatigue_sensitivity: float = 0.50

    def with_medication(self, med: "Medication") -> "Phenotype":
        """Return a copy of this phenotype with a medication effect applied."""
        return replace(
            self,
            name=f"{self.name} + {med.name}",
            description=f"{self.description} Under {med.name}: {med.description}",
            tonic_dopamine=_clip(self.tonic_dopamine + med.d_tonic_dopamine),
            phasic_gain=max(0.1, self.phasic_gain + med.d_phasic_gain),
            acetylcholine_baseline=_clip(
                self.acetylcholine_baseline + med.d_acetylcholine
            ),
            ne_tonic=_clip(self.ne_tonic + med.d_ne_tonic),
            lapse_rate=max(0.0, self.lapse_rate + med.d_lapse_rate),
        )


@dataclass(frozen=True)
class Medication:
    """A pharmacological modifier expressed as deltas on phenotype parameters.

    These are mechanism-level approximations of each drug's primary action, not
    dose-response models. Magnitudes are illustrative; the *direction* of each
    effect is what is grounded in pharmacology.
    """

    name: str
    description: str
    d_tonic_dopamine: float = 0.0
    d_phasic_gain: float = 0.0
    d_acetylcholine: float = 0.0
    d_ne_tonic: float = 0.0
    d_lapse_rate: float = 0.0
    # Metadata for display and grouping. Not used by with_medication().
    drug_class: str = ""
    brand_names: str = ""


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Phenotypes
# ---------------------------------------------------------------------------

NEUROTYPICAL = Phenotype(
    name="Neurotypical",
    description=(
        "Reference profile. Moderate dopaminergic tone keeps phasic reward "
        "responses proportionate, cholinergic tone supports stable precision, "
        "and attentional lapses are infrequent."
    ),
    tonic_dopamine=0.45,
    phasic_gain=1.00,
    acetylcholine_baseline=0.60,
    acetylcholine_gain=0.30,
    ne_tonic=0.50,
    ne_phasic_gain=0.40,
    lapse_rate=0.035,
    lapse_depth=0.20,
    precision_jitter=0.14,
    fatigue_rate=0.045,
    fatigue_ceiling=0.35,
    fatigue_sensitivity=0.50,
)

ADHD_COMBINED = Phenotype(
    name="ADHD — combined presentation",
    description=(
        "Reduced dopaminergic tone disinhibits phasic bursts, so immediate "
        "reward pulls disproportionately hard while sustained, unrewarded "
        "effort decays quickly. Lapse rate is elevated, producing the heavy "
        "slow tail seen in ADHD reaction-time distributions."
    ),
    tonic_dopamine=0.28,
    phasic_gain=1.60,          # inverse to tonic, per Grace's autoreceptor account
    acetylcholine_baseline=0.57,
    acetylcholine_gain=0.28,
    ne_tonic=0.42,
    ne_phasic_gain=0.52,       # larger surprise responses
    lapse_rate=0.060,
    lapse_depth=0.26,
    precision_jitter=0.26,
    fatigue_rate=0.055,
    fatigue_ceiling=0.45,
    fatigue_sensitivity=0.55,
)

ADHD_INATTENTIVE = Phenotype(
    name="ADHD — predominantly inattentive",
    description=(
        "Dopaminergic tone is reduced but less so than in the combined "
        "presentation; the dominant feature is a cholinergic precision deficit "
        "together with lower noradrenergic arousal. This yields frequent, deep "
        "lapses and a sluggish rather than impulsive profile."
    ),
    tonic_dopamine=0.33,
    phasic_gain=1.35,
    acetylcholine_baseline=0.55,   # precision deficit is the defining feature
    acetylcholine_gain=0.26,
    ne_tonic=0.38,                 # hypoarousal
    ne_phasic_gain=0.34,
    lapse_rate=0.070,              # most frequent lapses
    lapse_depth=0.30,
    precision_jitter=0.28,
    fatigue_rate=0.060,
    fatigue_ceiling=0.50,
    fatigue_sensitivity=0.58,
)

PHENOTYPES = {
    "neurotypical": NEUROTYPICAL,
    "adhd_combined": ADHD_COMBINED,
    "adhd_inattentive": ADHD_INATTENTIVE,
}


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------

# A note on scope: this model has no pharmacokinetics — no onset, no half-life,
# no dose. It represents each drug by the *direction* its primary mechanism
# pushes the phenotype parameters at steady state. Two consequences:
#   - Drugs that are the same molecule at the synapse (amphetamine and its
#     prodrug lisdexamfetamine) get near-identical parameters here, even though
#     their clinical feel differs a lot because of release kinetics.
#   - Nothing here says anything about who should take what. Direction of
#     effect is grounded in pharmacology; magnitude is illustrative.

# --- Stimulants: raise tonic dopamine, which re-engages autoreceptor feedback
#     and brings oversized phasic bursts down. This is why a stimulant can
#     reduce rather than increase impulsivity in ADHD.

METHYLPHENIDATE = Medication(
    name="Methylphenidate",
    drug_class="Stimulant",
    brand_names="Ritalin, Concerta, Medikinet, Equasym",
    description=(
        "blocks the dopamine and norepinephrine transporters, raising "
        "synaptic tone for both. Raising tonic DA re-engages autoreceptor "
        "feedback, which brings oversized phasic bursts back down — the "
        "reason a stimulant can reduce rather than increase impulsivity."
    ),
    d_tonic_dopamine=+0.15,
    d_phasic_gain=-0.35,
    d_ne_tonic=+0.10,
    d_lapse_rate=-0.035,
)

AMPHETAMINE = Medication(
    name="Amphetamine",
    drug_class="Stimulant",
    brand_names="Adderall, Attentin, Dexedrine (dexamfetamine)",
    description=(
        "blocks reuptake and additionally promotes vesicular release, so the "
        "rise in tonic tone is larger than with methylphenidate."
    ),
    d_tonic_dopamine=+0.20,
    d_phasic_gain=-0.40,
    d_ne_tonic=+0.14,
    d_lapse_rate=-0.040,
)

LISDEXAMFETAMINE = Medication(
    name="Lisdexamfetamine",
    drug_class="Stimulant (prodrug)",
    brand_names="Elvanse, Vyvanse",
    description=(
        "is an inactive prodrug that red blood cells slowly convert into "
        "dexamfetamine, so at the synapse it is amphetamine. The clinical "
        "difference — gradual onset, ~12-hour duration, smoother curve, lower "
        "misuse potential — is pharmacokinetic, which this model does not "
        "represent. Its parameters are therefore nearly the same as "
        "amphetamine's; the slightly gentler phasic reduction stands in for "
        "the absence of a sharp onset spike."
    ),
    d_tonic_dopamine=+0.18,
    d_phasic_gain=-0.36,
    d_ne_tonic=+0.12,
    d_lapse_rate=-0.038,
)

# --- Norepinephrine reuptake inhibitors: act mainly on the arousal/precision
#     axis. Prefrontal dopamine is cleared largely by NET rather than DAT, so
#     these raise prefrontal DA too, but without the striatal reward effect.

ATOMOXETINE = Medication(
    name="Atomoxetine",
    drug_class="Norepinephrine reuptake inhibitor",
    brand_names="Strattera",
    description=(
        "selectively inhibits the norepinephrine transporter. Because "
        "prefrontal dopamine is cleared largely by NET rather than DAT, this "
        "also raises prefrontal DA, but it acts mainly on the arousal/precision "
        "axis rather than on reward gating. Takes weeks to reach full effect."
    ),
    d_tonic_dopamine=+0.06,
    d_phasic_gain=-0.10,
    d_acetylcholine=+0.05,
    d_ne_tonic=+0.16,
    d_lapse_rate=-0.025,
)

VILOXAZINE = Medication(
    name="Viloxazine",
    drug_class="Norepinephrine reuptake inhibitor",
    brand_names="Qelbree",
    description=(
        "inhibits the norepinephrine transporter like atomoxetine, with "
        "additional serotonergic modulation (5-HT2B antagonism, 5-HT2C "
        "agonism). Approved for ADHD in 2021. In this model it behaves as a "
        "slightly milder atomoxetine; the serotonergic component is not "
        "represented."
    ),
    d_tonic_dopamine=+0.05,
    d_phasic_gain=-0.08,
    d_acetylcholine=+0.04,
    d_ne_tonic=+0.14,
    d_lapse_rate=-0.022,
)

# --- Alpha-2 adrenergic agonists: strengthen prefrontal signalling and reduce
#     noise. They act on precision, not on reward tone.

GUANFACINE = Medication(
    name="Guanfacine",
    drug_class="Alpha-2A agonist",
    brand_names="Intuniv, Tenex",
    description=(
        "is a selective alpha-2A adrenergic agonist that strengthens prefrontal "
        "network signalling at postsynaptic receptors. It acts on precision and "
        "noise rather than on reward tone, so it reduces lapses with little "
        "effect on phasic gain."
    ),
    d_acetylcholine=+0.10,
    d_ne_tonic=+0.04,
    d_lapse_rate=-0.030,
)

CLONIDINE = Medication(
    name="Clonidine",
    drug_class="Alpha-2 agonist (non-selective)",
    brand_names="Kapvay, Catapres",
    description=(
        "is a less selective alpha-2 agonist than guanfacine. Alongside the "
        "prefrontal effect it hits presynaptic autoreceptors on the locus "
        "coeruleus, which dampens noradrenergic firing — hence its sedating, "
        "blood-pressure-lowering profile. The model shows this as a drop in "
        "tonic NE rather than a rise. It tends to suit hyperactive and "
        "sleep-disturbed presentations better than hypoaroused, sluggish ones, "
        "and you can see why by applying it to the inattentive phenotype."
    ),
    d_acetylcholine=+0.06,
    d_ne_tonic=-0.06,
    d_lapse_rate=-0.020,
)

# --- Other agents used for ADHD, usually off-label or second line.

BUPROPION = Medication(
    name="Bupropion",
    drug_class="Norepinephrine–dopamine reuptake inhibitor",
    brand_names="Wellbutrin, Zyban, Elontril",
    description=(
        "weakly inhibits both the dopamine and norepinephrine transporters — "
        "the same mechanism as a stimulant, at much lower potency. Used "
        "off-label for ADHD, often when depression or nicotine dependence is "
        "also present. Effects build over weeks rather than hours."
    ),
    d_tonic_dopamine=+0.08,
    d_phasic_gain=-0.12,
    d_ne_tonic=+0.08,
    d_lapse_rate=-0.015,
)

MODAFINIL = Medication(
    name="Modafinil",
    drug_class="Wakefulness promoter",
    brand_names="Provigil, Modiodal, Vigil",
    description=(
        "is a weak dopamine transporter inhibitor whose main effect runs "
        "through orexin and histamine systems that promote wakefulness. It "
        "raises arousal and vigilance far more than it raises reward tone, so "
        "in this model it reduces lapses mostly via the norepinephrine axis. "
        "Used off-label for ADHD, more often in adults."
    ),
    d_tonic_dopamine=+0.05,
    d_acetylcholine=+0.04,
    d_ne_tonic=+0.14,
    d_lapse_rate=-0.022,
)

MEDICATIONS = {
    "none": None,
    # stimulants
    "methylphenidate": METHYLPHENIDATE,
    "amphetamine": AMPHETAMINE,
    "lisdexamfetamine": LISDEXAMFETAMINE,
    # norepinephrine reuptake inhibitors
    "atomoxetine": ATOMOXETINE,
    "viloxazine": VILOXAZINE,
    # alpha-2 agonists
    "guanfacine": GUANFACINE,
    "clonidine": CLONIDINE,
    # other
    "bupropion": BUPROPION,
    "modafinil": MODAFINIL,
}


def get_phenotype(name: str, medication: str = "none") -> Phenotype:
    """Look up a phenotype by key, optionally applying a medication.

    Args:
        name: one of ``PHENOTYPES``.
        medication: one of ``MEDICATIONS``.

    Raises:
        KeyError: if either key is unknown.
    """
    if name not in PHENOTYPES:
        raise KeyError(f"Unknown phenotype {name!r}. Options: {sorted(PHENOTYPES)}")
    if medication not in MEDICATIONS:
        raise KeyError(
            f"Unknown medication {medication!r}. Options: {sorted(MEDICATIONS)}"
        )
    pheno = PHENOTYPES[name]
    med = MEDICATIONS[medication]
    return pheno if med is None else pheno.with_medication(med)
