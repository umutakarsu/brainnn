"""BrainState — dynamic state tracking for the neurodivergent brain simulator.

Architectural note (v0.2, Dayan correction)
-------------------------------------------
Earlier versions conflated several attentional functions into the dopamine
signal. Following Yu & Dayan (2005, *Neuron* 46:681-692) this module separates:

    - **Acetylcholine (ACh)** — expected uncertainty, hence attentional
      *precision*. High ACh means the system knows how noisy things are and can
      sharpen accordingly.
    - **Norepinephrine (NE)** — unexpected uncertainty and arousal. Tonic NE
      sets a background level on an inverted-U; phasic NE bursts to surprising
      events (Aston-Jones & Cohen 2005, *Annu Rev Neurosci* 28:403-450).
    - **Dopamine (DA)** — value-of-attention gating. *Not* a precision control.

v0.3 — tonic/phasic dynamics and phenotypes
-------------------------------------------
Three things were wrong or missing in v0.2, all fixed here:

1. **NE had no phasic dynamics.** ``receive_stimulus`` raised NE, but the very
   next ``update`` overwrote it with an algebraic function of arousal and
   fatigue, so the surprise response survived zero timesteps. NE is now a
   genuine state variable: a tonic level plus a phasic component that decays.

2. **Dopamine had no tonic/phasic separation.** The tonic/phasic distinction
   (Grace 1991, *Neuroscience* 41:1-24) is the core computational account of
   ADHD: low tonic DA disinhibits phasic bursts via autoreceptor feedback, so
   immediate reward pulls disproportionately hard. Without the split the model
   could not express this at all. ``dopamine`` is now the effective level
   ``tonic + phasic``, with the components tracked separately.

3. **``focus`` weights did not sum to one.** The expression
   ``precision * 0.6 + value_gate * 0.6`` peaked at 1.2, so focus saturated and
   the hyperfocus threshold fired far too readily. Weights are now normalised.

Also added: **attentional lapses**, a per-step stochastic collapse of precision.
This is the mechanism that generates the heavy right tail of the reaction-time
distribution, which is the most robustly replicated behavioural signature of
ADHD (see ``brainnn.behavior``).

Backward compatibility
----------------------
``dopamine``, ``norepinephrine`` and ``acetylcholine`` remain plain settable
fields holding the *effective* level, so existing code and tests that assign to
them directly continue to work. If you construct a state with an explicit
``dopamine`` or ``norepinephrine`` and do not specify the corresponding tonic
component, the tonic level is initialised to match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
except ImportError:  # torch is optional — the state model is pure Python
    torch = None

from brainnn.core.phenotypes import Phenotype, NEUROTYPICAL


@dataclass
class BrainState:
    """Dynamic internal state of the simulated brain.

    All neuromodulator levels are dimensionless and clamped to [0, 1].

    Semantics:
        acetylcholine  → expected uncertainty / attentional precision
        norepinephrine → unexpected uncertainty / arousal (tonic + phasic)
        dopamine       → value-of-attention gating (tonic + phasic)
    """

    arousal: float = 0.5
    focus: float = 0.5
    fatigue: float = 0.0

    # Effective neuromodulator levels. These are what the rest of the model
    # reads, and they remain directly settable for backward compatibility.
    dopamine: float = 0.3
    acetylcholine: float = 0.5
    norepinephrine: float = 0.5

    # Generative components. When left as None they are initialised from the
    # corresponding effective level in __post_init__.
    tonic_dopamine: Optional[float] = None
    phasic_dopamine: float = 0.0
    ne_tonic: Optional[float] = None
    ne_phasic: float = 0.0

    # Phenotype supplying the parameters that differentiate ADHD from
    # neurotypical. Defaults to the neurotypical reference profile.
    phenotype: Phenotype = NEUROTYPICAL

    # True on steps where precision has transiently collapsed.
    lapse_active: bool = False

    step: int = 0

    history: dict[str, list[float]] = field(default_factory=lambda: {
        "arousal": [],
        "focus": [],
        "fatigue": [],
        "dopamine": [],
        "acetylcholine": [],
        "norepinephrine": [],
        "tonic_dopamine": [],
        "phasic_dopamine": [],
        "precision": [],
        "value_gate": [],
        "lapse": [],
    })

    # Not part of equality or repr — only used for lapse sampling.
    _rng: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tonic_dopamine is None:
            self.tonic_dopamine = self.dopamine
        if self.ne_tonic is None:
            self.ne_tonic = self.norepinephrine
        if self._rng is None:
            import numpy as np
            self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    @property
    def precision(self) -> float:
        """Attentional precision, from the cholinergic and noradrenergic systems.

        Per Yu & Dayan (2005):
            - High ACh → good estimate of expected uncertainty → high precision.
            - NE acts on an inverted-U (Yerkes-Dodson): mid-range is optimal,
              while both hypoarousal and jitteriness degrade precision.
            - Fatigue erodes the capacity to sustain precision, with
              phenotype-dependent sensitivity.
            - During an attentional lapse precision transiently collapses.

        Dopamine deliberately does not appear here. It gates *what* is worth
        attending to, not *how sharply* you attend.
        """
        ne_stability = max(0.0, 1.0 - abs(self.norepinephrine - 0.5) * 2.0)
        raw = self.acetylcholine * 0.7 + ne_stability * 0.3
        raw *= 1.0 - self.fatigue * self.phenotype.fatigue_sensitivity
        if self.lapse_active:
            raw *= 1.0 - self.phenotype.lapse_depth
        return self._clamp(raw)

    @property
    def snr(self) -> float:
        """Backward-compatible alias for :attr:`precision`."""
        return self.precision

    @property
    def value_gate(self) -> float:
        """Dopaminergic value-of-attention gate.

        High effective DA produces strong selection — lock onto high-value
        targets and suppress the rest. Low DA leaves gating weak and attention
        scattered. Fatigue reduces the willingness to spend effort at all.
        """
        return self._clamp(self.dopamine * (1.0 - self.fatigue * 0.3))

    @property
    def is_hyperfocused(self) -> bool:
        return self.focus > 0.8

    @property
    def phasic_ratio(self) -> float:
        """Phasic dopamine as a fraction of total effective dopamine.

        This is the quantity that most directly expresses the Grace account of
        ADHD: a high ratio means behaviour is dominated by immediate reward
        bursts rather than by sustained tone.
        """
        total = self.tonic_dopamine + self.phasic_dopamine
        return 0.0 if total <= 0 else self.phasic_dopamine / total

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        dopamine_decay: float,
        fatigue_rate: Optional[float] = None,
        task_demand: float = 0.5,
    ) -> None:
        """Advance the state by one timestep.

        Args:
            dt: timestep length in arbitrary units.
            dopamine_decay: rate at which tonic dopamine falls in the absence of
                reward. Unrewarded effort is costly.
            fatigue_rate: accumulation rate. Defaults to the phenotype's value.
            task_demand: in [0, 1]. How much precision the current task calls
                for; drives cholinergic recruitment.
        """
        p = self.phenotype
        if fatigue_rate is None:
            fatigue_rate = p.fatigue_rate

        # Tonic dopamine relaxes toward the phenotype's set point. "Tonic"
        # means a maintained background level, so it must be homeostatic — an
        # unconditional decay would drain it to zero over a long session and
        # make the whole run a monotonic slowdown rather than a stationary
        # process with variability.
        self.tonic_dopamine = self._clamp(
            self.tonic_dopamine
            + (p.tonic_dopamine - self.tonic_dopamine) * dopamine_decay * dt
        )

        # Phasic dopamine decays fast back toward zero. Crucially this is a
        # decay, not an overwrite, so a burst persists for several steps.
        self.phasic_dopamine *= math.exp(-p.phasic_decay * dt)

        self.dopamine = self._clamp(self.tonic_dopamine + self.phasic_dopamine)

        # Fatigue approaches a ceiling rather than accumulating without bound.
        # The vigilance decrement in sustained attention plateaus.
        self.fatigue = self._clamp(
            self.fatigue + fatigue_rate * (p.fatigue_ceiling - self.fatigue) * dt
        )

        # Acetylcholine is recruited toward a demand-dependent target and
        # eroded by fatigue. Previously its drift rate was tied to fatigue_rate,
        # which conflated two unrelated parameters.
        ach_target = p.acetylcholine_baseline + p.acetylcholine_gain * task_demand
        self.acetylcholine = self._clamp(
            self.acetylcholine
            + (ach_target - self.acetylcholine) * 0.25 * dt
            - self.fatigue * 0.05 * dt
        )

        # Norepinephrine: tonic level modulated by arousal and fatigue, plus a
        # decaying phasic component carrying the surprise response.
        self.ne_phasic *= math.exp(-p.ne_phasic_decay * dt)
        self.norepinephrine = self._clamp(
            p.ne_tonic
            + self.ne_phasic
            + (self.arousal - 0.5) * 0.20
            - self.fatigue * 0.20
        )

        # Sample an attentional lapse for this step. Read by `precision`.
        self.lapse_active = bool(self._rng.random() < p.lapse_rate)

        # Focus combines precision and value gating with normalised weights.
        self.focus = self._clamp(
            0.5 * self.precision + 0.5 * self.value_gate - self.fatigue * 0.25
        )

        self.arousal = self._clamp(0.3 + self.dopamine * 0.5 - self.fatigue * 0.3)

        self.step += 1
        self._record()

    def receive_stimulus(self, reward_signal: float, surprise: Optional[float] = None) -> None:
        """React to an incoming stimulus.

        Args:
            reward_signal: value of the stimulus in [0, 1]. Drives a phasic
                dopamine burst scaled by the phenotype's ``phasic_gain`` — the
                mechanism by which low tonic tone produces oversized responses
                to immediate reward.
            surprise: unexpectedness in [0, 1], driving the phasic NE burst. If
                omitted it defaults to ``reward_signal``, treating reward and
                novelty as co-occurring.
        """
        p = self.phenotype
        if surprise is None:
            surprise = reward_signal

        # Phasic DA burst, amplified when tonic tone is low.
        self.phasic_dopamine = self._clamp(
            self.phasic_dopamine + reward_signal * p.phasic_gain
        )
        self.dopamine = self._clamp(self.tonic_dopamine + self.phasic_dopamine)

        # Phasic NE burst. Stored as its own component so the next update
        # decays it rather than discarding it.
        self.ne_phasic = self._clamp(self.ne_phasic + surprise * p.ne_phasic_gain)
        self.norepinephrine = self._clamp(self.norepinephrine + surprise * p.ne_phasic_gain)

        # A strong stimulus transiently offsets fatigue.
        if reward_signal > 0.3:
            self.fatigue = self._clamp(self.fatigue - reward_signal * 0.2)

    def reset(self, baseline_dopamine: Optional[float] = None) -> None:
        """Reset to the phenotype's baseline.

        Args:
            baseline_dopamine: override for the starting tonic dopamine level.
                Defaults to the phenotype's ``tonic_dopamine``.
        """
        p = self.phenotype
        tonic = p.tonic_dopamine if baseline_dopamine is None else baseline_dopamine

        self.arousal = 0.5
        self.focus = 0.5
        self.fatigue = 0.0
        self.tonic_dopamine = tonic
        self.phasic_dopamine = 0.0
        self.dopamine = self._clamp(tonic)
        self.acetylcholine = p.acetylcholine_baseline
        self.ne_tonic = p.ne_tonic
        self.ne_phasic = 0.0
        self.norepinephrine = p.ne_tonic
        self.lapse_active = False
        self.step = 0
        self.history = {k: [] for k in self.history}

    def _record(self) -> None:
        h = self.history
        h["arousal"].append(self.arousal)
        h["focus"].append(self.focus)
        h["fatigue"].append(self.fatigue)
        h["dopamine"].append(self.dopamine)
        h["acetylcholine"].append(self.acetylcholine)
        h["norepinephrine"].append(self.norepinephrine)
        h["tonic_dopamine"].append(self.tonic_dopamine)
        h["phasic_dopamine"].append(self.phasic_dopamine)
        h["precision"].append(self.precision)
        h["value_gate"].append(self.value_gate)
        h["lapse"].append(1.0 if self.lapse_active else 0.0)

    def to_tensor(self):
        """Return the state as a tensor.

        Order: [arousal, focus, fatigue, dopamine, acetylcholine, norepinephrine]
        """
        if torch is None:
            raise RuntimeError("torch is required for to_tensor()")
        return torch.tensor([
            self.arousal, self.focus, self.fatigue,
            self.dopamine, self.acetylcholine, self.norepinephrine,
        ])
