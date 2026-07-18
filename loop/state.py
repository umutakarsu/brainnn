"""BrainState — neuromodulator state container.

Copied and extended from the research engine (brainnn/src/brainnn/core/state.py)
and made standalone: the torch dependency is removed (this product is a
pharmacokinetic curve model, not a neural network — see simulate.py).

Architectural note (inherited from the Dayan-correction refactor):
    The neuromodulators are kept separated rather than conflated into a single
    "dopamine" signal, following Yu & Dayan (2005):

    - Acetylcholine (ACh): expected uncertainty / precision sharpening.
    - Norepinephrine (NE): unexpected uncertainty + arousal (inverted-U).
    - Dopamine (DA): value-of-attention gating, NOT precision.

Extensions for Loop (this product):
    - cortisol:       post-episode stress / "shame" axis. Elevated cortisol
                      lowers prefrontal control and raises relapse probability —
                      this is the mechanistic reason the copy never adds shame.
    - opioid:         β-endorphin "relief/liking" signal at consumption, kept
                      separate from dopamine "wanting" (Berridge wanting≠liking).
    - baseline_drift: allostatic downward shift of the dopamine set-point with
                      repeated episodes (Koob & Le Moal 2001). Negative = eroded.

Note: `update()` is retained from the engine for completeness, but the Loop
simulation does NOT use its discrete timestep loop — it is too coarse. See
simulate.py for the continuous opponent-process model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BrainState:
    """Tracks the dynamic internal state of the simulated brain.

    Core neuromodulators are in [0, 1]. `baseline_drift` is a signed offset
    (0 = intact baseline, negative = allostatic erosion).
    """

    arousal: float = 0.5
    focus: float = 0.5
    fatigue: float = 0.0
    dopamine: float = 0.3

    # Separated neuromodulators (Yu & Dayan 2005 framework)
    acetylcholine: float = 0.5   # expected uncertainty / precision sharpening
    norepinephrine: float = 0.5  # unexpected uncertainty / arousal-linked

    # Loop extensions
    cortisol: float = 0.2        # stress axis; elevated → lower prefrontal control
    opioid: float = 0.0          # β-endorphin "relief/liking" at consumption
    baseline_drift: float = 0.0  # allostatic set-point shift (<=0 with tolerance)

    step: int = 0
    history: dict[str, list[float]] = field(default_factory=lambda: {
        "arousal": [],
        "focus": [],
        "fatigue": [],
        "dopamine": [],
        "acetylcholine": [],
        "norepinephrine": [],
        "cortisol": [],
        "opioid": [],
        "baseline_drift": [],
    })

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    @property
    def precision(self) -> float:
        """Effective precision (signal sharpness) from ACh + NE (Yu & Dayan 2005).

        Dopamine is intentionally absent: DA governs *which* targets to attend to,
        not *how precisely*. NE has an inverted-U (Yerkes-Dodson) effect.
        """
        ne_stability = max(0.0, 1.0 - abs(self.norepinephrine - 0.5) * 2.0)
        precision_raw = self.acetylcholine * 0.7 + ne_stability * 0.3
        return self._clamp(precision_raw * (1.0 - self.fatigue * 0.5))

    @property
    def snr(self) -> float:
        """Backward-compatible alias for `precision`."""
        return self.precision

    @property
    def value_gate(self) -> float:
        """Dopaminergic value-of-attention gate (cost-benefit selection strength)."""
        return self._clamp(self.dopamine * (1.0 - self.fatigue * 0.3))

    @property
    def effective_baseline(self) -> float:
        """Dopamine baseline after allostatic drift (Koob & Le Moal 2001)."""
        return self._clamp(self.dopamine + self.baseline_drift)

    @property
    def is_hyperfocused(self) -> bool:
        return self.focus > 0.8

    def update(self, dt: float, dopamine_decay: float, fatigue_rate: float) -> None:
        """Advance state by one discrete timestep.

        Retained from the research engine. The Loop simulation does not call this
        (see simulate.py for the continuous model) — it is kept so the state
        container stays faithful to the copied engine and remains reusable.
        """
        self.dopamine = self._clamp(self.dopamine - dopamine_decay * dt)
        self.fatigue = self._clamp(self.fatigue + fatigue_rate * dt)

        ach_decay = fatigue_rate * 0.5
        self.acetylcholine = self._clamp(
            self.acetylcholine + (0.5 - self.acetylcholine) * ach_decay * dt
            - self.fatigue * 0.05 * dt
        )
        self.norepinephrine = self._clamp(0.4 + self.arousal * 0.4 - self.fatigue * 0.3)
        self.focus = self._clamp(
            self.precision * 0.6 + self.value_gate * 0.6 - self.fatigue * 0.4
        )
        self.arousal = self._clamp(0.3 + self.dopamine * 0.5 - self.fatigue * 0.3)

        self.step += 1
        self._record()

    def reset(self, baseline_dopamine: float = 0.3) -> None:
        self.arousal = 0.5
        self.focus = 0.5
        self.fatigue = 0.0
        self.dopamine = baseline_dopamine
        self.acetylcholine = 0.5
        self.norepinephrine = 0.5
        self.cortisol = 0.2
        self.opioid = 0.0
        self.baseline_drift = 0.0
        self.step = 0
        self.history = {k: [] for k in self.history}

    def _record(self) -> None:
        self.history["arousal"].append(self.arousal)
        self.history["focus"].append(self.focus)
        self.history["fatigue"].append(self.fatigue)
        self.history["dopamine"].append(self.dopamine)
        self.history["acetylcholine"].append(self.acetylcholine)
        self.history["norepinephrine"].append(self.norepinephrine)
        self.history["cortisol"].append(self.cortisol)
        self.history["opioid"].append(self.opioid)
        self.history["baseline_drift"].append(self.baseline_drift)
