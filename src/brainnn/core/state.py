"""BrainState — dynamic state tracking for the neurodivergent brain simulator.

Architectural note (v0.2, Dayan-correction):
    Earlier versions conflated several attentional functions into the dopamine signal.
    Following Yu & Dayan (2005) and subsequent work, this module now separates:

    - Acetylcholine (ACh): expected uncertainty / precision sharpening.
      Drives the noise floor in attention — high ACh means the system "knows it
      knows," so precision is high and attention scores are crisp.

    - Norepinephrine (NE): unexpected uncertainty + arousal.
      Drives volatility — high NE means the system is alert to surprise; both
      very low and very high NE degrade precision (yerkes-dodson-like).

    - Dopamine (DA): value-of-attention gating ("is this head worth attending?").
      Does NOT control precision/SNR directly. Instead it modulates which heads
      get amplified vs. suppressed, based on expected value of attending.

    The `snr` / `precision` properties are now derived from ACh and NE, not dopamine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
try:
    import torch
except ImportError:
    torch = None


@dataclass
class BrainState:
    """Tracks the dynamic internal state of the simulated brain.

    All values are in [0, 1] range.

    Neuromodulator semantics:
        acetylcholine  → expected uncertainty / precision (sharpness of attention)
        norepinephrine → unexpected uncertainty / arousal (volatility, surprise gain)
        dopamine       → value-of-attention gate (which heads to amplify)
    """

    arousal: float = 0.5
    focus: float = 0.5
    fatigue: float = 0.0
    dopamine: float = 0.3

    # Newly separated neuromodulators (Yu & Dayan 2005 framework)
    acetylcholine: float = 0.5  # expected uncertainty / precision sharpening
    norepinephrine: float = 0.5  # unexpected uncertainty / arousal-linked

    # Timestep counter
    step: int = 0
    # History for visualization
    history: dict[str, list[float]] = field(default_factory=lambda: {
        "arousal": [],
        "focus": [],
        "fatigue": [],
        "dopamine": [],
        "acetylcholine": [],
        "norepinephrine": [],
    })

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    @property
    def precision(self) -> float:
        """Effective precision (signal sharpness) — from cholinergic + noradrenergic systems.

        Per Yu & Dayan (2005):
            - High acetylcholine → high expected-uncertainty awareness → high precision
            - Norepinephrine has an inverted-U effect (Yerkes-Dodson): mid-range NE
              optimal; very low (unaware) or very high (jittery) both degrade precision.
            - Fatigue erodes the system's ability to maintain precision.

        Note: dopamine is NOT in this expression. Dopamine governs which heads to
        attend to (value-gating), not how precisely you attend.
        """
        # NE peaks at 0.5 (alert but not jittery); both extremes hurt precision
        ne_stability = 1.0 - abs(self.norepinephrine - 0.5) * 2.0
        ne_stability = max(0.0, ne_stability)
        precision_raw = self.acetylcholine * 0.7 + ne_stability * 0.3
        # Fatigue caps precision
        return self._clamp(precision_raw * (1.0 - self.fatigue * 0.5))

    @property
    def snr(self) -> float:
        """Backward-compatible alias for `precision`.

        Older code (and ADHDAttention.noise_injector) calls this `snr`. The
        underlying quantity is precision, driven by ACh + NE — not dopamine.
        """
        return self.precision

    @property
    def value_gate(self) -> float:
        """Dopaminergic value-of-attention gate.

        High dopamine → strong selection (lock onto high-value heads, suppress others).
        Low dopamine → weak gating (uniform / scattered attention).

        This replaces the earlier (incorrect) use of dopamine as a precision/SNR
        controller. It is the cost-benefit "is this worth attending to?" signal.
        """
        return self._clamp(self.dopamine * (1.0 - self.fatigue * 0.3))

    @property
    def is_hyperfocused(self) -> bool:
        return self.focus > 0.8

    def update(self, dt: float, dopamine_decay: float, fatigue_rate: float) -> None:
        """Advance state by one timestep."""
        # Dopamine decays over time
        self.dopamine = self._clamp(self.dopamine - dopamine_decay * dt)

        # Fatigue accumulates
        self.fatigue = self._clamp(self.fatigue + fatigue_rate * dt)

        # ACh tracks focus demand — slowly drifts toward 0.5 baseline as fatigue grows
        ach_decay = fatigue_rate * 0.5
        self.acetylcholine = self._clamp(
            self.acetylcholine + (0.5 - self.acetylcholine) * ach_decay * dt
            - self.fatigue * 0.05 * dt
        )

        # NE follows arousal and fatigue: tired → blunted NE
        self.norepinephrine = self._clamp(
            0.4 + self.arousal * 0.4 - self.fatigue * 0.3
        )

        # Focus is modulated by precision (ACh/NE) and value-gate (DA), not just DA
        self.focus = self._clamp(self.precision * 0.6 + self.value_gate * 0.6 - self.fatigue * 0.4)

        # Arousal is a blend of dopamine and inverse fatigue
        self.arousal = self._clamp(0.3 + self.dopamine * 0.5 - self.fatigue * 0.3)

        self.step += 1
        self._record()

    def receive_stimulus(self, reward_signal: float) -> None:
        """React to an incoming stimulus / reward.

        - Reward signals bump dopamine (value-of-attention spike)
        - Surprise component bumps norepinephrine (unexpected uncertainty)
        - Strong stimuli also temporarily reduce fatigue
        """
        self.dopamine = self._clamp(self.dopamine + reward_signal)
        # Reward also signals novelty/surprise → NE bump
        self.norepinephrine = self._clamp(self.norepinephrine + reward_signal * 0.4)
        # Strong stimuli can temporarily reduce fatigue (adrenaline effect)
        if reward_signal > 0.3:
            self.fatigue = self._clamp(self.fatigue - reward_signal * 0.2)

    def reset(self, baseline_dopamine: float = 0.3) -> None:
        """Reset to initial state."""
        self.arousal = 0.5
        self.focus = 0.5
        self.fatigue = 0.0
        self.dopamine = baseline_dopamine
        self.acetylcholine = 0.5
        self.norepinephrine = 0.5
        self.step = 0
        self.history = {k: [] for k in self.history}

    def _record(self) -> None:
        self.history["arousal"].append(self.arousal)
        self.history["focus"].append(self.focus)
        self.history["fatigue"].append(self.fatigue)
        self.history["dopamine"].append(self.dopamine)
        self.history["acetylcholine"].append(self.acetylcholine)
        self.history["norepinephrine"].append(self.norepinephrine)

    def to_tensor(self):
        """Return state as a tensor [arousal, focus, fatigue, dopamine, ACh, NE]."""
        if torch is None:
            raise RuntimeError("torch is required for to_tensor()")
        return torch.tensor([
            self.arousal, self.focus, self.fatigue,
            self.dopamine, self.acetylcholine, self.norepinephrine,
        ])
