"""BrainState — dynamic state tracking for the neurodivergent brain simulator."""

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
    """

    arousal: float = 0.5
    focus: float = 0.5
    fatigue: float = 0.0
    dopamine: float = 0.3
    # Timestep counter
    step: int = 0
    # History for visualization
    history: dict[str, list[float]] = field(default_factory=lambda: {
        "arousal": [],
        "focus": [],
        "fatigue": [],
        "dopamine": [],
    })

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    @property
    def snr(self) -> float:
        """Effective signal-to-noise ratio derived from dopamine and fatigue."""
        return self._clamp(self.dopamine * (1.0 - self.fatigue * 0.5))

    @property
    def is_hyperfocused(self) -> bool:
        return self.focus > 0.8

    def update(self, dt: float, dopamine_decay: float, fatigue_rate: float) -> None:
        """Advance state by one timestep."""
        # Dopamine decays over time
        self.dopamine = self._clamp(self.dopamine - dopamine_decay * dt)

        # Fatigue accumulates
        self.fatigue = self._clamp(self.fatigue + fatigue_rate * dt)

        # Focus is modulated by dopamine and inversely by fatigue
        self.focus = self._clamp(self.dopamine * 1.2 - self.fatigue * 0.4)

        # Arousal is a blend of dopamine and inverse fatigue
        self.arousal = self._clamp(0.3 + self.dopamine * 0.5 - self.fatigue * 0.3)

        self.step += 1
        self._record()

    def receive_stimulus(self, reward_signal: float) -> None:
        """React to an incoming stimulus / reward. Boosts dopamine."""
        self.dopamine = self._clamp(self.dopamine + reward_signal)
        # Strong stimuli can temporarily reduce fatigue (adrenaline effect)
        if reward_signal > 0.3:
            self.fatigue = self._clamp(self.fatigue - reward_signal * 0.2)

    def reset(self, baseline_dopamine: float = 0.3) -> None:
        """Reset to initial state."""
        self.arousal = 0.5
        self.focus = 0.5
        self.fatigue = 0.0
        self.dopamine = baseline_dopamine
        self.step = 0
        self.history = {k: [] for k in self.history}

    def _record(self) -> None:
        self.history["arousal"].append(self.arousal)
        self.history["focus"].append(self.focus)
        self.history["fatigue"].append(self.fatigue)
        self.history["dopamine"].append(self.dopamine)

    def to_tensor(self):
        """Return state as a tensor [arousal, focus, fatigue, dopamine]."""
        if torch is None:
            raise RuntimeError("torch is required for to_tensor()")
        return torch.tensor([self.arousal, self.focus, self.fatigue, self.dopamine])
