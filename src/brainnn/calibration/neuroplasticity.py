"""Neuroplasticity Onboarding Controller — gradual intensity ramp for cortical adaptation.

The brain cannot instantly interpret new artificial sensory input.
If you activate a synesthesia module at 100% intensity on day one,
the cortex perceives it as meaningless noise and actively suppresses it.

This controller implements a Hebbian-inspired onboarding schedule:
    - Day 1:  10% intensity — brain registers faint, novel patterns
    - Day 3:  20% — cortical maps begin forming (orientation columns adapt)
    - Day 7:  60% — cross-modal associations stabilize
    - Day 14: 100% — full integration, user perceives new modality as "natural"

The schedule is adaptive: if the user's neural response quality drops
(measured via evoked potential consistency), the controller pauses or
reduces intensity until the cortex catches up.

Critical constraint: intensity must NEVER jump more than 15% in a single
session. Larger jumps cause cortical confusion → headaches, phantom percepts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class OnboardingSession:
    """Record of a single onboarding session."""
    day: int
    session_number: int
    target_intensity: float
    actual_intensity: float
    duration_minutes: float
    # Mean evoked response consistency [0, 1] — how reliably the cortex responds
    response_consistency: float
    # User-reported comfort [0, 10]
    comfort_score: float
    # Whether the session was completed or aborted
    completed: bool
    notes: str = ""


@dataclass
class OnboardingState:
    """Persistent state for the onboarding process."""
    current_day: int = 1
    current_intensity: float = 0.10
    total_days: int = 14
    sessions: list[OnboardingSession] = field(default_factory=list)
    paused: bool = False
    pause_reason: str = ""


class NeuroplasticityController:
    """Manages the gradual onboarding of new sensory modules.

    Usage:
        controller = NeuroplasticityController()

        # Get today's target intensity
        intensity = controller.get_target_intensity(day=3)

        # After a session, report results
        controller.report_session(day=3, session=1,
            response_consistency=0.72, comfort_score=7.5,
            duration_minutes=30, completed=True)

        # Controller adapts: if consistency drops, it slows down
        next_intensity = controller.get_target_intensity(day=4)
    """

    # Maximum allowed intensity jump per session (15% = 0.15)
    MAX_JUMP = 0.15
    # Minimum response consistency to proceed with ramp-up
    MIN_CONSISTENCY = 0.5
    # Minimum comfort score to proceed
    MIN_COMFORT = 4.0

    # Default schedule: day → target intensity (sigmoid curve)
    @staticmethod
    def default_schedule(total_days: int = 14) -> dict[int, float]:
        """Generate a sigmoid-based intensity schedule.

        Uses a logistic curve centered at day total_days/2 to model
        natural cortical adaptation rates.
        """
        schedule = {}
        for day in range(1, total_days + 1):
            # Logistic sigmoid: slow start, fast middle, plateau at end
            x = 10 * (day / total_days) - 5  # center at midpoint
            intensity = 1.0 / (1.0 + math.exp(-x))
            # Ensure day 1 starts at 10% minimum, day N ends at 100%
            intensity = max(0.10, min(1.0, intensity))
            schedule[day] = round(intensity, 3)
        schedule[total_days] = 1.0
        return schedule

    def __init__(
        self,
        total_days: int = 14,
        config_path: str | Path | None = None,
    ) -> None:
        self.state = OnboardingState(total_days=total_days)
        self.schedule = self.default_schedule(total_days)
        self._config_path = Path(config_path) if config_path else None

        if self._config_path and self._config_path.exists():
            self._load_state()

    def get_target_intensity(self, day: int | None = None) -> float:
        """Get the target intensity for a given day.

        If the controller is paused or response quality is low,
        returns the last safe intensity instead of the schedule target.
        """
        if day is None:
            day = self.state.current_day

        if self.state.paused:
            return self.state.current_intensity

        schedule_target = self.schedule.get(day, 1.0)

        # Safety: never jump more than MAX_JUMP from current
        max_allowed = self.state.current_intensity + self.MAX_JUMP
        safe_target = min(schedule_target, max_allowed)

        # Check if recent sessions show poor adaptation
        if self._should_slow_down():
            safe_target = min(safe_target, self.state.current_intensity)

        return round(safe_target, 3)

    def report_session(
        self,
        day: int,
        session: int,
        response_consistency: float,
        comfort_score: float,
        duration_minutes: float,
        completed: bool,
        notes: str = "",
    ) -> dict[str, any]:
        """Report results of an onboarding session.

        Returns a status dict with recommendations.
        """
        actual_intensity = self.state.current_intensity
        record = OnboardingSession(
            day=day, session_number=session,
            target_intensity=self.get_target_intensity(day),
            actual_intensity=actual_intensity,
            duration_minutes=duration_minutes,
            response_consistency=response_consistency,
            comfort_score=comfort_score,
            completed=completed,
            notes=notes,
        )
        self.state.sessions.append(record)

        # Decide next action
        recommendation = self._evaluate_session(record)

        # Update state
        if recommendation["action"] == "advance":
            self.state.current_day = day + 1
            self.state.current_intensity = self.get_target_intensity(day + 1)
        elif recommendation["action"] == "hold":
            pass  # keep same day and intensity
        elif recommendation["action"] == "pause":
            self.state.paused = True
            self.state.pause_reason = recommendation["reason"]
        elif recommendation["action"] == "reduce":
            self.state.current_intensity = max(
                0.10, self.state.current_intensity - 0.05
            )

        return recommendation

    def resume(self) -> None:
        """Resume onboarding after a pause."""
        self.state.paused = False
        self.state.pause_reason = ""

    def get_progress(self) -> dict:
        """Get onboarding progress summary."""
        return {
            "current_day": self.state.current_day,
            "total_days": self.state.total_days,
            "current_intensity": self.state.current_intensity,
            "target_intensity": self.get_target_intensity(),
            "progress_pct": round(self.state.current_day / self.state.total_days * 100, 1),
            "paused": self.state.paused,
            "pause_reason": self.state.pause_reason,
            "total_sessions": len(self.state.sessions),
            "schedule": self.schedule,
        }

    def _evaluate_session(self, session: OnboardingSession) -> dict:
        """Evaluate a session and decide next steps."""
        if not session.completed:
            return {
                "action": "hold",
                "reason": "Session was not completed. Retry same day.",
                "next_intensity": self.state.current_intensity,
            }

        if session.comfort_score < self.MIN_COMFORT:
            return {
                "action": "reduce",
                "reason": f"Comfort score ({session.comfort_score}) below threshold ({self.MIN_COMFORT}). "
                          f"Reducing intensity by 5%.",
                "next_intensity": max(0.10, self.state.current_intensity - 0.05),
            }

        if session.response_consistency < self.MIN_CONSISTENCY:
            return {
                "action": "pause",
                "reason": f"Response consistency ({session.response_consistency:.2f}) below "
                          f"threshold ({self.MIN_CONSISTENCY}). Cortex needs more time to adapt.",
                "next_intensity": self.state.current_intensity,
            }

        return {
            "action": "advance",
            "reason": "Session completed successfully. Advancing to next day.",
            "next_intensity": self.get_target_intensity(session.day + 1),
        }

    def _should_slow_down(self) -> bool:
        """Check if recent sessions indicate poor adaptation."""
        recent = self.state.sessions[-3:]
        if len(recent) < 2:
            return False
        avg_consistency = np.mean([s.response_consistency for s in recent])
        return avg_consistency < self.MIN_CONSISTENCY

    def _load_state(self) -> None:
        """Load state from config file."""
        with open(self._config_path) as f:
            config = json.load(f)
        np_cfg = config.get("neuroplasticity", {})
        self.state.current_day = np_cfg.get("current_day", 1)
        schedule = np_cfg.get("intensity_schedule", {})
        if schedule:
            self.schedule = {
                int(k.replace("day_", "")): v
                for k, v in schedule.items()
            }
        self.state.current_intensity = self.schedule.get(self.state.current_day, 0.10)
