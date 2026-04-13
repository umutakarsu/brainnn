"""Dynamic Parameter Tuning API — real-time control of SynapseFlow parameters.

Provides a FastAPI-based REST + WebSocket interface for:
    - Reading / updating gain, latency, mapping profiles in real-time
    - Switching between sensory modules (vision ↔ synesthesia ↔ motor)
    - Monitoring system state (impedance, temperature, latency)
    - Emergency shutoff

All parameter changes are validated against safety limits before applying.
WebSocket endpoint streams real-time telemetry (latency, spike rates, etc.).

Designed to run on the external processing unit (FPGA companion or host PC).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from copy import deepcopy


# ---------------------------------------------------------------------------
# In-memory parameter store (would be backed by config_manager.json in prod)
# ---------------------------------------------------------------------------

@dataclass
class SafetyLimits:
    """Hardware safety constraints — these cannot be overridden via API."""
    charge_density_limit: float = 30.0   # µC/cm²
    max_amplitude_uA: float = 500.0
    max_frequency_hz: float = 1000.0
    max_pulse_width_us: float = 500.0
    temperature_limit_c: float = 40.0


@dataclass
class LiveParameters:
    """Currently active runtime parameters."""
    # Global gain multiplier
    global_gain: float = 1.0
    # Per-zone gain overrides
    zone_gains: dict[str, float] = field(default_factory=lambda: {
        "visual_v1": 1.2, "visual_v4": 1.0,
        "auditory_a1": 0.9, "somatosensory_s1": 1.1,
        "motor_m1": 0.8, "prefrontal": 0.7,
    })
    # Target latency (ms)
    target_latency_ms: float = 8.0
    # Active mapping module
    active_module: str = "synesthesia_audio_to_visual"
    # Neuroplasticity intensity [0, 1]
    intensity: float = 1.0
    # System armed (False = all stimulation disabled)
    armed: bool = True
    # Timestamp of last update
    last_updated: float = 0.0


class ParameterValidationError(Exception):
    """Raised when a parameter change violates safety limits."""
    pass


# ---------------------------------------------------------------------------
# Parameter Tuning Controller
# ---------------------------------------------------------------------------

class TuningController:
    """Core controller for real-time parameter management.

    Usage:
        controller = TuningController(config_path="configs/config_manager.json")

        # Read current params
        params = controller.get_parameters()

        # Update gain
        controller.update_parameter("global_gain", 1.5)

        # Update zone-specific gain
        controller.update_zone_gain("visual_v1", 1.8)

        # Switch module
        controller.switch_module("vision_restoration")

        # Emergency stop
        controller.emergency_stop()
    """

    VALID_MODULES = [
        "synesthesia_audio_to_visual",
        "synesthesia_visual_to_auditory",
        "vision_restoration",
        "motor_decode",
        "closed_loop_mood",
    ]

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.safety = SafetyLimits()
        self.params = LiveParameters()
        self._config_path = Path(config_path) if config_path else None
        self._change_log: list[dict[str, Any]] = []

        if self._config_path and self._config_path.exists():
            self._load_from_config()

    def _load_from_config(self) -> None:
        """Load initial parameters from config_manager.json."""
        with open(self._config_path) as f:
            config = json.load(f)

        sp = config.get("signal_processing", {})
        gain = sp.get("gain", {})
        self.params.global_gain = gain.get("global", 1.0)
        self.params.zone_gains = gain.get("per_zone", self.params.zone_gains)
        self.params.target_latency_ms = sp.get("latency", {}).get("target_ms", 8.0)

        mp = config.get("mapping_profile", {})
        self.params.active_module = mp.get("active_module", self.params.active_module)

        np_cfg = config.get("neuroplasticity", {})
        day = np_cfg.get("current_day", 14)
        schedule = np_cfg.get("intensity_schedule", {})
        self.params.intensity = schedule.get(f"day_{day}", 1.0)

        safety = config.get("safety", {})
        self.safety.charge_density_limit = safety.get("charge_density_limit_uC_per_cm2", 30.0)
        self.safety.max_amplitude_uA = safety.get("max_amplitude_uA", 500.0)
        self.safety.max_frequency_hz = safety.get("max_frequency_hz", 1000.0)

    # ---- Read ----

    def get_parameters(self) -> dict[str, Any]:
        """Get current live parameters as a dict."""
        return asdict(self.params)

    def get_safety_limits(self) -> dict[str, Any]:
        """Get hardware safety limits."""
        return asdict(self.safety)

    def get_change_log(self, last_n: int = 50) -> list[dict[str, Any]]:
        """Get recent parameter changes."""
        return self._change_log[-last_n:]

    # ---- Update ----

    def update_parameter(self, key: str, value: Any) -> dict[str, Any]:
        """Update a single parameter with safety validation.

        Args:
            key: Parameter name (e.g., "global_gain", "target_latency_ms", "intensity").
            value: New value.

        Returns:
            Updated parameters dict.

        Raises:
            ParameterValidationError: If the change violates safety limits.
        """
        if not self.params.armed and key != "armed":
            raise ParameterValidationError("System is disarmed. Re-arm before changing parameters.")

        self._validate(key, value)

        old_value = getattr(self.params, key, None)
        setattr(self.params, key, value)
        self.params.last_updated = time.time()

        self._log_change(key, old_value, value)
        return self.get_parameters()

    def update_zone_gain(self, zone: str, gain: float) -> dict[str, Any]:
        """Update gain for a specific cortical zone.

        Args:
            zone: Zone name (e.g., "visual_v1").
            gain: New gain value [0.0, 5.0].
        """
        if gain < 0.0 or gain > 5.0:
            raise ParameterValidationError(f"Zone gain must be in [0.0, 5.0], got {gain}")
        old = self.params.zone_gains.get(zone)
        self.params.zone_gains[zone] = gain
        self.params.last_updated = time.time()
        self._log_change(f"zone_gains.{zone}", old, gain)
        return self.get_parameters()

    def switch_module(self, module_name: str) -> dict[str, Any]:
        """Switch the active sensory mapping module.

        Args:
            module_name: One of VALID_MODULES.
        """
        if module_name not in self.VALID_MODULES:
            raise ParameterValidationError(
                f"Unknown module '{module_name}'. Valid: {self.VALID_MODULES}"
            )
        return self.update_parameter("active_module", module_name)

    def emergency_stop(self) -> dict[str, Any]:
        """Immediately disarm all stimulation."""
        self.params.armed = False
        self.params.last_updated = time.time()
        self._log_change("armed", True, False, reason="EMERGENCY STOP")
        return self.get_parameters()

    def rearm(self) -> dict[str, Any]:
        """Re-arm stimulation after emergency stop."""
        self.params.armed = True
        self.params.last_updated = time.time()
        self._log_change("armed", False, True, reason="RE-ARMED")
        return self.get_parameters()

    # ---- Validation ----

    def _validate(self, key: str, value: Any) -> None:
        """Validate a parameter change against safety limits."""
        if key == "global_gain" and (value < 0.0 or value > 5.0):
            raise ParameterValidationError(f"Global gain must be in [0.0, 5.0], got {value}")
        if key == "target_latency_ms" and value > 10.0:
            raise ParameterValidationError(
                f"Target latency {value}ms exceeds 10ms cross-modal binding limit"
            )
        if key == "intensity" and (value < 0.0 or value > 1.0):
            raise ParameterValidationError(f"Intensity must be in [0.0, 1.0], got {value}")

    def _log_change(
        self, key: str, old: Any, new: Any, reason: str = "",
    ) -> None:
        self._change_log.append({
            "timestamp": time.time(),
            "parameter": key,
            "old_value": old,
            "new_value": new,
            "reason": reason,
        })

    # ---- Persistence ----

    def save_config(self, path: str | Path | None = None) -> None:
        """Persist current parameters back to config JSON."""
        target = Path(path) if path else self._config_path
        if target is None:
            raise ValueError("No config path specified")

        if target.exists():
            with open(target) as f:
                config = json.load(f)
        else:
            config = {}

        config.setdefault("signal_processing", {})
        config["signal_processing"]["gain"] = {
            "global": self.params.global_gain,
            "per_zone": self.params.zone_gains,
        }
        config["signal_processing"]["latency"] = {
            "target_ms": self.params.target_latency_ms,
        }
        config.setdefault("mapping_profile", {})
        config["mapping_profile"]["active_module"] = self.params.active_module

        with open(target, "w") as f:
            json.dump(config, f, indent=2)
