"""Neural Calibration Engine — per-user electrode-to-neuron mapping.

Every brain is unique. This module maps each electrode's position to the
neural population it can most effectively stimulate, creating a personalized
"neural fingerprint" that all other modules rely on.

Pipeline:
    1. Record baseline neural activity per electrode (impedance + spontaneous firing)
    2. Deliver test pulses at varying amplitudes / frequencies
    3. Measure evoked responses (spike sorting + LFP analysis)
    4. Build a ResponseMap: electrode → (optimal_amplitude, optimal_frequency, latency, selectivity)
    5. Cluster electrodes into functional zones (sensory, motor, associative)

This calibration must run before any cross-modal mapping or closed-loop control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CorticalZone(Enum):
    """Functional brain region an electrode maps to."""
    VISUAL_V1 = "visual_v1"
    VISUAL_V4 = "visual_v4"
    AUDITORY_A1 = "auditory_a1"
    SOMATOSENSORY_S1 = "somatosensory_s1"
    MOTOR_M1 = "motor_m1"
    PREFRONTAL = "prefrontal"
    UNKNOWN = "unknown"


@dataclass
class ElectrodeProfile:
    """Calibration result for a single electrode."""
    electrode_id: int
    # Position on the array grid (row, col)
    grid_position: tuple[int, int]
    # Impedance at 1 kHz (ohms) — healthy range: 100k–1M
    impedance_ohm: float
    # Optimal stimulation amplitude (µA)
    optimal_amplitude_uA: float
    # Optimal stimulation frequency (Hz)
    optimal_frequency_hz: float
    # Response latency from stimulus to first evoked spike (ms)
    response_latency_ms: float
    # Selectivity index [0, 1] — how specific this electrode is to one neural pop
    selectivity: float
    # Which cortical zone this electrode primarily activates
    zone: CorticalZone = CorticalZone.UNKNOWN
    # Charge density safety limit (µC/cm²) — must stay below 30 for Pt electrodes
    charge_density_limit: float = 30.0


@dataclass
class CalibrationResult:
    """Full calibration output for one user."""
    user_id: str
    electrode_profiles: list[ElectrodeProfile] = field(default_factory=list)
    grid_shape: tuple[int, int] = (32, 32)
    timestamp: str = ""
    # Global quality metric [0, 1]
    overall_quality: float = 0.0

    @property
    def active_electrodes(self) -> list[ElectrodeProfile]:
        """Electrodes with acceptable impedance (< 1 MΩ) and selectivity (> 0.3)."""
        return [e for e in self.electrode_profiles
                if e.impedance_ohm < 1_000_000 and e.selectivity > 0.3]

    def zone_map(self) -> dict[CorticalZone, list[int]]:
        """Group electrode IDs by cortical zone."""
        zones: dict[CorticalZone, list[int]] = {}
        for ep in self.active_electrodes:
            zones.setdefault(ep.zone, []).append(ep.electrode_id)
        return zones


# ---------------------------------------------------------------------------
# Calibration Engine
# ---------------------------------------------------------------------------

class CalibrationEngine:
    """Runs the full calibration pipeline on raw electrode recordings.

    Usage:
        engine = CalibrationEngine(grid_shape=(32, 32))
        result = engine.calibrate(
            user_id="patient_001",
            impedance_data=impedance_array,      # (N_electrodes,)
            baseline_lfp=baseline_lfp_array,      # (N_electrodes, T_samples)
            evoked_responses=evoked_response_dict, # {amplitude: (N_electrodes, T_samples)}
        )
    """

    # Test pulse amplitudes (µA) used during calibration sweep
    SWEEP_AMPLITUDES = [5, 10, 20, 50, 100, 200, 500]
    # Test pulse frequencies (Hz)
    SWEEP_FREQUENCIES = [10, 50, 100, 200, 500, 1000]
    # Impedance threshold for electrode rejection
    MAX_IMPEDANCE = 1_500_000  # 1.5 MΩ
    # Minimum evoked response SNR to consider electrode responsive
    MIN_RESPONSE_SNR = 3.0

    def __init__(self, grid_shape: tuple[int, int] = (32, 32)) -> None:
        self.grid_shape = grid_shape
        self.n_electrodes = grid_shape[0] * grid_shape[1]

    def calibrate(
        self,
        user_id: str,
        impedance_data: np.ndarray,
        baseline_lfp: np.ndarray,
        evoked_responses: dict[float, np.ndarray],
    ) -> CalibrationResult:
        """Run full calibration.

        Args:
            user_id: Unique patient / user identifier.
            impedance_data: (N_electrodes,) impedance in ohms at 1 kHz.
            baseline_lfp: (N_electrodes, T) spontaneous LFP in µV.
            evoked_responses: {amplitude_uA: (N_electrodes, T)} evoked response per test amp.

        Returns:
            CalibrationResult with per-electrode profiles.
        """
        profiles = []
        for idx in range(self.n_electrodes):
            row, col = divmod(idx, self.grid_shape[1])
            impedance = float(impedance_data[idx])

            # Skip dead electrodes
            if impedance > self.MAX_IMPEDANCE:
                profiles.append(ElectrodeProfile(
                    electrode_id=idx, grid_position=(row, col),
                    impedance_ohm=impedance,
                    optimal_amplitude_uA=0, optimal_frequency_hz=0,
                    response_latency_ms=0, selectivity=0,
                    zone=CorticalZone.UNKNOWN,
                ))
                continue

            # Find optimal amplitude — lowest amplitude with SNR > threshold
            baseline_std = float(np.std(baseline_lfp[idx]))
            baseline_std = max(baseline_std, 1e-6)  # avoid division by zero

            best_amp = 0.0
            best_snr = 0.0
            best_latency = 0.0
            for amp in sorted(evoked_responses.keys()):
                response = evoked_responses[amp][idx]
                peak_val = float(np.max(np.abs(response)))
                snr = peak_val / baseline_std
                if snr > best_snr:
                    best_snr = snr
                    best_amp = amp
                    # Latency = time to first significant deflection
                    threshold = baseline_std * self.MIN_RESPONSE_SNR
                    above = np.where(np.abs(response) > threshold)[0]
                    best_latency = float(above[0] / 30.0) if len(above) > 0 else 0.0

            # Selectivity: ratio of peak response to mean response across neighbours
            selectivity = self._compute_selectivity(idx, evoked_responses, best_amp)

            # Zone classification from grid position (simplified spatial mapping)
            zone = self._classify_zone(row, col)

            profiles.append(ElectrodeProfile(
                electrode_id=idx,
                grid_position=(row, col),
                impedance_ohm=impedance,
                optimal_amplitude_uA=best_amp,
                optimal_frequency_hz=200.0,  # default; refined in frequency sweep
                response_latency_ms=best_latency,
                selectivity=selectivity,
                zone=zone,
            ))

        result = CalibrationResult(
            user_id=user_id,
            electrode_profiles=profiles,
            grid_shape=self.grid_shape,
        )
        result.overall_quality = self._compute_quality(result)
        return result

    def _compute_selectivity(
        self, idx: int, evoked: dict[float, np.ndarray], amp: float,
    ) -> float:
        """Selectivity = this electrode's peak / mean of 8-neighbour peaks."""
        if amp == 0 or amp not in evoked:
            return 0.0
        data = evoked[amp]
        this_peak = float(np.max(np.abs(data[idx])))

        row, col = divmod(idx, self.grid_shape[1])
        neighbour_peaks = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid_shape[0] and 0 <= nc < self.grid_shape[1]:
                    nidx = nr * self.grid_shape[1] + nc
                    neighbour_peaks.append(float(np.max(np.abs(data[nidx]))))

        if not neighbour_peaks or max(neighbour_peaks) == 0:
            return 1.0
        mean_neighbour = float(np.mean(neighbour_peaks))
        if mean_neighbour == 0:
            return 1.0
        return min(1.0, this_peak / (this_peak + mean_neighbour))

    def _classify_zone(self, row: int, col: int) -> CorticalZone:
        """Simplified zone classification from electrode grid position.

        In a real system this would use MRI co-registration data.
        Here we use a spatial partition of the 32×32 grid.
        """
        r_norm = row / self.grid_shape[0]
        c_norm = col / self.grid_shape[1]

        if r_norm < 0.3 and c_norm < 0.5:
            return CorticalZone.VISUAL_V1
        elif r_norm < 0.3 and c_norm >= 0.5:
            return CorticalZone.VISUAL_V4
        elif 0.3 <= r_norm < 0.5:
            return CorticalZone.AUDITORY_A1
        elif 0.5 <= r_norm < 0.7 and c_norm < 0.5:
            return CorticalZone.SOMATOSENSORY_S1
        elif 0.5 <= r_norm < 0.7 and c_norm >= 0.5:
            return CorticalZone.MOTOR_M1
        else:
            return CorticalZone.PREFRONTAL

    def _compute_quality(self, result: CalibrationResult) -> float:
        """Global calibration quality: fraction of electrodes that are active & selective."""
        if not result.electrode_profiles:
            return 0.0
        return len(result.active_electrodes) / len(result.electrode_profiles)


# ---------------------------------------------------------------------------
# Neural Response Model (for simulation / sandbox testing)
# ---------------------------------------------------------------------------

class NeuralResponseModel(nn.Module):
    """Learned model that predicts evoked neural response from stimulus parameters.

    This allows the simulation sandbox to predict how a calibrated brain
    will respond to a given stimulus without needing real hardware.

    Input:  (batch, 4) → [amplitude, frequency, pulse_width, electrode_impedance]
    Output: (batch, T) → predicted evoked response waveform
    """

    def __init__(self, response_length: int = 100, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, response_length),
            nn.Tanh(),  # normalized response in [-1, 1]
        )

    def forward(self, stimulus_params: torch.Tensor) -> torch.Tensor:
        """Predict evoked response waveform from stimulus parameters."""
        return self.net(stimulus_params)
