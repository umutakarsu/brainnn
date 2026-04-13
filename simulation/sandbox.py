"""Neural Simulation Sandbox — test algorithms without hardware.

End-to-end simulation pipeline that connects all SynapseFlow components:
    1. CalibrationEngine generates synthetic electrode profiles
    2. SynesthesiaEngine transforms input signals
    3. SpikeTrainEncoder produces stimulation patterns
    4. A simulated cortical response model closes the loop
    5. NeuroplasticityController manages intensity over time

This lets researchers iterate on algorithms, tune parameters, and validate
safety constraints before any implant touches a real brain.

Can integrate with:
    - PyTorch (this file) for GPU-accelerated neural models
    - NEST simulator for biophysically detailed spiking networks
    - The existing brainnn NeuroDivergentBrainSimulator for full brain sim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from brainnn.calibration.calibration_logic import (
    CalibrationEngine, CalibrationResult, CorticalZone,
)
from brainnn.synesthesia.synesthesia_engine import (
    SynesthesiaEngine, SynesthesiaConfig, SpikeTrainEncoder,
)
from brainnn.calibration.neuroplasticity import NeuroplasticityController


# ---------------------------------------------------------------------------
# Synthetic data generators (for testing without hardware)
# ---------------------------------------------------------------------------

def generate_synthetic_impedance(
    n_electrodes: int = 1024,
    mean_ohm: float = 500_000,
    std_ohm: float = 200_000,
    dead_fraction: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate realistic synthetic impedance data.

    Args:
        n_electrodes: Number of electrodes on the array.
        mean_ohm: Mean impedance (healthy Pt electrode ≈ 200k–800k Ω).
        std_ohm: Standard deviation.
        dead_fraction: Fraction of electrodes with impedance > 2 MΩ (dead).
        rng: Random number generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    impedance = rng.normal(mean_ohm, std_ohm, n_electrodes)
    impedance = np.clip(impedance, 50_000, 1_200_000)

    # Mark some as dead
    n_dead = int(n_electrodes * dead_fraction)
    dead_indices = rng.choice(n_electrodes, n_dead, replace=False)
    impedance[dead_indices] = rng.uniform(2_000_000, 5_000_000, n_dead)

    return impedance


def generate_synthetic_lfp(
    n_electrodes: int = 1024,
    n_samples: int = 30000,
    base_amplitude_uV: float = 50.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic Local Field Potential recordings.

    Produces band-limited noise resembling real cortical LFP
    (dominant power in 1–100 Hz range).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Pink noise approximation: 1/f spectrum
    freqs = np.fft.rfftfreq(n_samples, d=1/30000)
    freqs[0] = 1  # avoid division by zero

    lfp = np.zeros((n_electrodes, n_samples))
    for i in range(n_electrodes):
        # Random phase, 1/f amplitude
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        amplitudes = base_amplitude_uV / np.sqrt(freqs)
        spectrum = amplitudes * np.exp(1j * phases)
        lfp[i] = np.fft.irfft(spectrum, n=n_samples)

    return lfp


def generate_synthetic_evoked(
    n_electrodes: int = 1024,
    n_samples: int = 3000,
    amplitudes_uA: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[float, np.ndarray]:
    """Generate synthetic evoked responses for calibration sweep.

    Higher stimulus amplitude → larger, earlier evoked response.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if amplitudes_uA is None:
        amplitudes_uA = [5, 10, 20, 50, 100, 200, 500]

    t = np.arange(n_samples) / 30000  # 30 kHz sample rate
    responses = {}

    for amp in amplitudes_uA:
        evoked = np.zeros((n_electrodes, n_samples))
        for i in range(n_electrodes):
            # Evoked response: damped sinusoid starting after latency
            latency_s = max(0.001, 0.02 - amp * 0.00003)  # higher amp → shorter latency
            peak_uV = amp * 0.5 * (1 + 0.3 * rng.normal())  # proportional to stimulus
            mask = t > latency_s
            t_shifted = t - latency_s
            evoked[i, mask] = (
                peak_uV
                * np.exp(-t_shifted[mask] / 0.01)
                * np.sin(2 * np.pi * 200 * t_shifted[mask])
            )
            # Add noise
            evoked[i] += rng.normal(0, 10, n_samples)

        responses[amp] = evoked

    return responses


# ---------------------------------------------------------------------------
# Sandbox Runner
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Output from a sandbox simulation run."""
    calibration: CalibrationResult
    # Per-frame RGB output from synesthesia engine
    rgb_frames: np.ndarray          # (N_frames, 3)
    # Spike trains for each frame
    spike_trains: list[np.ndarray]  # list of (N_channels, N_timesteps)
    # Simulated cortical response per frame
    cortical_response: np.ndarray   # (N_frames,)
    # Neuroplasticity state
    onboarding_progress: dict
    # Timing metrics
    latency_ms: float
    total_time_ms: float


class SimulationSandbox:
    """Full end-to-end simulation without hardware.

    Usage:
        sandbox = SimulationSandbox()
        result = sandbox.run(
            audio_input=np.random.randn(44100),  # 1 second of audio
            user_id="test_patient",
            onboarding_day=5,
        )
        print(f"Generated {len(result.rgb_frames)} RGB frames")
        print(f"Calibration quality: {result.calibration.overall_quality:.2f}")
        print(f"Onboarding: {result.onboarding_progress['progress_pct']}%")
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (32, 32),
        synesthesia_config: SynesthesiaConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.grid_shape = grid_shape
        self.n_electrodes = grid_shape[0] * grid_shape[1]
        self.rng = np.random.default_rng(seed)

        # Initialize components
        self.calibration_engine = CalibrationEngine(grid_shape=grid_shape)
        # Use a small FFT window (256 samples @ 44.1kHz ≈ 5.8ms) to stay under
        # the 10ms cross-modal binding latency budget
        if synesthesia_config is None:
            synesthesia_config = SynesthesiaConfig(fft_window=256, hop_size=128)
        self.synesthesia_engine = SynesthesiaEngine(synesthesia_config)
        self.spike_encoder = SpikeTrainEncoder()
        self.plasticity = NeuroplasticityController()

    def run(
        self,
        audio_input: np.ndarray,
        user_id: str = "sandbox_user",
        onboarding_day: int = 14,
    ) -> SandboxResult:
        """Run a full end-to-end simulation.

        Args:
            audio_input: (N_samples,) mono audio, normalized to [-1, 1].
            user_id: Identifier for calibration profile.
            onboarding_day: Which day of onboarding to simulate.
        """
        import time
        t0 = time.perf_counter()

        # 1. Calibration — generate synthetic data and calibrate
        impedance = generate_synthetic_impedance(self.n_electrodes, rng=self.rng)
        baseline = generate_synthetic_lfp(self.n_electrodes, rng=self.rng)
        evoked = generate_synthetic_evoked(self.n_electrodes, rng=self.rng)

        calibration = self.calibration_engine.calibrate(
            user_id=user_id,
            impedance_data=impedance,
            baseline_lfp=baseline,
            evoked_responses=evoked,
        )

        # 2. Set neuroplasticity intensity — in sandbox mode, use schedule directly
        #    (in real use, the controller advances day-by-day with session reports)
        intensity = self.plasticity.schedule.get(onboarding_day, 1.0)
        self.plasticity.state.current_day = onboarding_day
        self.plasticity.state.current_intensity = intensity
        self.synesthesia_engine.config.intensity = intensity

        # 3. Synesthesia: audio → RGB
        rgb_frames = self.synesthesia_engine.audio_to_rgb(audio_input)

        # 4. Encode RGB as spike trains
        spike_trains = []
        for frame in rgb_frames:
            spikes = self.spike_encoder.encode(frame, duration_ms=20.0)
            spike_trains.append(spikes)

        # 5. Simulate cortical response (simplified: response = f(spike_rate, calibration_quality))
        cortical_response = np.zeros(len(rgb_frames))
        for i, spikes in enumerate(spike_trains):
            mean_rate = np.mean(spikes) * 1000 / 20.0  # spikes/sec
            cortical_response[i] = (
                mean_rate * calibration.overall_quality * intensity
            )

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        latency_per_frame = total_ms / max(len(rgb_frames), 1)

        return SandboxResult(
            calibration=calibration,
            rgb_frames=rgb_frames,
            spike_trains=spike_trains,
            cortical_response=cortical_response,
            onboarding_progress=self.plasticity.get_progress(),
            latency_ms=latency_per_frame,
            total_time_ms=total_ms,
        )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SynapseFlow — Neural Simulation Sandbox")
    print("=" * 60)

    # Generate 1 second of synthetic audio (440 Hz sine + noise)
    sr = 44100
    t = np.arange(sr) / sr
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(sr)
    audio = audio / np.max(np.abs(audio))

    sandbox = SimulationSandbox()

    # Simulate at different onboarding stages
    for day in [1, 7, 14]:
        result = sandbox.run(audio, onboarding_day=day)
        print(f"\n--- Day {day:2d} ---")
        print(f"  Intensity:           {result.onboarding_progress['current_intensity']:.0%}")
        print(f"  Calibration quality: {result.calibration.overall_quality:.2%}")
        print(f"  Active electrodes:   {len(result.calibration.active_electrodes)}/{sandbox.n_electrodes}")
        print(f"  RGB frames:          {len(result.rgb_frames)}")
        print(f"  Mean cortical resp:  {np.mean(result.cortical_response):.2f}")
        print(f"  Latency/frame:       {result.latency_ms:.2f} ms")
        print(f"  Total time:          {result.total_time_ms:.0f} ms")
