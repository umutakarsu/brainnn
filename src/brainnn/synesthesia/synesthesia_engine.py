"""Synesthesia Engine — parametric cross-modal signal transformer.

Converts input from one sensory channel (e.g., microphone audio) into
electrical spike-train patterns that a target brain region can interpret
as a different modality (e.g., visual cortex perceives colors).

Primary use case: Audio → Visual Cortex RGB mapping ("chromesthesia")
    1. Audio → FFT → frequency bands
    2. Frequency bands → parametric mapping → RGB triplets
    3. RGB triplets → spike-train encoder → electrode stimulation pattern

The mapping is fully parametric so the same engine can be reused for:
    - Audio → Tactile (vibrotactile frequency coding)
    - Visual → Auditory (sonification for hearing restoration)
    - Any N-to-M sensory remapping

All parameters are exposed through the config system for real-time tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from brainnn.core.config import ModalityType


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SynesthesiaMapping:
    """Defines how one frequency band maps to an output channel."""
    # Frequency band edges (Hz)
    freq_low: float
    freq_high: float
    # Output channel index (e.g., 0=R, 1=G, 2=B for visual)
    output_channel: int
    # Gain multiplier for this band [0, ∞)
    gain: float = 1.0
    # Phase offset in radians (for temporal alignment)
    phase_offset: float = 0.0
    # Non-linearity: "linear", "log", "sigmoid", "exponential"
    transfer_function: str = "log"


@dataclass
class SynesthesiaConfig:
    """Full configuration for the synesthesia engine."""
    # Source and target modalities
    source_modality: ModalityType = ModalityType.AUDITORY
    target_modality: ModalityType = ModalityType.VISUAL
    # Sample rate of input signal (Hz)
    sample_rate: int = 44100
    # FFT window size (samples)
    fft_window: int = 2048
    # Hop size for STFT (samples)
    hop_size: int = 512
    # Number of output channels (3 for RGB, 1 for mono tactile, etc.)
    n_output_channels: int = 3
    # Per-band mappings
    mappings: list[SynesthesiaMapping] = field(default_factory=lambda: [
        # Low frequencies → Red (bass = warm colors)
        SynesthesiaMapping(freq_low=20, freq_high=250, output_channel=0,
                           gain=1.2, transfer_function="log"),
        # Mid frequencies → Green (vocals/melody = balanced)
        SynesthesiaMapping(freq_low=250, freq_high=4000, output_channel=1,
                           gain=1.0, transfer_function="log"),
        # High frequencies → Blue (treble/harmonics = cool colors)
        SynesthesiaMapping(freq_low=4000, freq_high=20000, output_channel=2,
                           gain=0.8, transfer_function="log"),
    ])
    # Global output gain
    master_gain: float = 1.0
    # Maximum latency budget (ms) — must stay under 10ms for perceptual binding
    max_latency_ms: float = 10.0
    # Neuroplasticity onboarding intensity [0, 1]
    intensity: float = 1.0


# ---------------------------------------------------------------------------
# Transfer functions
# ---------------------------------------------------------------------------

def _apply_transfer(values: np.ndarray, func_name: str) -> np.ndarray:
    """Apply a named non-linear transfer function."""
    if func_name == "linear":
        return values
    elif func_name == "log":
        return np.log1p(values)
    elif func_name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-values))
    elif func_name == "exponential":
        return np.expm1(np.clip(values, 0, 5))
    else:
        return values


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class SynesthesiaEngine:
    """Transforms audio frequency data into visual-cortex RGB stimulation values.

    This is the parametric "translator" between sensory domains.

    Usage:
        engine = SynesthesiaEngine(config)
        rgb_frames = engine.audio_to_rgb(audio_waveform)

        # Or for real-time streaming:
        for chunk in audio_stream:
            rgb = engine.process_chunk(chunk)
            stimulator.send(rgb)
    """

    def __init__(self, config: SynesthesiaConfig | None = None) -> None:
        self.config = config or SynesthesiaConfig()
        self._validate_latency()
        # Pre-compute frequency bin edges for each mapping
        self._freq_bins = np.fft.rfftfreq(self.config.fft_window, 1.0 / self.config.sample_rate)

    def _validate_latency(self) -> None:
        """Verify that processing latency fits within the budget."""
        # FFT latency ≈ window_size / sample_rate * 1000 (ms)
        fft_latency_ms = (self.config.fft_window / self.config.sample_rate) * 1000
        if fft_latency_ms > self.config.max_latency_ms:
            raise ValueError(
                f"FFT window latency ({fft_latency_ms:.1f}ms) exceeds budget "
                f"({self.config.max_latency_ms}ms). Reduce fft_window or increase budget."
            )

    def audio_to_rgb(self, audio: np.ndarray) -> np.ndarray:
        """Convert a full audio waveform to RGB frame sequence.

        Args:
            audio: (N_samples,) mono audio waveform, normalized to [-1, 1].

        Returns:
            (N_frames, 3) array of RGB values in [0, 1].
        """
        # Compute STFT magnitude
        spectrogram = self._stft_magnitude(audio)  # (N_frames, N_freq_bins)
        n_frames = spectrogram.shape[0]

        # Initialize output
        rgb = np.zeros((n_frames, self.config.n_output_channels))

        # Apply each frequency-band mapping
        for mapping in self.config.mappings:
            band_energy = self._extract_band_energy(spectrogram, mapping)
            transformed = _apply_transfer(band_energy, mapping.transfer_function)
            rgb[:, mapping.output_channel] += transformed * mapping.gain

        # Apply master gain and intensity (neuroplasticity onboarding)
        rgb *= self.config.master_gain * self.config.intensity

        # Normalize to [0, 1]
        max_val = np.max(rgb)
        if max_val > 0:
            rgb /= max_val

        return rgb

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process a single audio chunk for real-time streaming.

        Args:
            chunk: (fft_window,) audio samples.

        Returns:
            (n_output_channels,) single RGB frame.
        """
        if len(chunk) < self.config.fft_window:
            chunk = np.pad(chunk, (0, self.config.fft_window - len(chunk)))

        # Single-frame FFT
        spectrum = np.abs(np.fft.rfft(chunk))
        output = np.zeros(self.config.n_output_channels)

        for mapping in self.config.mappings:
            mask = (self._freq_bins >= mapping.freq_low) & (self._freq_bins < mapping.freq_high)
            band_energy = float(np.mean(spectrum[mask])) if np.any(mask) else 0.0
            transformed = _apply_transfer(np.array([band_energy]), mapping.transfer_function)[0]
            output[mapping.output_channel] += transformed * mapping.gain

        output *= self.config.master_gain * self.config.intensity

        # Clamp to [0, 1]
        max_val = np.max(output)
        if max_val > 0:
            output /= max_val

        return output

    def frequency_to_rgb(self, frequency_hz: float, amplitude: float = 1.0) -> tuple[float, float, float]:
        """Map a single pure tone to an RGB color.

        Useful for visualization and debugging.

        Args:
            frequency_hz: Tone frequency in Hz.
            amplitude: Tone amplitude [0, 1].

        Returns:
            (R, G, B) tuple in [0, 1].
        """
        rgb = [0.0, 0.0, 0.0]
        for mapping in self.config.mappings:
            if mapping.freq_low <= frequency_hz < mapping.freq_high:
                # Position within the band [0, 1]
                band_width = mapping.freq_high - mapping.freq_low
                position = (frequency_hz - mapping.freq_low) / band_width
                value = _apply_transfer(
                    np.array([amplitude * position]),
                    mapping.transfer_function,
                )[0] * mapping.gain
                rgb[mapping.output_channel] += value

        # Normalize
        max_val = max(rgb) if max(rgb) > 0 else 1.0
        return tuple(v / max_val for v in rgb)

    def _stft_magnitude(self, audio: np.ndarray) -> np.ndarray:
        """Compute STFT magnitude spectrogram."""
        window = self.config.fft_window
        hop = self.config.hop_size
        n_frames = max(1, (len(audio) - window) // hop + 1)

        spectrogram = np.zeros((n_frames, window // 2 + 1))
        for i in range(n_frames):
            start = i * hop
            frame = audio[start : start + window]
            if len(frame) < window:
                frame = np.pad(frame, (0, window - len(frame)))
            # Apply Hann window for spectral leakage reduction
            windowed = frame * np.hanning(window)
            spectrogram[i] = np.abs(np.fft.rfft(windowed))

        return spectrogram

    def _extract_band_energy(
        self, spectrogram: np.ndarray, mapping: SynesthesiaMapping,
    ) -> np.ndarray:
        """Extract mean energy in a frequency band across all frames."""
        mask = (self._freq_bins >= mapping.freq_low) & (self._freq_bins < mapping.freq_high)
        if not np.any(mask):
            return np.zeros(spectrogram.shape[0])
        return np.mean(spectrogram[:, mask], axis=1)


# ---------------------------------------------------------------------------
# Spike Train Encoder — converts RGB to electrode stimulation patterns
# ---------------------------------------------------------------------------

class SpikeTrainEncoder:
    """Converts continuous RGB values into discrete spike trains for electrode stimulation.

    Uses rate coding: higher RGB value → higher spike rate.
    Each output channel drives a group of electrodes in the target cortical zone.
    """

    def __init__(
        self,
        max_rate_hz: float = 300.0,
        min_rate_hz: float = 5.0,
        dt_ms: float = 1.0,
    ) -> None:
        self.max_rate = max_rate_hz
        self.min_rate = min_rate_hz
        self.dt = dt_ms / 1000.0  # convert to seconds

    def encode(self, values: np.ndarray, duration_ms: float = 20.0) -> np.ndarray:
        """Convert continuous values to spike train.

        Args:
            values: (N_channels,) continuous values in [0, 1].
            duration_ms: Duration of the spike train window.

        Returns:
            (N_channels, N_timesteps) binary spike array.
        """
        n_channels = len(values)
        n_steps = int(duration_ms / (self.dt * 1000))
        spikes = np.zeros((n_channels, n_steps), dtype=np.int8)

        for ch in range(n_channels):
            rate = self.min_rate + values[ch] * (self.max_rate - self.min_rate)
            spike_prob = rate * self.dt
            spikes[ch] = (np.random.rand(n_steps) < spike_prob).astype(np.int8)

        return spikes
