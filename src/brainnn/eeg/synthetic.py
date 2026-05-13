"""Synthetic EEG generator for BCI experiments.

Generates realistic multi-channel EEG with:
- Physiological frequency bands (delta through gamma)
- 1/f pink noise (characteristic of neural background activity)
- Cross-frequency coupling (theta-gamma PAC)
- Motor imagery patterns (Event-Related Desynchronization)

Supports 3-channel (C3/Cz/C4) and 22-channel (BCI Competition IV) montages,
with 2-class (left/right hand) or 4-class (left/right/feet/tongue) paradigms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brainnn.eeg.channels import CLASS_NAMES, MONTAGES, MOTOR_IMAGERY_CHANNELS_22

FREQ_BANDS = {
    "delta": (1, 4, 20.0),
    "theta": (4, 8, 10.0),
    "alpha": (8, 13, 15.0),
    "beta": (13, 30, 5.0),
    "gamma": (30, 50, 2.0),
}


@dataclass
class SyntheticEEGConfig:
    n_channels: int = 22
    n_classes: int = 4
    sfreq: float = 128.0
    duration: float = 2.0
    snr: float = 0.5
    pink_noise: bool = True
    seed: int | None = None


class SyntheticMotorImagery:
    """Generate synthetic motor imagery EEG data.

    Simulates Event-Related Desynchronization (ERD) in mu/beta rhythms
    with realistic spatial patterns over motor cortex. Includes 1/f
    background noise and cross-frequency coupling.
    """

    def __init__(self, config: SyntheticEEGConfig | None = None) -> None:
        self.config = config or SyntheticEEGConfig()
        self.rng = np.random.default_rng(self.config.seed)

        if self.config.n_channels not in MONTAGES:
            raise ValueError(f"Supported channel counts: {list(MONTAGES.keys())}")

        class_names = CLASS_NAMES.get(self.config.n_classes)
        if class_names is None:
            raise ValueError(f"Supported class counts: {list(CLASS_NAMES.keys())}")

    def generate(self, n_trials: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic motor imagery dataset.

        Returns:
            X: (n_trials, n_channels, n_samples)
            y: (n_trials,) class labels (0-indexed)
        """
        n_ch = self.config.n_channels
        n_samples = int(self.config.sfreq * self.config.duration)

        X = np.zeros((n_trials, n_ch, n_samples))
        y = self.rng.integers(0, self.config.n_classes, size=n_trials)
        t = np.linspace(0, self.config.duration, n_samples, endpoint=False)

        for i in range(n_trials):
            X[i] = self._generate_base_eeg(t)
            X[i] += self._add_pac(t)
            X[i] += self._motor_imagery_modulation(t, label=y[i])
            X[i] += self._generate_noise(n_ch, n_samples)

        return X, y

    def _generate_base_eeg(self, t: np.ndarray) -> np.ndarray:
        n_ch = self.config.n_channels
        n_samples = len(t)
        eeg = np.zeros((n_ch, n_samples))

        for ch in range(n_ch):
            for _name, (f_low, f_high, amp) in FREQ_BANDS.items():
                freq = self.rng.uniform(f_low, f_high)
                phase = self.rng.uniform(0, 2 * np.pi)
                eeg[ch] += amp * np.sin(2 * np.pi * freq * t + phase)

        # Spatial correlation: nearby channels share some signal
        if n_ch > 3:
            positions = MONTAGES[n_ch]["positions"]
            for ch in range(n_ch):
                for other in range(n_ch):
                    if ch != other:
                        dist = np.linalg.norm(positions[ch] - positions[other])
                        if dist < 0.4:
                            eeg[ch] += 0.3 * (1 - dist / 0.4) * eeg[other]

        return eeg

    def _add_pac(self, t: np.ndarray) -> np.ndarray:
        """Add cross-frequency coupling (theta phase → gamma amplitude)."""
        n_ch = self.config.n_channels
        pac = np.zeros((n_ch, len(t)))

        for ch in range(n_ch):
            theta_freq = self.rng.uniform(5, 7)
            gamma_freq = self.rng.uniform(35, 45)
            theta_phase = np.sin(2 * np.pi * theta_freq * t)

            # Gamma amplitude is modulated by theta phase
            gamma_envelope = 1.0 + 0.6 * theta_phase
            gamma_signal = gamma_envelope * np.sin(2 * np.pi * gamma_freq * t)
            pac[ch] = 2.0 * gamma_signal * self.rng.uniform(0.5, 1.5)

        return pac

    def _motor_imagery_modulation(self, t: np.ndarray, label: int) -> np.ndarray:
        n_ch = self.config.n_channels
        mod = np.zeros((n_ch, len(t)))

        class_name = CLASS_NAMES[self.config.n_classes][label]

        mu_freq = self.rng.uniform(9, 11)
        beta_freq = self.rng.uniform(18, 22)
        mu_signal = np.sin(2 * np.pi * mu_freq * t)
        beta_signal = np.sin(2 * np.pi * beta_freq * t)

        if n_ch == 3:
            if class_name == "left_hand":
                contra, ipsi = 2, 0
            else:
                contra, ipsi = 0, 2
            mod[contra] -= 10.0 * mu_signal + 4.0 * beta_signal
            mod[ipsi] += 3.0 * mu_signal + 1.5 * beta_signal
        else:
            channels = MOTOR_IMAGERY_CHANNELS_22[class_name]
            for ch in channels["contra"]:
                strength = self.rng.uniform(0.8, 1.2)
                mod[ch] -= strength * (10.0 * mu_signal + 4.0 * beta_signal)
            for ch in channels.get("ipsi", []):
                strength = self.rng.uniform(0.5, 0.9)
                mod[ch] += strength * (3.0 * mu_signal + 1.5 * beta_signal)

        return mod

    def _generate_noise(self, n_ch: int, n_samples: int) -> np.ndarray:
        noise_std = 1.0 / max(self.config.snr, 1e-6)

        if self.config.pink_noise:
            noise = np.zeros((n_ch, n_samples))
            for ch in range(n_ch):
                noise[ch] = noise_std * _pink_noise(n_samples, self.rng)
            return noise

        return self.rng.normal(0, noise_std, (n_ch, n_samples))


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f pink noise (characteristic of neural background activity)."""
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0
    magnitude = 1.0 / np.sqrt(freqs)
    phase = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = magnitude * np.exp(1j * phase)
    signal = np.fft.irfft(spectrum, n=n_samples)
    return signal / (signal.std() + 1e-10)
