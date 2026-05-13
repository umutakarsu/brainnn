"""Cross-frequency coupling and connectivity analysis.

Phase-amplitude coupling (PAC) is a key mechanism for long-range neural
communication. Low-frequency oscillations (theta, alpha) modulate the
amplitude of high-frequency activity (gamma), coordinating information
transfer across cortical regions.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def analytic_signal(x: np.ndarray) -> np.ndarray:
    """Compute analytic signal via Hilbert transform.

    Returns complex array: real part = original signal,
    imaginary part = Hilbert transform.
    """
    N = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    h = np.zeros(N)
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    if N % 2 == 0:
        h[N // 2] = 1
    return np.fft.ifft(X * h, axis=-1)


def instantaneous_phase(x: np.ndarray) -> np.ndarray:
    """Extract instantaneous phase from signal."""
    return np.angle(analytic_signal(x))


def instantaneous_amplitude(x: np.ndarray) -> np.ndarray:
    """Extract instantaneous amplitude envelope."""
    return np.abs(analytic_signal(x))


def phase_amplitude_coupling(
    x_phase: np.ndarray,
    x_amp: np.ndarray,
    sfreq: float,
    phase_band: tuple[float, float] = (4, 8),
    amp_band: tuple[float, float] = (30, 50),
    n_bins: int = 18,
) -> float:
    """Compute modulation index (MI) for phase-amplitude coupling.

    Measures how much the amplitude of high-frequency oscillations is
    modulated by the phase of low-frequency oscillations.

    Args:
        x_phase: signal to extract phase from
        x_amp: signal to extract amplitude from
        sfreq: sampling frequency
        phase_band: frequency range for phase extraction
        amp_band: frequency range for amplitude extraction
        n_bins: number of phase bins

    Returns:
        Modulation index (0 = no coupling, higher = stronger coupling)
    """
    nyq = sfreq / 2.0

    b_phase, a_phase = sp_signal.butter(4, [phase_band[0] / nyq, phase_band[1] / nyq], "band")
    b_amp, a_amp = sp_signal.butter(4, [amp_band[0] / nyq, amp_band[1] / nyq], "band")

    phase_filtered = sp_signal.filtfilt(b_phase, a_phase, x_phase, axis=-1)
    amp_filtered = sp_signal.filtfilt(b_amp, a_amp, x_amp, axis=-1)

    phase = instantaneous_phase(phase_filtered)
    amplitude = instantaneous_amplitude(amp_filtered)

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.any():
            mean_amp[i] = amplitude[mask].mean()

    if mean_amp.sum() == 0:
        return 0.0

    mean_amp /= mean_amp.sum()
    uniform = np.ones(n_bins) / n_bins
    kl = np.sum(mean_amp * np.log(mean_amp / uniform + 1e-10))
    return kl / np.log(n_bins)


def spectral_coherence(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude-squared coherence between two signals.

    Returns:
        (frequencies, coherence) arrays
    """
    freqs, Cxy = sp_signal.coherence(x, y, fs=sfreq, nperseg=min(nperseg, len(x)))
    return freqs, Cxy


def connectivity_matrix(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float] = (8, 13),
    method: str = "coherence",
) -> np.ndarray:
    """Compute pairwise connectivity between channels.

    Args:
        data: (n_channels, n_samples)
        sfreq: sampling frequency
        band: frequency band for connectivity
        method: 'coherence' or 'plv' (phase locking value)

    Returns:
        (n_channels, n_channels) connectivity matrix
    """
    n_ch = data.shape[0]
    conn = np.zeros((n_ch, n_ch))

    if method == "coherence":
        for i in range(n_ch):
            for j in range(i, n_ch):
                freqs, coh = spectral_coherence(data[i], data[j], sfreq)
                mask = (freqs >= band[0]) & (freqs <= band[1])
                conn[i, j] = conn[j, i] = coh[mask].mean() if mask.any() else 0.0

    elif method == "plv":
        nyq = sfreq / 2.0
        b, a = sp_signal.butter(4, [band[0] / nyq, band[1] / nyq], "band")
        filtered = sp_signal.filtfilt(b, a, data, axis=-1)
        phases = instantaneous_phase(filtered)

        for i in range(n_ch):
            for j in range(i, n_ch):
                plv = np.abs(np.mean(np.exp(1j * (phases[i] - phases[j]))))
                conn[i, j] = conn[j, i] = plv

    return conn
