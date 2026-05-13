"""Signal processing, feature extraction, and connectivity analysis."""

from brainnn.signal.coupling import (
    connectivity_matrix,
    instantaneous_amplitude,
    instantaneous_phase,
    phase_amplitude_coupling,
    spectral_coherence,
)
from brainnn.signal.features import (
    STANDARD_BANDS,
    all_band_powers,
    band_power,
    compute_csp,
    csp_features,
)

__all__ = [
    "STANDARD_BANDS",
    "all_band_powers",
    "band_power",
    "compute_csp",
    "connectivity_matrix",
    "csp_features",
    "instantaneous_amplitude",
    "instantaneous_phase",
    "phase_amplitude_coupling",
    "spectral_coherence",
]
