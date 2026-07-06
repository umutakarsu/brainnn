"""Multi-subject EEG dataset loading via MOABB.

Uses BNCI2014_001 (Brunner et al., 2008) — the standard 4-class motor-imagery
dataset for cross-subject BCI benchmarking:
  - 9 subjects
  - 22 EEG channels @ 250 Hz
  - 4 classes: left hand, right hand, feet, tongue
  - 6 runs per session × 2 sessions per subject
  - Public, well-documented, small enough to fit in memory

Reference: Brunner et al. (2008), BCI Competition IV Dataset 2a.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Cache directory for MOABB downloads
CACHE_DIR = "/Users/umutakarsu/brainnn/data/moabb"


@dataclass
class SubjectRecording:
    """One subject's data in a normalized shape."""
    subject_id: int
    # (N_epochs, N_channels, N_timepoints) - epoched EEG
    X: np.ndarray
    # (N_epochs,) - class labels [0..N_classes-1]
    y: np.ndarray
    # Sampling rate in Hz
    sfreq: float
    # Channel names (length = N_channels)
    ch_names: list[str]
    # Class name lookup
    class_names: list[str]

    @property
    def n_epochs(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_timepoints(self) -> int:
        return self.X.shape[2]


def load_bnci_subject(
    subject_id: int,
    tmin: float = 0.5,
    tmax: float = 3.5,
    bandpass: tuple[float, float] = (4.0, 40.0),
) -> SubjectRecording:
    """Load one subject from BNCI2014_001 with basic preprocessing.

    Args:
        subject_id: 1..9
        tmin, tmax: epoch window in seconds relative to motor-imagery cue.
                    Default (0.5, 3.5) skips the visual cue and captures the MI period.
        bandpass: filter cutoffs. (4, 40) Hz captures mu + beta rhythms.

    Returns:
        SubjectRecording with normalized channel means / stds.
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery, MotorImagery

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4,
        fmin=bandpass[0],
        fmax=bandpass[1],
        tmin=tmin,
        tmax=tmax,
        resample=250,  # native rate; forced explicit
    )

    X, y_str, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    # y comes as string labels ('left_hand', etc.) - convert to int
    unique_labels = sorted(set(y_str))
    label_to_int = {name: i for i, name in enumerate(unique_labels)}
    y = np.array([label_to_int[s] for s in y_str], dtype=np.int64)

    # Per-channel z-score normalization within this subject
    # (Cross-subject models learn better when each subject has zero-mean / unit-std
    # per-channel signals; otherwise subject-specific gains dominate.)
    X = X.astype(np.float32)
    ch_mean = X.mean(axis=(0, 2), keepdims=True)
    ch_std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    X = (X - ch_mean) / ch_std

    # MOABB doesn't expose channel names directly through paradigm.get_data;
    # grab from underlying MNE raw for one session for reference
    channels = [f"ch{i:02d}" for i in range(X.shape[1])]

    return SubjectRecording(
        subject_id=subject_id,
        X=X,
        y=y,
        sfreq=250.0,
        ch_names=channels,
        class_names=unique_labels,
    )


def load_all_bnci_subjects(
    subject_ids: Optional[list[int]] = None,
    verbose: bool = True,
) -> list[SubjectRecording]:
    """Load all 9 BNCI2014_001 subjects (or a subset).

    First call will download ~1.5 GB total to MOABB's cache. Subsequent calls are fast.
    """
    if subject_ids is None:
        subject_ids = list(range(1, 10))  # 1..9

    recordings = []
    for sid in subject_ids:
        if verbose:
            print(f"Loading subject {sid}...")
        rec = load_bnci_subject(sid)
        if verbose:
            print(f"  {rec.n_epochs} epochs, {rec.n_channels} channels, "
                  f"{rec.n_timepoints} timepoints, classes={rec.class_names}")
        recordings.append(rec)
    return recordings


def stack_subjects(recordings: list[SubjectRecording]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate subjects into (X, y, subject_ids) arrays for batch training."""
    X = np.concatenate([r.X for r in recordings], axis=0)
    y = np.concatenate([r.y for r in recordings], axis=0)
    sids = np.concatenate([
        np.full(r.n_epochs, r.subject_id, dtype=np.int64)
        for r in recordings
    ])
    return X, y, sids


if __name__ == "__main__":
    # Quick smoke test: load subject 1 only
    print("=" * 60)
    print("BNCI2014_001 dataset smoke test — subject 1 only")
    print("=" * 60)
    rec = load_bnci_subject(subject_id=1)
    print(f"Subject:      {rec.subject_id}")
    print(f"X shape:      {rec.X.shape}  (epochs, channels, timepoints)")
    print(f"y shape:      {rec.y.shape}")
    print(f"Sample rate:  {rec.sfreq} Hz")
    print(f"Classes:      {rec.class_names}")
    print(f"Class counts: {dict(zip(*np.unique(rec.y, return_counts=True)))}")
    print(f"X range:      [{rec.X.min():.2f}, {rec.X.max():.2f}]")
    print(f"X mean/std:   {rec.X.mean():.3f} / {rec.X.std():.3f}")
