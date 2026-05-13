"""EEG channel montages and electrode positions."""

from __future__ import annotations

import numpy as np

MONTAGES: dict[int, dict] = {
    3: {
        "names": ["C3", "Cz", "C4"],
        "positions": np.array([(-0.35, 0.0), (0.0, 0.0), (0.35, 0.0)]),
    },
    22: {
        "names": [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CPz", "CP2", "CP4",
            "P1", "Pz", "P2", "POz",
        ],
        "positions": np.array([
            (0.0, 0.6),
            (-0.35, 0.3), (-0.17, 0.3), (0.0, 0.3), (0.17, 0.3), (0.35, 0.3),
            (-0.7, 0.0), (-0.35, 0.0), (-0.17, 0.0), (0.0, 0.0),
            (0.17, 0.0), (0.35, 0.0), (0.7, 0.0),
            (-0.35, -0.3), (-0.17, -0.3), (0.0, -0.3), (0.17, -0.3), (0.35, -0.3),
            (-0.17, -0.6), (0.0, -0.6), (0.17, -0.6),
            (0.0, -0.8),
        ]),
    },
}

# Motor cortex channel indices for each imagery class (22-channel layout)
MOTOR_IMAGERY_CHANNELS_22: dict[str, dict[str, list[int]]] = {
    "left_hand": {
        "contra": [11, 10, 5, 17],   # C4, C2, FC4, CP4
        "ipsi": [7, 8, 1, 13],       # C3, C1, FC3, CP3
    },
    "right_hand": {
        "contra": [7, 8, 1, 13],     # C3, C1, FC3, CP3
        "ipsi": [11, 10, 5, 17],     # C4, C2, FC4, CP4
    },
    "feet": {
        "contra": [9, 3, 15],        # Cz, FCz, CPz
        "ipsi": [],
    },
    "tongue": {
        "contra": [7, 11, 0, 3],     # C3, C4, Fz, FCz
        "ipsi": [],
    },
}

CLASS_NAMES = {
    2: ["left_hand", "right_hand"],
    4: ["left_hand", "right_hand", "feet", "tongue"],
}


def get_montage(n_channels: int) -> dict:
    if n_channels not in MONTAGES:
        raise ValueError(f"No montage for {n_channels} channels. Available: {list(MONTAGES.keys())}")
    return MONTAGES[n_channels]


def channel_distances(n_channels: int) -> np.ndarray:
    """Pairwise Euclidean distances between electrodes."""
    positions = get_montage(n_channels)["positions"]
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))
