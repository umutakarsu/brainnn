"""Base decoder interface for BCI classification."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDecoder(ABC):
    """Abstract base class for BCI decoders.

    Both classical (CSP+LDA) and neural (EEGNet) decoders follow this interface.
    Neural decoders also inherit from nn.Module for PyTorch compatibility.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseDecoder:
        """Train the decoder on labeled EEG data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for EEG data."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        return float(np.mean(self.predict(X) == y))
