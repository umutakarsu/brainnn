"""CSP + LDA — classical baseline decoder for motor imagery BCI.

Common Spatial Patterns extract spatial filters that maximize variance
difference between classes. Linear Discriminant Analysis classifies
the log-variance features. Supports 2-class and multi-class (one-vs-rest).
"""

from __future__ import annotations

import numpy as np

from brainnn.decoder.base import BaseDecoder
from brainnn.signal.features import compute_csp, csp_features


class LDA:
    """Minimal 2-class Linear Discriminant Analysis."""

    def __init__(self) -> None:
        self.w: np.ndarray | None = None
        self.threshold: float = 0.0
        self.classes: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LDA:
        self.classes = np.unique(y)
        X0, X1 = X[y == self.classes[0]], X[y == self.classes[1]]

        mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
        S0 = (X0 - mu0).T @ (X0 - mu0)
        S1 = (X1 - mu1).T @ (X1 - mu1)
        Sw = S0 + S1 + np.eye(X.shape[1]) * 1e-6

        self.w = np.linalg.solve(Sw, mu1 - mu0)
        self.threshold = 0.5 * self.w @ (mu0 + mu1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        projections = X @ self.w
        return np.where(projections > self.threshold, self.classes[1], self.classes[0])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w - self.threshold


class CSPLDADecoder(BaseDecoder):
    """CSP + LDA decoder for motor imagery.

    For 2 classes: standard CSP + LDA.
    For >2 classes: one-vs-rest with per-class CSP spatial filters.

    Args:
        n_components: number of CSP spatial filters per class pair
    """

    def __init__(self, n_components: int = 4) -> None:
        self.n_components = n_components
        self.W: np.ndarray | None = None
        self.lda = LDA()
        self._ovr: list[tuple[int, np.ndarray, LDA]] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> CSPLDADecoder:
        classes = np.unique(y)

        if len(classes) == 2:
            self.W = compute_csp(X, y, self.n_components)
            features = csp_features(X, self.W)
            self.lda.fit(features, y)
        else:
            self._ovr = []
            for c in classes:
                binary_y = np.where(y == c, 1, 0)
                W = compute_csp(X, binary_y, self.n_components)
                features = csp_features(X, W)
                lda = LDA()
                lda.fit(features, binary_y)
                self._ovr.append((c, W, lda))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._ovr is not None:
            scores = np.column_stack([
                csp_features(X, W) @ lda.w
                for _, W, lda in self._ovr
            ])
            classes = np.array([c for c, _, _ in self._ovr])
            return classes[scores.argmax(axis=1)]

        if self.W is None:
            raise RuntimeError("Call fit() before predict()")
        features = csp_features(X, self.W)
        return self.lda.predict(features)
