"""EEGNet — compact CNN for EEG-based brain-computer interfaces.

Based on Lawhern et al. 2018: "EEGNet: A Compact Convolutional Neural
Network for EEG-based Brain-Computer Interfaces"

Architecture:
    Temporal Conv → Depthwise Spatial Conv → Separable Conv → Dense
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """EEGNet — learns temporal, spatial, and spatio-temporal features from raw EEG.

    Args:
        n_channels: number of EEG channels (e.g. 3 for C3/Cz/C4)
        n_classes: number of output classes (e.g. 2 for left/right)
        n_samples: number of time samples per trial
        F1: number of temporal filters
        D: depth multiplier for depthwise conv
        F2: number of pointwise filters in separable conv
        dropout: dropout rate
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
        n_samples: int = 256,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        kernel_length = max(n_samples // 4, 4)

        # Block 1: temporal convolution + depthwise spatial convolution
        self.temporal_conv = nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable convolution (depthwise + pointwise)
        self.sep_depthwise = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False)
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier
        t_out = n_samples // 32
        assert t_out > 0, f"n_samples must be >= 32, got {n_samples}"
        self.fc = nn.Linear(F2 * t_out, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_channels, n_samples) raw EEG

        Returns:
            (batch, n_classes) logits
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        # Block 1
        x = self.bn1(self.temporal_conv(x))
        x = F.elu(self.bn2(self.depthwise_conv(x)))
        x = self.drop1(self.pool1(x))

        # Block 2
        x = self.sep_depthwise(x)
        x = F.elu(self.bn3(self.sep_pointwise(x)))
        x = self.drop2(self.pool2(x))

        return self.fc(x.flatten(1))
