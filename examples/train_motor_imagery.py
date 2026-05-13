"""Train a motor imagery decoder from scratch using synthetic EEG data.

Compares three approaches:
  1. CSP + LDA (classical baseline)
  2. EEGNet (compact CNN)
  3. EEG Transformer

Usage:
    python examples/train_motor_imagery.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from brainnn.eeg.synthetic import SyntheticMotorImagery, SyntheticEEGConfig
from brainnn.eeg.preprocessing import bandpass_filter, normalize
from brainnn.decoder.csp_lda import CSPLDADecoder
from brainnn.decoder.eegnet import EEGNet
from brainnn.decoder.transformer import EEGTransformer
from brainnn.training.trainer import Trainer


def main():
    # --- Generate synthetic motor imagery data ---
    print("Generating synthetic motor imagery data (C3/Cz/C4)...")
    config = SyntheticEEGConfig(n_channels=3, sfreq=128, duration=2.0, snr=0.5, seed=42)
    gen = SyntheticMotorImagery(config)
    X, y = gen.generate(n_trials=600)

    n_samples = X.shape[-1]
    print(f"  {X.shape[0]} trials, {X.shape[1]} channels, {n_samples} samples/trial")

    # --- Preprocess ---
    X = bandpass_filter(X, low=4, high=40, sfreq=config.sfreq)
    X = normalize(X)

    # --- Train/val split ---
    split = int(0.8 * len(y))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"  train={len(y_train)}, val={len(y_val)}")

    # --- 1. CSP + LDA baseline ---
    print("\n=== CSP + LDA ===")
    csp = CSPLDADecoder(n_components=4)
    csp.fit(X_train, y_train)
    print(f"  train acc: {csp.score(X_train, y_train):.3f}")
    print(f"  val acc:   {csp.score(X_val, y_val):.3f}")

    # --- Prepare PyTorch DataLoaders ---
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # --- 2. EEGNet ---
    print("\n=== EEGNet ===")
    eegnet = EEGNet(n_channels=3, n_classes=2, n_samples=n_samples)
    trainer = Trainer(eegnet, lr=1e-3)
    trainer.fit(train_loader, val_loader, epochs=30)

    # --- 3. EEG Transformer ---
    print("\n=== EEG Transformer ===")
    transformer = EEGTransformer(n_channels=3, n_classes=2, n_samples=n_samples)
    trainer = Trainer(transformer, lr=5e-4)
    trainer.fit(train_loader, val_loader, epochs=30)

    print("\nDone!")


if __name__ == "__main__":
    main()
