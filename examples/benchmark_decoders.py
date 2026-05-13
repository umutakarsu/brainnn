"""Benchmark all BCI decoders on 22-channel, 4-class motor imagery.

Compares:
  1. CSP + LDA (classical baseline)
  2. EEGNet (compact CNN — Lawhern et al. 2018)
  3. EEG Transformer (patch-based attention)
  4. SynapseFlow (neuromorphic decoder)

Usage:
    python examples/benchmark_decoders.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from brainnn.eeg.channels import CLASS_NAMES
from brainnn.eeg.preprocessing import bandpass_filter, normalize
from brainnn.eeg.synthetic import SyntheticEEGConfig, SyntheticMotorImagery
from brainnn.decoder.csp_lda import CSPLDADecoder
from brainnn.decoder.eegnet import EEGNet
from brainnn.decoder.transformer import EEGTransformer
from brainnn.decoder.synapseflow import SynapseFlowDecoder
from brainnn.training.trainer import Trainer
from brainnn.viz.training import plot_model_comparison


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_pytorch_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str,
    epochs: int = 40,
    lr: float = 1e-3,
) -> dict[str, float]:
    print(f"\n{'=' * 50}")
    print(f"  {name}  ({count_params(model):,} parameters)")
    print(f"{'=' * 50}")

    trainer = Trainer(model, lr=lr)
    history = trainer.fit(train_loader, val_loader, epochs=epochs)

    return {
        "train_acc": history["train_acc"][-1],
        "val_acc": history["val_acc"][-1],
        "params": count_params(model),
    }


def main():
    # --- Configuration ---
    config = SyntheticEEGConfig(
        n_channels=22,
        n_classes=4,
        sfreq=128,
        duration=2.0,
        snr=0.5,
        pink_noise=True,
        seed=42,
    )
    n_trials = 800
    epochs = 40
    class_names = CLASS_NAMES[config.n_classes]
    n_samples = int(config.sfreq * config.duration)

    # --- Generate data ---
    print("=" * 50)
    print("  SynapseFlow Decoder Benchmark")
    print("=" * 50)
    print(f"\nGenerating {n_trials} trials ({config.n_channels}ch, {config.n_classes}-class)...")

    gen = SyntheticMotorImagery(config)
    X, y = gen.generate(n_trials)
    X = bandpass_filter(X, low=4, high=40, sfreq=config.sfreq)
    X = normalize(X)

    print(f"  Shape: {X.shape}")
    print(f"  Classes: {class_names}")
    print(f"  Distribution: {[int((y == i).sum()) for i in range(config.n_classes)]}")

    # --- Split ---
    split = int(0.8 * len(y))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

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

    results: dict[str, dict[str, float]] = {}

    # --- 1. CSP + LDA ---
    print(f"\n{'=' * 50}")
    print("  CSP + LDA  (classical baseline)")
    print(f"{'=' * 50}")
    csp = CSPLDADecoder(n_components=4)
    csp.fit(X_train, y_train)
    csp_train = csp.score(X_train, y_train)
    csp_val = csp.score(X_val, y_val)
    print(f"  train acc: {csp_train:.3f}")
    print(f"  val acc:   {csp_val:.3f}")
    results["CSP+LDA"] = {"train_acc": csp_train, "val_acc": csp_val, "params": 0}

    # --- 2. EEGNet ---
    eegnet = EEGNet(n_channels=22, n_classes=4, n_samples=n_samples)
    results["EEGNet"] = train_pytorch_model(
        eegnet, train_loader, val_loader, "EEGNet", epochs=epochs
    )

    # --- 3. EEG Transformer ---
    transformer = EEGTransformer(n_channels=22, n_classes=4, n_samples=n_samples)
    results["Transformer"] = train_pytorch_model(
        transformer, train_loader, val_loader, "EEG Transformer",
        epochs=epochs, lr=5e-4
    )

    # --- 4. SynapseFlow ---
    synapseflow = SynapseFlowDecoder(n_channels=22, n_classes=4, n_samples=n_samples)
    results["SynapseFlow"] = train_pytorch_model(
        synapseflow, train_loader, val_loader, "SynapseFlow (Neuromorphic)",
        epochs=epochs, lr=1e-3
    )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  {'Model':<16} {'Train':>8} {'Val':>8} {'Params':>10}")
    print(f"  {'-' * 44}")
    for name, res in results.items():
        p = f"{res['params']:,}" if res["params"] > 0 else "-"
        print(f"  {name:<16} {res['train_acc']:>7.1%} {res['val_acc']:>7.1%} {p:>10}")
    print(f"{'=' * 60}")

    # --- Save comparison plot ---
    fig = plot_model_comparison(results)
    fig.savefig("benchmark_results.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved benchmark_results.png")


if __name__ == "__main__":
    main()
