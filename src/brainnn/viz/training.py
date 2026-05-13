"""Training visualization — curves, confusion matrices, model comparison."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_training_history(
    history: dict[str, list[float | None]],
    title: str = "Training History",
) -> Figure:
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train", linewidth=1.5)
    if history.get("val_loss") and history["val_loss"][0] is not None:
        ax1.plot(epochs, history["val_loss"], "r-", label="Val", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train", linewidth=1.5)
    if history.get("val_acc") and history["val_acc"][0] is not None:
        ax2.plot(epochs, history["val_acc"], "r-", label="Val", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> Figure:
    """Plot confusion matrix."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")

    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=11)

    labels = class_names or [str(c) for c in classes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    title: str = "Decoder Benchmark",
) -> Figure:
    """Bar chart comparing model performance.

    Args:
        results: {model_name: {"train_acc": ..., "val_acc": ..., "params": ...}}
    """
    models = list(results.keys())
    train_accs = [results[m].get("train_acc", 0) for m in models]
    val_accs = [results[m].get("val_acc", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, train_accs, width, label="Train", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, val_accs, width, label="Validation", color="#DD8452", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=8)

    # Add parameter count annotations
    for i, m in enumerate(models):
        params = results[m].get("params")
        if params is not None:
            label = f"{params / 1000:.1f}K" if params >= 1000 else str(params)
            ax.text(i, -0.05, label, ha="center", fontsize=7, color="gray")

    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
