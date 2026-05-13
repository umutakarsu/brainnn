"""Trainer — training loop for PyTorch BCI decoders."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Training loop for PyTorch-based BCI decoders (EEGNet, Transformer, etc).

    Args:
        model: nn.Module to train
        lr: learning rate
        weight_decay: L2 regularization
        device: 'cpu', 'cuda', or 'mps'
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.history: dict[str, list[float | None]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        verbose: bool = True,
    ) -> dict[str, list[float | None]]:
        """Train the model.

        Args:
            train_loader: training DataLoader yielding (X, y) batches
            val_loader: optional validation DataLoader
            epochs: number of training epochs
            verbose: print progress each epoch

        Returns:
            Training history dict
        """
        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_loader)

            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if verbose:
                msg = f"[{epoch + 1:3d}/{epochs}] loss={train_loss:.4f} acc={train_acc:.3f}"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
                print(msg)

        return self.history

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluate model on a DataLoader. Returns (loss, accuracy)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
        return total_loss / total, correct / total

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint["history"]

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        return total_loss / total, correct / total
