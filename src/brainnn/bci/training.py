"""Leave-one-subject-out (LOSO) training for cross-subject EEG transformer.

The prototype question: does training on 8 subjects and evaluating on a 9th
(zero-shot) give better than chance (25% for 4-class) — and how does it compare
to per-subject fine-tuning?

This is the toy version of the "Hugging Face of BCI" thesis:
    Pre-train cross-subject → fine-tune per person → deploy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Note on device selection: for models this small (~665K params) the
    Metal/MPS backend is actually SLOWER than CPU because transfer
    overhead dominates the compute. We default to CPU.
    """
    n_epochs: int = 25
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    val_split: float = 0.15
    seed: int = 42
    device: str = "cpu"


@dataclass
class LOSOResult:
    """Results for one leave-one-subject-out fold."""
    held_out_subject: int
    zero_shot_acc: float          # accuracy on held-out subject WITHOUT any training on them
    per_subject_baseline_acc: float = 0.0  # subject-specific model (single-subject train)
    train_curve: list[float] = field(default_factory=list)
    val_curve: list[float] = field(default_factory=list)
    train_time_s: float = 0.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _epoch_accuracy(model, loader, device, use_subject_id: bool = True) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, sid = batch
                x, y, sid = x.to(device), y.to(device), sid.to(device)
                logits = model(x, subject_id=sid if use_subject_id else None)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x, subject_id=None)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train_cross_subject(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_sids: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    model,
    cfg: TrainConfig,
    verbose: bool = True,
) -> tuple[list[float], list[float]]:
    """Train `model` on (train_X, train_y, train_sids), evaluate on (test_X, test_y).

    Subject IDs on train are used for conditioning; the held-out test subject
    uses subject_id=None (zero-shot).

    Returns:
        (train_acc_curve, val_acc_curve) — one accuracy per epoch
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    set_seed(cfg.seed)

    # Convert to tensors
    train_X_t = torch.from_numpy(train_X).float()
    train_y_t = torch.from_numpy(train_y).long()
    train_sids_t = torch.from_numpy(train_sids).long()
    test_X_t = torch.from_numpy(test_X).float()
    test_y_t = torch.from_numpy(test_y).long()

    # Simple within-subject train/val split
    n = train_X_t.shape[0]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    n_val = int(n * cfg.val_split)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(train_X_t[tr_idx], train_y_t[tr_idx], train_sids_t[tr_idx])
    val_ds = TensorDataset(train_X_t[val_idx], train_y_t[val_idx], train_sids_t[val_idx])
    test_ds = TensorDataset(test_X_t, test_y_t)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Optimizer + cosine schedule with warmup
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        # cosine to 0.1x over remaining
        prog = (epoch - cfg.warmup_epochs) / max(cfg.n_epochs - cfg.warmup_epochs, 1)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * prog))

    train_acc_curve = []
    val_acc_curve = []

    for epoch in range(cfg.n_epochs):
        # Set LR
        for pg in optim.param_groups:
            pg["lr"] = cfg.lr * lr_at(epoch)

        model.train()
        total_loss = 0.0
        n_correct = 0
        n_total = 0
        for x, y, sid in train_loader:
            x, y, sid = x.to(device), y.to(device), sid.to(device)
            optim.zero_grad()
            logits = model(x, subject_id=sid)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item() * y.size(0)
            n_correct += (logits.argmax(dim=-1) == y).sum().item()
            n_total += y.size(0)

        train_acc = n_correct / n_total
        val_acc = _epoch_accuracy(model, val_loader, device, use_subject_id=True)
        train_acc_curve.append(train_acc)
        val_acc_curve.append(val_acc)

        if verbose and (epoch % 5 == 0 or epoch == cfg.n_epochs - 1):
            print(f"  ep {epoch+1:>3}/{cfg.n_epochs}  "
                  f"loss={total_loss/n_total:.3f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    return train_acc_curve, val_acc_curve


def run_loso(
    recordings,
    cfg: TrainConfig | None = None,
    model_cfg=None,
    verbose: bool = True,
) -> list[LOSOResult]:
    """Full leave-one-subject-out evaluation.

    For each subject i in {1..9}:
        - Train on all other 8 subjects (with their subject IDs)
        - Test on subject i with subject_id=None (zero-shot cross-subject)
        - Return zero-shot accuracy
    """
    from brainnn.bci.datasets import stack_subjects
    from brainnn.bci.models import EEGTransformer, EEGTransformerConfig

    if cfg is None:
        cfg = TrainConfig()
    if model_cfg is None:
        model_cfg = EEGTransformerConfig()

    results = []
    for i, held_out in enumerate(recordings):
        if verbose:
            print(f"\n=== LOSO fold {i+1}/{len(recordings)} — held out subject {held_out.subject_id} ===")

        # Train set: all except held-out
        train_recs = [r for r in recordings if r.subject_id != held_out.subject_id]
        train_X, train_y, train_sids = stack_subjects(train_recs)

        # Re-index subject IDs to be contiguous 0..7 for embedding table
        unique_sids = sorted(set(train_sids.tolist()))
        sid_remap = {s: i for i, s in enumerate(unique_sids)}
        train_sids_remapped = np.array([sid_remap[s] for s in train_sids], dtype=np.int64)

        # Model — subject_embed dim needs to match n_train_subjects + 1 (for unknown)
        this_model_cfg = EEGTransformerConfig(
            n_channels=train_X.shape[1],
            n_timepoints=train_X.shape[2],
            n_classes=int(train_y.max() + 1),
            n_subjects=len(unique_sids),
        )
        model = EEGTransformer(this_model_cfg)

        if verbose:
            print(f"  train: {train_X.shape[0]} epochs from {len(unique_sids)} subjects")
            print(f"  test:  {held_out.n_epochs} epochs from subject {held_out.subject_id}")
            print(f"  model params: {model.n_parameters():,}")

        t0 = time.time()
        train_curve, val_curve = train_cross_subject(
            train_X, train_y, train_sids_remapped,
            held_out.X, held_out.y,
            model, cfg, verbose=verbose,
        )
        train_time_s = time.time() - t0

        # Zero-shot evaluation on held-out subject (subject_id=None)
        model.eval()
        with torch.no_grad():
            device = torch.device(cfg.device)
            test_X_t = torch.from_numpy(held_out.X).float().to(device)
            test_y_t = torch.from_numpy(held_out.y).long().to(device)
            logits = model(test_X_t, subject_id=None)
            zero_shot_acc = (logits.argmax(dim=-1) == test_y_t).float().mean().item()

        if verbose:
            print(f"  ZERO-SHOT held-out acc: {zero_shot_acc:.3f}  "
                  f"(chance = {1/model_cfg.n_classes:.3f})")

        results.append(LOSOResult(
            held_out_subject=held_out.subject_id,
            zero_shot_acc=zero_shot_acc,
            train_curve=train_curve,
            val_curve=val_curve,
            train_time_s=train_time_s,
        ))

    return results


def summarize_loso(results: list[LOSOResult]) -> dict:
    """Aggregate stats across folds."""
    accs = [r.zero_shot_acc for r in results]
    return {
        "n_folds": len(results),
        "zero_shot_mean": float(np.mean(accs)),
        "zero_shot_std": float(np.std(accs)),
        "zero_shot_max": float(np.max(accs)),
        "zero_shot_min": float(np.min(accs)),
        "per_fold": [
            {"subject": r.held_out_subject, "acc": r.zero_shot_acc}
            for r in results
        ],
    }
