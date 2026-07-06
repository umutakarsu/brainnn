"""Full LOSO training run — kicks off cross-subject evaluation on all 9 subjects.

Saves:
    /Users/umutakarsu/brainnn/checkpoints/loso_results.json
    /Users/umutakarsu/brainnn/checkpoints/model_fold_9.pt   (last fold, for dashboard)
"""

import sys
sys.path.insert(0, "/Users/umutakarsu/brainnn/src")

import json
import time
from pathlib import Path

import torch

from brainnn.bci.datasets import load_all_bnci_subjects, stack_subjects
from brainnn.bci.models import EEGTransformer, EEGTransformerConfig
from brainnn.bci.training import run_loso, TrainConfig, summarize_loso


def main():
    ckpt_dir = Path("/Users/umutakarsu/brainnn/checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print("Loading all 9 BNCI2014_001 subjects...")
    recordings = load_all_bnci_subjects(subject_ids=list(range(1, 10)), verbose=True)

    print(f"\nTotal epochs across subjects: "
          f"{sum(r.n_epochs for r in recordings)}")

    # 25 epochs is enough for prototype; can grow later
    train_cfg = TrainConfig(n_epochs=25, batch_size=64, lr=3e-4)

    t0 = time.time()
    results = run_loso(recordings, cfg=train_cfg, verbose=True)
    total_time = time.time() - t0

    summary = summarize_loso(results)
    summary["total_train_time_s"] = total_time
    summary["dataset"] = "BNCI2014_001"
    summary["train_config"] = {
        "n_epochs": train_cfg.n_epochs,
        "batch_size": train_cfg.batch_size,
        "lr": train_cfg.lr,
    }

    # Full per-fold with training curves
    summary["per_fold_full"] = [
        {
            "subject": r.held_out_subject,
            "zero_shot_acc": r.zero_shot_acc,
            "train_curve": r.train_curve,
            "val_curve": r.val_curve,
            "train_time_s": r.train_time_s,
        }
        for r in results
    ]

    out_json = ckpt_dir / "loso_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FULL LOSO COMPLETE — {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"Zero-shot mean:  {summary['zero_shot_mean']:.3f}")
    print(f"Zero-shot std:   {summary['zero_shot_std']:.3f}")
    print(f"Best fold:       {summary['zero_shot_max']:.3f}")
    print(f"Worst fold:      {summary['zero_shot_min']:.3f}")
    print(f"Chance:          0.250 (4-class)")
    print(f"\nSaved to: {out_json}")

    # Retrain a model on subjects 1-8, save for dashboard use
    print("\n>>> Retraining single model on subjects 1-8 for dashboard...")
    train_recs = recordings[:8]  # subjects 1-8
    train_X, train_y, train_sids = stack_subjects(train_recs)

    unique_sids = sorted(set(train_sids.tolist()))
    sid_remap = {s: i for i, s in enumerate(unique_sids)}
    import numpy as np
    train_sids_remapped = np.array([sid_remap[s] for s in train_sids], dtype=np.int64)

    model_cfg = EEGTransformerConfig(
        n_channels=22, n_timepoints=751, n_classes=4,
        n_subjects=len(unique_sids),
    )
    model = EEGTransformer(model_cfg)

    from brainnn.bci.training import train_cross_subject
    train_cross_subject(
        train_X, train_y, train_sids_remapped,
        recordings[8].X, recordings[8].y,  # held out = subject 9
        model, train_cfg, verbose=True,
    )

    ckpt = {
        "model_state": model.state_dict(),
        "model_cfg": {
            "n_channels": model_cfg.n_channels,
            "n_timepoints": model_cfg.n_timepoints,
            "n_classes": model_cfg.n_classes,
            "n_subjects": model_cfg.n_subjects,
            "embed_dim": model_cfg.embed_dim,
        },
        "class_names": recordings[0].class_names,
        "sid_remap": sid_remap,
    }
    ckpt_path = ckpt_dir / "eeg_transformer_subj1-8.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved dashboard checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
