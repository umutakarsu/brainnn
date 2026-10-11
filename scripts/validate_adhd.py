"""Validate the ADHD model against the published reaction-time signature.

Runs simulated neurotypical and ADHD groups through a sustained-attention
task, fits an ex-Gaussian to each participant, and compares the group effect
sizes against the pattern reported in the meta-analytic literature:

    tau  (slow tail)         — large effect
    sigma (Gaussian spread)  — moderate effect
    mu   (central tendency)  — small effect

Outputs:
    checkpoints/adhd_validation.json   — all numbers
    docs/img/adhd_validation.png       — RT distributions + effect-size chart

Run:
    python scripts/validate_adhd.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from brainnn.behavior import (  # noqa: E402
    simulate_group, simulate_session, compare_groups,
    check_against_literature, LITERATURE_PATTERN,
)
from brainnn.core.phenotypes import get_phenotype  # noqa: E402


N_PARTICIPANTS = 80
N_TRIALS = 300


def main() -> int:
    groups = {
        "neurotypical": simulate_group("neurotypical", N_PARTICIPANTS, N_TRIALS, seed=1),
        "adhd_combined": simulate_group("adhd_combined", N_PARTICIPANTS, N_TRIALS, seed=2),
        "adhd_inattentive": simulate_group("adhd_inattentive", N_PARTICIPANTS, N_TRIALS, seed=3),
    }
    ctrl = groups["neurotypical"]

    report = {
        "n_participants": N_PARTICIPANTS,
        "n_trials": N_TRIALS,
        "literature": LITERATURE_PATTERN,
        "groups": {},
        "effects_vs_neurotypical": {},
    }

    print(f"{'phenotype':36s} {'mean':>6s} {'mu':>12s} {'sigma':>11s} {'tau':>12s}")
    for key, g in groups.items():
        report["groups"][key] = {
            k: {"mean": float(g[k].mean()), "sd": float(g[k].std())}
            for k in ("mean_rt", "mu", "sigma", "tau", "observed_lapse_rate")
        }
        print(f"{g['phenotype'][:36]:36s} {g['mean_rt'].mean():6.0f} "
              f"{g['mu'].mean():6.0f}({g['mu'].std():4.0f}) "
              f"{g['sigma'].mean():5.0f}({g['sigma'].std():3.0f}) "
              f"{g['tau'].mean():6.0f}({g['tau'].std():4.0f})")

    print()
    print("effect sizes vs neurotypical (Cohen's d)")
    print(f"  literature: {LITERATURE_PATTERN['ordering']}  bands={LITERATURE_PATTERN['bands']}")
    all_pass = True
    for key in ("adhd_combined", "adhd_inattentive"):
        e = compare_groups(groups[key], ctrl)
        chk = check_against_literature(e)
        report["effects_vs_neurotypical"][key] = {"d": e, "check": chk}
        status = "PASS" if chk["ordering_ok"] else "FAIL"
        all_pass &= chk["ordering_ok"]
        print(f"  {key:18s} tau={e['tau']:+.2f} sigma={e['sigma']:+.2f} mu={e['mu']:+.2f} "
              f"-> {chk['ordering_observed']}  [{status}]")

    (ROOT / "checkpoints").mkdir(exist_ok=True)
    out = ROOT / "checkpoints" / "adhd_validation.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}")

    _plot(groups, report, ROOT / "docs" / "img" / "adhd_validation.png")
    return 0 if all_pass else 1


def _plot(groups: dict, report: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"neurotypical": "#2a9d8f", "adhd_combined": "#e63946",
              "adhd_inattentive": "#f4a261"}

    # Left: pooled RT distributions from one representative participant each.
    ax = axes[0]
    for key in groups:
        s = simulate_session(get_phenotype(key), n_trials=1200, seed=42)
        ax.hist(s["rt"], bins=np.linspace(150, 1600, 90), density=True, alpha=0.45,
                color=colors[key], label=groups[key]["phenotype"])
    ax.set_xlabel("reaction time (ms)")
    ax.set_ylabel("density")
    ax.set_title("Simulated RT distributions — the ADHD slow tail")
    ax.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Right: effect sizes against literature bands.
    ax = axes[1]
    params = ("tau", "sigma", "mu")
    x = np.arange(len(params))
    bands = report["literature"]["bands"]
    for i, k in enumerate(params):
        lo, hi = bands[k]
        ax.add_patch(plt.Rectangle((i - 0.4, lo), 0.8, hi - lo, color="#dddddd", zorder=0))
    w = 0.3
    for j, key in enumerate(("adhd_combined", "adhd_inattentive")):
        d = report["effects_vs_neurotypical"][key]["d"]
        ax.bar(x + (j - 0.5) * w, [d[k] for k in params], width=w,
               color=colors[key], label=groups[key]["phenotype"], zorder=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\tau$ (tail)", r"$\sigma$ (spread)", r"$\mu$ (centre)"])
    ax.set_ylabel("Cohen's d vs neurotypical")
    ax.set_title("Effect sizes vs meta-analytic bands (grey)")
    ax.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
