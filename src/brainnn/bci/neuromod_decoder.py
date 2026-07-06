"""Neuromodulator-conditioned decoder wrapper.

Extends the base EEG transformer with FiLM-style conditioning on the
Yu & Dayan (2005) neuromodulator triple:
    - acetylcholine (ACh) → expected uncertainty / precision
    - norepinephrine (NE) → unexpected uncertainty / arousal
    - dopamine (DA)       → value-of-attention gate

Motivation (following Peter Dayan's feedback):
    A brain-to-text decoder's optimal behavior depends on the user's current
    neuromodulator state. A vigilant user (high ACh, mid NE) has crisp neural
    signals — the decoder should trust them more. A tired user (low ACh,
    fatigue-elevated DA suppression) has noisy signals — the decoder should
    default to language-model priors more heavily.

    This module implements that intuition as FiLM (Feature-wise Linear
    Modulation): the CLS token features from the transformer are scaled and
    shifted based on the neuromodulator triple before classification.

Architecture:
    (ACh, NE, DA) → 3-dim conditioning vector
                   ↓
                   MLP(3 → 64 → 2 × embed_dim)  [gamma, beta]
                   ↓
    features_out = features * (1 + gamma) + beta
                   ↓
                   Classifier head → logits

At training: neuromodulator state is treated as an auxiliary input.
At inference: dashboard can toggle values to explore "what would the decoder
do if the user were tired vs. focused?"

Falsifiability: pharmacological manipulations should produce distinct
signatures — donepezil ↑ ACh should improve precision-limited decoding;
atomoxetine ↑ NE should improve novelty detection; methylphenidate ↑ DA
should improve value-gated confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brainnn.bci.models import EEGTransformer, EEGTransformerConfig
from brainnn.core.state import BrainState


@dataclass
class NeuromodConditioningConfig:
    """Configuration for the neuromodulator conditioning module."""
    # Base features from the EEG transformer
    feature_dim: int = 128
    # Hidden width of the FiLM generator MLP
    hidden_dim: int = 64
    # Neuromodulator input dim: [ACh, NE, DA]
    neuromod_dim: int = 3
    # Whether to also condition on precision + value_gate derived quantities
    # (adds 2 extra input dims → 5-dim conditioning vector)
    use_derived_features: bool = True
    # LR for FiLM head (usually smaller than base to avoid overriding base features)
    film_init_scale: float = 0.01


class FiLMGenerator(nn.Module):
    """MLP that maps neuromodulator state → (gamma, beta) FiLM parameters.

    The FiLM formulation (Perez et al., 2018) is: y = x * (1 + gamma) + beta
    We use (1 + gamma) so the initial state (gamma near 0) is a near-identity
    map, letting the base model dominate at initialization.
    """

    def __init__(self, cfg: NeuromodConditioningConfig) -> None:
        super().__init__()
        input_dim = cfg.neuromod_dim + (2 if cfg.use_derived_features else 0)
        self.use_derived = cfg.use_derived_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.feature_dim),
        )
        # Zero-init the final layer so FiLM starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.feature_dim = cfg.feature_dim

    def forward(self, neuromod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            neuromod: (batch, 3) or (batch, 5) — [ACh, NE, DA] (+ precision, value_gate)
        Returns:
            (gamma, beta): each (batch, feature_dim)
        """
        params = self.net(neuromod)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta


# ---------------------------------------------------------------------------
# Full conditioned decoder
# ---------------------------------------------------------------------------

class NeuromodConditionedDecoder(nn.Module):
    """Wraps an EEGTransformer with FiLM conditioning on ACh / NE / DA.

    Two ways to use:
      1. Pass explicit `neuromod` tensor (batch, 3) or (batch, 5)
      2. Pass a `BrainState` (single state applied to whole batch) — convenience
         for dashboard exploration and pharmacological ablation.
    """

    def __init__(
        self,
        base_model: EEGTransformer,
        cfg: NeuromodConditioningConfig | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        if cfg is None:
            cfg = NeuromodConditioningConfig(
                feature_dim=base_model.cfg.embed_dim,
            )
        self.cfg = cfg
        self.film = FiLMGenerator(cfg)

        # Fresh classifier head to be trained with FiLM
        self.head = nn.Linear(cfg.feature_dim, base_model.cfg.n_classes)
        # Initialize as copy of base model's head (so first pass matches base model)
        with torch.no_grad():
            self.head.weight.copy_(base_model.head.weight)
            self.head.bias.copy_(base_model.head.bias)

    def _neuromod_from_state(self, state: BrainState, batch: int, device) -> torch.Tensor:
        """Convert a single BrainState to a (batch, 3 or 5) tensor."""
        vals = [state.acetylcholine, state.norepinephrine, state.dopamine]
        if self.cfg.use_derived_features:
            vals.extend([state.precision, state.value_gate])
        t = torch.tensor(vals, dtype=torch.float32, device=device)
        return t.unsqueeze(0).expand(batch, -1)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | None = None,
        neuromod: torch.Tensor | BrainState | None = None,
        return_gamma_beta: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints)
            subject_id: optional (batch,) subject IDs for base model
            neuromod: either
                - (batch, 3) or (batch, 5) tensor  (per-sample)
                - a BrainState object (broadcast to whole batch)
                - None → defaults to neutral state [0.5, 0.5, 0.5]
        Returns:
            logits (batch, n_classes), and optionally (gamma, beta)
        """
        b = x.shape[0]
        device = x.device

        # Get CLS features from base model (frozen or fine-tuned)
        _, cls_feat = self.base_model(x, subject_id=subject_id, return_features=True)

        # Get neuromod vector
        if isinstance(neuromod, BrainState):
            nm = self._neuromod_from_state(neuromod, b, device)
        elif neuromod is None:
            # Neutral: 0.5 for all
            n_dim = 5 if self.cfg.use_derived_features else 3
            nm = torch.full((b, n_dim), 0.5, device=device)
        else:
            nm = neuromod

        # FiLM
        gamma, beta = self.film(nm)
        conditioned = cls_feat * (1.0 + gamma) + beta

        logits = self.head(conditioned)

        if return_gamma_beta:
            return logits, gamma, beta
        return logits

    def freeze_base_model(self) -> None:
        """Freeze the base transformer weights — useful for training only the FiLM head."""
        for p in self.base_model.parameters():
            p.requires_grad = False

    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Behavioral markers → neuromodulator predictor (heuristic v0)
# ---------------------------------------------------------------------------
#
# HONESTY NOTE — evidence hierarchy per marker (updated after Jul 2026
# literature synthesis; see `neuromod_bci_synthesis.md` at repo root):
#
#   NE : Pupillometry is the ONLY validated real-time neuromodulator marker
#        [Joshi, Li, Kalwani & Gold 2016, Neuron 89(1):221; Reimer et al.
#        2014, 2016]. EEG alpha power is a distal proxy — related to arousal
#        but NOT specifically to LC-NE firing. When pupillometry is available
#        it should REPLACE the alpha-based proxy below.
#
#   ACh: NO standalone validated real-time EEG-only marker exists. Pupillometry
#        conflates NE and ACh contributions [Reimer et al. 2016]. Our
#        signal-variance heuristic below is a placeholder rooted in "expected
#        uncertainty" intuition [Yu & Dayan 2005] but is NOT empirically
#        validated as an ACh estimator. Treat all ACh values below as
#        exploratory, not measurements.
#
#   DA : No validated real-time DA estimator exists at all [Berke 2018, Nat
#        Neurosci 21(6):787]. Correlational RT-variability associations do
#        not translate to closed-loop use. Our default of 0.5 is not an
#        inference — it is honestly a placeholder awaiting a marker.
#
# In any publication, this hierarchy MUST be surfaced. Do not present ACh/DA
# as "measured"; only as "conditioned-upon".
# ---------------------------------------------------------------------------

def infer_neuromod_from_signal(x: torch.Tensor) -> BrainState:
    """Prototype-only neuromodulator estimator from a single EEG epoch.

    See HONESTY NOTE above. This function returns:
        - a NE proxy (alpha-power based; distal, not a validated marker)
        - an ACh proxy (signal-variance based; theoretical, not validated)
        - a DA default of 0.5 (no validated closed-loop marker exists)

    For any real experimental deployment:
        - Replace the NE proxy with pupillometry
          [Joshi et al. 2016, DOI:10.1016/j.neuron.2015.11.028]
        - Treat ACh conditioning as ablation-testable but not measured
        - Treat DA conditioning as a knob to test pharmacologically
          (L-DOPA / haloperidol / methylphenidate), not to infer from signal

    Args:
        x: (channels, timepoints) EEG epoch, sample rate ~250 Hz.
    Returns:
        BrainState with the fields above; treat each per the hierarchy in
        the module-level HONESTY NOTE.
    """
    with torch.no_grad():
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # ACh proxy (PLACEHOLDER — not validated as an ACh estimator)
        # Rationale: expected uncertainty → precision → ACh; higher signal
        # variance ≈ noisier ≈ lower expected precision.
        signal_var = float(x.var().item())
        acetylcholine = float(np.clip(1.0 - signal_var / 2.0, 0.1, 0.9))

        # NE proxy (DISTAL — alpha power is arousal-related but not
        # specifically LC-NE; pupillometry is the validated marker)
        fft = torch.fft.rfft(x[-8:], dim=-1)  # posterior-ish channels
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1/250.0)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_power = float(fft[..., alpha_mask].abs().mean().item())
        norepinephrine = float(np.clip(1.0 - alpha_power / 3.0, 0.2, 0.8))

        # DA (NOT INFERRED — no validated real-time marker)
        # Default = 0.5. In experiments, set this via pharmacological arm
        # or task-engagement design, not derived from EEG.
        dopamine = 0.5

    return BrainState(
        acetylcholine=acetylcholine,
        norepinephrine=norepinephrine,
        dopamine=dopamine,
    )


# ---------------------------------------------------------------------------
# Shuffle-control ablation (falsifiability test)
# ---------------------------------------------------------------------------
#
# Following Frank, Seeberger & O'Reilly (2004, Science 306:1940), a genuine
# conditioning effect must be dissociable from added-capacity artifact. If we
# shuffle the neuromodulator vector across trials — breaking the trial <->
# state correlation — a truly conditioning-based model should DEGRADE, while
# a model that is merely using the extra parameters as noise-regularization
# should show UNCHANGED performance.
#
# This is the primary in-silico falsifiability test for the paper.
# ---------------------------------------------------------------------------

def shuffled_neuromod(neuromod: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Shuffle a batch of neuromodulator vectors across the batch dimension.

    Args:
        neuromod: (batch, dim) — per-sample neuromodulator states.
        seed: reproducibility seed.
    Returns:
        Shuffled tensor, same shape, same distribution, but the trial <->
        state pairing is broken.
    """
    assert neuromod.ndim == 2, "Expected (batch, dim)"
    g = torch.Generator(device=neuromod.device).manual_seed(seed)
    perm = torch.randperm(neuromod.shape[0], generator=g, device=neuromod.device)
    return neuromod[perm]


def ablation_summary(
    decoder: "NeuromodConditionedDecoder",
    x: torch.Tensor,
    y: torch.Tensor,
    neuromod: torch.Tensor,
    subject_id: torch.Tensor | None = None,
    n_shuffle_seeds: int = 5,
) -> dict:
    """Run the shuffle-control ablation on a fitted decoder.

    Compares:
        - true:      (trial, state) pairing preserved
        - shuffled:  state randomly re-paired to trials (n_shuffle_seeds averages)
        - no-cond:   FiLM turned off (γ=0, β=0)

    Args:
        decoder: a NeuromodConditionedDecoder (typically fitted).
        x: (batch, channels, timepoints)
        y: (batch,) ground-truth labels
        neuromod: (batch, dim) per-sample true neuromodulator states
        subject_id: optional subject IDs for the base model
        n_shuffle_seeds: number of shuffle repetitions to average over

    Returns:
        dict with 'true_acc', 'shuffled_acc_mean', 'shuffled_acc_std', 'nocond_acc'.

    Interpretation:
        - If true_acc >> shuffled_acc: conditioning genuinely helps → paper stands.
        - If true_acc ≈ shuffled_acc: FiLM only added capacity, not information.
    """
    decoder.eval()
    with torch.no_grad():
        # True conditioning
        logits_true = decoder(x, subject_id=subject_id, neuromod=neuromod)
        true_acc = float((logits_true.argmax(-1) == y).float().mean().item())

        # Shuffled — break state <-> trial pairing
        shuffled_accs = []
        for s in range(n_shuffle_seeds):
            nm_shuf = shuffled_neuromod(neuromod, seed=s)
            logits_s = decoder(x, subject_id=subject_id, neuromod=nm_shuf)
            shuffled_accs.append(
                float((logits_s.argmax(-1) == y).float().mean().item())
            )
        shuf_mean = float(np.mean(shuffled_accs))
        shuf_std = float(np.std(shuffled_accs))

        # No conditioning at all — use base model's classifier directly
        logits_nocond = decoder.base_model(x, subject_id=subject_id)
        nocond_acc = float((logits_nocond.argmax(-1) == y).float().mean().item())

    return {
        "true_acc": true_acc,
        "shuffled_acc_mean": shuf_mean,
        "shuffled_acc_std": shuf_std,
        "nocond_acc": nocond_acc,
        "conditioning_effect": true_acc - shuf_mean,
    }


if __name__ == "__main__":
    # Smoke test
    base_cfg = EEGTransformerConfig(n_subjects=8)
    base_model = EEGTransformer(base_cfg)

    decoder = NeuromodConditionedDecoder(base_model)
    print(f"Trainable params: {decoder.n_trainable_parameters():,}")

    x = torch.randn(4, 22, 751)

    # Test 1: no conditioning (neutral)
    logits = decoder(x)
    print(f"Neutral logits shape: {logits.shape}")

    # Test 2: BrainState conditioning
    state = BrainState(acetylcholine=0.9, norepinephrine=0.5, dopamine=0.7)
    logits_focused = decoder(x, neuromod=state)
    print(f"Focused state logits shape: {logits_focused.shape}")

    # Test 3: heuristic estimator on a synthetic signal
    inferred = infer_neuromod_from_signal(x[0])
    print(f"Inferred state from signal: ACh={inferred.acetylcholine:.2f}, "
          f"NE={inferred.norepinephrine:.2f}, DA={inferred.dopamine:.2f}")

    # Test 4: gamma/beta return
    logits, gamma, beta = decoder(x, neuromod=state, return_gamma_beta=True)
    print(f"gamma range: [{gamma.min():.4f}, {gamma.max():.4f}]")
    print(f"beta range:  [{beta.min():.4f}, {beta.max():.4f}]")

    # Test 5: freeze base and count trainable
    decoder.freeze_base_model()
    print(f"After freeze — trainable: {decoder.n_trainable_parameters():,}")
