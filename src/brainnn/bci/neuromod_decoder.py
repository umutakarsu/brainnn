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

def infer_neuromod_from_signal(x: torch.Tensor) -> BrainState:
    """Heuristic estimator of neuromodulator state from a single EEG epoch.

    Uses spectral features as proxies (rough, prototype-level):
        - Alpha (8-12 Hz) power over posterior channels → inverse arousal → NE proxy
        - Global signal variance across trial → uncertainty → ACh proxy
        - Later-trial-index (fatigue-like) → DA suppression proxy

    This is a placeholder to demonstrate the interface — a real implementation
    would use validated markers (pupil, HRV, HEP, EEG alpha over parietal cortex).

    Args:
        x: (channels, timepoints) — a single EEG epoch, sample-rate ~250 Hz assumed
    Returns:
        BrainState populated with heuristic estimates
    """
    with torch.no_grad():
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Global signal energy — high variance = high uncertainty (low ACh)
        signal_var = float(x.var().item())
        # Normalize into [0, 1]. Empirical scaling on BNCI normalized data.
        acetylcholine = float(np.clip(1.0 - signal_var / 2.0, 0.1, 0.9))

        # Alpha power estimation via FFT on posterior channels
        # (Take last 8 channels as proxy for posterior; real code would use montage)
        fft = torch.fft.rfft(x[-8:], dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1/250.0)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_power = float(fft[..., alpha_mask].abs().mean().item())
        # High alpha = relaxed = low arousal = low NE
        norepinephrine = float(np.clip(1.0 - alpha_power / 3.0, 0.2, 0.8))

        # Dopamine: default mid; would be modulated by task engagement in real use
        dopamine = 0.5

    return BrainState(
        acetylcholine=acetylcholine,
        norepinephrine=norepinephrine,
        dopamine=dopamine,
    )


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
