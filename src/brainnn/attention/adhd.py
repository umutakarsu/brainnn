"""ADHDAttention — multi-head attention with two-axis neuromodulation.

Architectural note (v0.2, Dayan-correction):
    Two distinct neuromodulatory axes act on attention, NOT one:

      1. Precision axis (ACh / NE)  →  controls noise floor on attention scores
                                       via `GaussianNoiseInjector`. High precision
                                       = crisp attention, low precision = scattered.

      2. Value axis (DA)            →  controls which heads are amplified vs.
                                       suppressed via `DopaminergicModulator`.
                                       High DA = strong cost-benefit gating
                                       (hyperfocus); low DA = weak gating
                                       (distractibility, "why bother").

    Earlier versions collapsed both axes onto dopamine, which doesn't match
    the current neuromodulator literature (Yu & Dayan 2005; Westbrook & Braver).

    The pharmacological implication of this split is that the model can now
    in principle be dissociated by drug class:
        - Cholinergic interventions (donepezil)   → affect precision noise
        - Noradrenergic interventions (atomoxetine) → affect arousal / NE
        - Dopaminergic interventions (methylphenidate, amphetamine) → affect value gain
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from brainnn.core.config import ADHDConfig
from brainnn.core.state import BrainState
from brainnn.attention.noise import GaussianNoiseInjector
from brainnn.attention.modulation import DopaminergicModulator


class ADHDAttention(nn.Module):
    """Multi-head attention modified to simulate ADHD attention patterns.

    Two-axis neuromodulation (per Yu & Dayan 2005 framework):

    PRECISION AXIS (ACh / NE) — applied to attention SCORES (pre-softmax):
        1. Cholinergic/noradrenergic precision controls noise on attention scores.
           Low precision = scattered focus; high precision = crisp attention.

    VALUE AXIS (DA) — applied to attention OUTPUTS (per-head, post-softmax):
        2. Dopaminergic value gating controls which heads are amplified.
           Low DA = weak gating + head dropout (distractibility).
           High DA = strong gating, lock onto high-value heads (hyperfocus).

    Pharmacologically dissociable: ACh drugs → precision; NE drugs → arousal/precision;
    DA drugs → value gain.

    Input:  (batch, seq_len, embed_dim) — can also work with (batch, embed_dim)
    Output: (batch, seq_len, embed_dim), attention_weights
    """

    def __init__(self, embed_dim: int, config: ADHDConfig) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = embed_dim // config.num_heads
        assert embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads"

        # Standard Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Two-axis neuromodulation:
        #   noise_injector      → ACh/NE-driven precision (acts on scores)
        #   dopamine_modulator  → DA-driven value gating (acts on per-head outputs)
        self.noise_injector = GaussianNoiseInjector(max_noise_std=config.max_noise_std)
        self.dopamine_modulator = DopaminergicModulator(config)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        brain_state: BrainState,
        key_value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with ADHD-modified attention.

        Args:
            x: (batch, seq_len, embed_dim) or (batch, embed_dim) query input
            brain_state: current BrainState for SNR and dopamine modulation
            key_value: optional (batch, kv_len, embed_dim) for cross-attention

        Returns:
            (output, attention_weights)
        """
        # Handle 2D input: add sequence dimension
        squeezed = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeezed = True

        if key_value is None:
            key_value = x
        elif key_value.ndim == 2:
            key_value = key_value.unsqueeze(1)

        batch, seq_len, _ = x.shape
        kv_len = key_value.shape[1]

        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # PRECISION AXIS (ACh / NE): inject noise on scores based on cholinergic precision.
        # `brain_state.precision` (aliased as .snr) is computed from acetylcholine and
        # norepinephrine — NOT dopamine. See `core/state.py` for the formula.
        scores = self.noise_injector(scores, brain_state.precision)

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Attend to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # VALUE AXIS (DA): cost-benefit gating over per-head outputs.
        # Modulator reads `brain_state.value_gate` (dopamine-derived) internally.
        attn_output = self.dopamine_modulator(attn_output, brain_state)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        if squeezed:
            output = output.squeeze(1)
            attn_weights = attn_weights.squeeze(2)

        return output, attn_weights
