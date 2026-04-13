"""ADHDAttention — multi-head attention with noise injection and dopaminergic modulation."""

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

    Key differences from standard attention:
    1. Noise injection into attention scores (low SNR = scattered focus)
    2. Dopaminergic gating of attention heads (some heads go dark)
    3. Stochastic head dropout (unpredictable attention shifts)
    4. Hyperfocus mode (dopamine spike → locked-on attention)

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

        # ADHD-specific components
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

        # ADHD modification 1: Inject noise based on SNR
        scores = self.noise_injector(scores, brain_state.snr)

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Attend to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # ADHD modification 2: Dopaminergic modulation of heads
        attn_output = self.dopamine_modulator(attn_output, brain_state)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        if squeezed:
            output = output.squeeze(1)
            attn_weights = attn_weights.squeeze(2)

        return output, attn_weights
