"""Cross-subject EEG transformer — small enough for CPU training in 8h.

Architecture:
    Input:            (batch, channels=22, timepoints=751)
    Conv patch embed: (batch, seq_len, embed_dim) — small conv stack
                      compresses time and spatial dimensions to tokens
    + subject embed:  optional per-subject conditioning (learned per subject)
    Transformer:      2-4 layers, ~256 embed dim, 8 heads
    Pool + head:      Mean pool over tokens + linear classifier

Design principles:
    - Small enough to train on CPU in <30 min (~1M-3M params)
    - Cross-subject aware via optional subject embedding
    - Convolutional front-end respects local temporal + channel structure
    - Layer-scale + pre-norm for training stability on small data
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EEGTransformerConfig:
    """Configuration for the small EEG transformer."""
    # Input dimensions
    n_channels: int = 22
    n_timepoints: int = 751
    n_classes: int = 4
    n_subjects: int = 9

    # Patch embedding
    patch_time: int = 32       # timepoints per patch
    patch_stride: int = 16     # stride (overlapping patches for richer tokens)
    embed_dim: int = 128

    # Transformer
    n_layers: int = 3
    n_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.15

    # Subject conditioning
    use_subject_embed: bool = True
    subject_embed_dim: int = 32


# ---------------------------------------------------------------------------
# Patch embedding: (channels, time) → sequence of tokens
# ---------------------------------------------------------------------------

class EEGPatchEmbed(nn.Module):
    """Convolutional patch embedding for multi-channel EEG.

    First a spatial conv mixes across all channels (learned spatial filter),
    then a temporal conv creates patches. Similar to CTNet / EEGConformer designs
    but smaller and simpler.
    """

    def __init__(self, cfg: EEGTransformerConfig) -> None:
        super().__init__()
        # Spatial convolution: mix across channels (like a learned Laplacian)
        # Input:  (batch, 1, channels, time)  — treat channels as spatial axis
        # Output: (batch, spatial_out, 1, time)
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=cfg.embed_dim // 2,
            kernel_size=(cfg.n_channels, 1),
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(cfg.embed_dim // 2)

        # Temporal convolution: patch along the time dimension
        # Input:  (batch, spatial_out, 1, time)
        # Output: (batch, embed_dim, 1, n_patches)
        self.temporal_conv = nn.Conv2d(
            in_channels=cfg.embed_dim // 2,
            out_channels=cfg.embed_dim,
            kernel_size=(1, cfg.patch_time),
            stride=(1, cfg.patch_stride),
        )

        self.n_patches = (cfg.n_timepoints - cfg.patch_time) // cfg.patch_stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints)
        Returns:
            tokens: (batch, n_patches, embed_dim)
        """
        # Add channel-of-1 axis for Conv2d: (batch, 1, channels, time)
        x = x.unsqueeze(1)
        # Spatial conv → (batch, embed_dim/2, 1, time)
        x = self.spatial_bn(self.spatial_conv(x))
        x = F.gelu(x)
        # Temporal conv → (batch, embed_dim, 1, n_patches)
        x = self.temporal_conv(x)
        # (batch, embed_dim, 1, n_patches) → (batch, n_patches, embed_dim)
        x = x.squeeze(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# Sinusoidal + learnable positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_positional_encoding(n_positions: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal PE."""
    pos = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_positions, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# Transformer block with pre-norm + layer scale
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: EEGTransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        mlp_dim = int(cfg.embed_dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(mlp_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )
        # Layer scale — helps stability with small data
        self.ls1 = nn.Parameter(torch.full((cfg.embed_dim,), 1e-4))
        self.ls2 = nn.Parameter(torch.full((cfg.embed_dim,), 1e-4))

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        # need_weights=True returns averaged-over-heads weights by default.
        # We use average_attn_weights=False to get per-head weights.
        attn_out, attn_weights = self.attn(
            h, h, h,
            need_weights=return_attn,
            average_attn_weights=False if return_attn else True,
        )
        x = x + self.ls1 * attn_out
        x = x + self.ls2 * self.mlp(self.norm2(x))
        if return_attn:
            return x, attn_weights
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class EEGTransformer(nn.Module):
    """Small cross-subject EEG transformer for motor imagery classification.

    Notes:
        - Passing `subject_id` uses learned subject embeddings (per-subject bias).
          Set to None during zero-shot transfer to a held-out subject.
        - `return_attn_map=True` returns per-layer attention weights for the dashboard.
    """

    def __init__(self, cfg: EEGTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.patch_embed = EEGPatchEmbed(cfg)

        # Positional encoding — sinusoidal, no learned param
        pe = sinusoidal_positional_encoding(self.patch_embed.n_patches, cfg.embed_dim)
        self.register_buffer("pos_embed", pe.unsqueeze(0))

        # CLS token — pooled representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Subject embedding — added as bias to CLS token
        if cfg.use_subject_embed:
            self.subject_embed = nn.Embedding(
                num_embeddings=cfg.n_subjects + 1,  # +1 for "unknown" during zero-shot
                embedding_dim=cfg.embed_dim,
            )
            nn.init.trunc_normal_(self.subject_embed.weight, std=0.02)
        else:
            self.subject_embed = None

        # Transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        # Classifier head
        self.head = nn.Linear(cfg.embed_dim, cfg.n_classes)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | None = None,
        return_features: bool = False,
        return_attention: bool = False,
        return_layer_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints)
            subject_id: (batch,) int subject IDs, or None for zero-shot transfer
            return_features: if True, include CLS features in the return
            return_attention: if True, include list of per-layer attention weights
                              each element: (batch, n_heads, n_tokens, n_tokens)
            return_layer_features: if True, include list of per-layer CLS features
                                    each element: (batch, embed_dim)

        Returns:
            logits, or a tuple (logits, cls_feat, attentions, layer_feats)
            with any of the optional pieces included depending on the flags.
        """
        b = x.shape[0]

        # Patch embed → (batch, n_patches, embed_dim)
        tokens = self.patch_embed(x)
        # + positional encoding
        tokens = tokens + self.pos_embed

        # Prepend CLS token
        cls_tok = self.cls_token.expand(b, -1, -1)
        # Add subject-specific bias to CLS
        if self.subject_embed is not None and subject_id is not None:
            subj_bias = self.subject_embed(subject_id).unsqueeze(1)
            cls_tok = cls_tok + subj_bias

        tokens = torch.cat([cls_tok, tokens], dim=1)

        attentions = []
        layer_cls_feats = []

        # Transformer stack
        for block in self.blocks:
            if return_attention:
                tokens, attn_w = block(tokens, return_attn=True)
                attentions.append(attn_w)
            else:
                tokens = block(tokens)
            if return_layer_features:
                # Apply the final norm even for intermediate features so they're
                # in the same space as the classifier expects
                layer_cls_feats.append(self.norm(tokens)[:, 0])
        tokens = self.norm(tokens)

        # Use CLS token as classification feature
        cls_feat = tokens[:, 0]
        logits = self.head(cls_feat)

        # Assemble return
        if not (return_features or return_attention or return_layer_features):
            return logits

        extras = [logits]
        if return_features:
            extras.append(cls_feat)
        if return_attention:
            extras.append(attentions)
        if return_layer_features:
            extras.append(layer_cls_feats)
        return tuple(extras)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    cfg = EEGTransformerConfig()
    model = EEGTransformer(cfg)
    print(f"Model params: {model.n_parameters():,}")
    print(f"n_patches: {model.patch_embed.n_patches}")

    x = torch.randn(4, cfg.n_channels, cfg.n_timepoints)
    sid = torch.tensor([0, 1, 2, 3])

    with torch.no_grad():
        logits = model(x, subject_id=sid)
        print(f"logits: {logits.shape}")  # (4, 4)

        # Zero-shot: no subject_id
        logits_zeroshot = model(x, subject_id=None)
        print(f"zero-shot logits: {logits_zeroshot.shape}")
