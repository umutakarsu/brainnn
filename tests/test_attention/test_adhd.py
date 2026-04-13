"""Tests for ADHD attention components."""

import torch
import pytest

from brainnn.core.config import ADHDConfig
from brainnn.core.state import BrainState
from brainnn.attention.noise import GaussianNoiseInjector
from brainnn.attention.modulation import DopaminergicModulator
from brainnn.attention.adhd import ADHDAttention


BATCH = 2
EMBED_DIM = 64
SEQ_LEN = 4
NUM_HEADS = 8


class TestGaussianNoiseInjector:
    def test_no_noise_at_high_snr(self):
        inj = GaussianNoiseInjector(max_noise_std=0.5)
        scores = torch.ones(BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        out = inj(scores, snr=1.0)
        assert torch.allclose(out, scores)

    def test_noise_added_at_low_snr(self):
        inj = GaussianNoiseInjector(max_noise_std=0.5)
        inj.train()
        scores = torch.ones(BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        out = inj(scores, snr=0.0)
        assert not torch.allclose(out, scores)

    def test_output_shape_preserved(self):
        inj = GaussianNoiseInjector()
        scores = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        out = inj(scores, snr=0.5)
        assert out.shape == scores.shape


class TestDopaminergicModulator:
    def test_output_shape(self):
        config = ADHDConfig(num_heads=NUM_HEADS)
        mod = DopaminergicModulator(config)
        attn_out = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, EMBED_DIM // NUM_HEADS)
        state = BrainState(dopamine=0.5, focus=0.5)
        result = mod(attn_out, state)
        assert result.shape == attn_out.shape

    def test_hyperfocus_amplifies(self):
        config = ADHDConfig(num_heads=NUM_HEADS, hyperfocus_threshold=0.8)
        mod = DopaminergicModulator(config)
        attn_out = torch.ones(BATCH, NUM_HEADS, SEQ_LEN, EMBED_DIM // NUM_HEADS)
        state = BrainState(dopamine=0.95, focus=0.9)
        result = mod(attn_out, state)
        # At least some heads should be amplified (> 1)
        assert result.max().item() > 1.0


class TestADHDAttention:
    def test_3d_input(self):
        config = ADHDConfig(num_heads=NUM_HEADS)
        attn = ADHDAttention(EMBED_DIM, config)
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        state = BrainState(dopamine=0.5)
        out, weights = attn(x, state)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_2d_input(self):
        config = ADHDConfig(num_heads=NUM_HEADS)
        attn = ADHDAttention(EMBED_DIM, config)
        x = torch.randn(BATCH, EMBED_DIM)
        state = BrainState(dopamine=0.5)
        out, weights = attn(x, state)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_gradient_flow(self):
        config = ADHDConfig(num_heads=NUM_HEADS)
        attn = ADHDAttention(EMBED_DIM, config)
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, requires_grad=True)
        state = BrainState(dopamine=0.5)
        out, _ = attn(x, state)
        out.sum().backward()
        assert x.grad is not None

    def test_cross_attention(self):
        config = ADHDConfig(num_heads=NUM_HEADS)
        attn = ADHDAttention(EMBED_DIM, config)
        q = torch.randn(BATCH, 3, EMBED_DIM)
        kv = torch.randn(BATCH, 5, EMBED_DIM)
        state = BrainState(dopamine=0.5)
        out, weights = attn(q, state, key_value=kv)
        assert out.shape == (BATCH, 3, EMBED_DIM)
