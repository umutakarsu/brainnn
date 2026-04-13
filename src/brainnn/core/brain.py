"""NeuroDivergentBrainSimulator — the unified brain that ties everything together."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from brainnn.core.config import BrainConfig, ModalityType
from brainnn.core.state import BrainState
from brainnn.synesthesia.network import SynestheticNet
from brainnn.attention.adhd import ADHDAttention
from brainnn.generation.cross_modal_gen import CrossModalGenerator


@dataclass
class BrainOutput:
    """Output from a single simulation step."""

    # Fused latent representation from all active modalities
    fused_latent: torch.Tensor
    # Per-modality encoder outputs
    modality_latents: dict[ModalityType, torch.Tensor]
    # Attention-gated latent (after ADHD filtering)
    attended_latent: torch.Tensor
    # Attention weights for visualization
    attention_weights: torch.Tensor
    # Generated cross-modal outputs (if requested)
    generated: dict[ModalityType, torch.Tensor] = field(default_factory=dict)
    # Snapshot of brain state at this step
    brain_state_snapshot: dict[str, float] = field(default_factory=dict)


class NeuroDivergentBrainSimulator(nn.Module):
    """The unified neurodivergent brain simulator.

    Combines:
    - SynestheticNet for cross-modal processing with lateral connections
    - ADHDAttention for dopamine-modulated attention
    - CrossModalGenerator for generative outputs

    Simulation flow:
        Input → ADHDAttention Gate → SynestheticNet → Generator → Output
                      ↑                                              │
                  BrainState ←──── update ←─────────────────────────┘
    """

    def __init__(self, config: BrainConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = BrainConfig.default_three_modality()
        self.config = config

        # Core components
        self.synesthetic_net = SynestheticNet(config)
        self.adhd_attention = ADHDAttention(
            embed_dim=config.latent_dim,
            config=config.adhd,
        )
        self.generator = CrossModalGenerator(config)

        # Brain state (not a nn parameter — simulation state)
        self.state = BrainState(dopamine=config.adhd.baseline_dopamine)

        # Pre-attention projection: raw input → embed_dim for attention
        self.input_projections = nn.ModuleDict()
        for mod_type, mod_config in config.modalities.items():
            # Lazy linear to handle variable input sizes
            self.input_projections[mod_type.value] = nn.LazyLinear(config.latent_dim)

    def reset(self) -> None:
        """Reset brain state to initial conditions."""
        self.state.reset(baseline_dopamine=self.config.adhd.baseline_dopamine)

    def step(
        self,
        inputs: dict[ModalityType, torch.Tensor],
        reward_signal: float = 0.0,
        generate: bool = True,
    ) -> BrainOutput:
        """Run a single simulation step.

        Args:
            inputs: {modality_type: raw_input_tensor} for active modalities
            reward_signal: external reward that boosts dopamine (e.g. interesting stimulus)
            generate: whether to produce cross-modal generative outputs

        Returns:
            BrainOutput with all results and state snapshot
        """
        # 1. Update brain state
        self.state.update(
            dt=self.config.dt,
            dopamine_decay=self.config.adhd.dopamine_decay,
            fatigue_rate=self.config.adhd.fatigue_rate,
        )
        if reward_signal > 0:
            self.state.receive_stimulus(reward_signal)

        # 2. ADHD Attention gate — filter/weight inputs before synesthetic processing
        # Project each input to latent space for attention
        projected_inputs = {}
        for mod, tensor in inputs.items():
            flat = tensor.flatten(1)
            proj = self.input_projections[mod.value](flat)
            projected_inputs[mod] = proj

        # Stack projected inputs as a sequence for attention
        mod_order = list(projected_inputs.keys())
        stacked = torch.stack([projected_inputs[m] for m in mod_order], dim=1)  # (B, N_mod, latent)

        attended, attn_weights = self.adhd_attention(stacked, self.state)

        # Unstack back to per-modality, apply attention weighting to raw inputs
        attended_inputs = {}
        for i, mod in enumerate(mod_order):
            # Use attention as a soft gate on the original inputs
            gate = attended[:, i:i+1, :]  # (B, 1, latent)
            # We pass original inputs to SynestheticNet (attention gates the output, not input)
            attended_inputs[mod] = inputs[mod]

        # 3. SynestheticNet — cross-modal processing with lateral connections
        fused_latent, modality_latents = self.synesthetic_net(attended_inputs)

        # Apply attention gating to the fused representation
        attended_latent = attended.mean(dim=1)  # (B, latent)
        gated_latent = fused_latent * torch.sigmoid(attended_latent)

        # 4. Generate cross-modal outputs if requested
        generated = {}
        if generate:
            generated = self.generator.generate_all(gated_latent)

        # 5. Build output
        return BrainOutput(
            fused_latent=fused_latent,
            modality_latents=modality_latents,
            attended_latent=gated_latent,
            attention_weights=attn_weights,
            generated=generated,
            brain_state_snapshot={
                "arousal": self.state.arousal,
                "focus": self.state.focus,
                "fatigue": self.state.fatigue,
                "dopamine": self.state.dopamine,
                "snr": self.state.snr,
                "is_hyperfocused": float(self.state.is_hyperfocused),
                "step": self.state.step,
            },
        )

    def simulate(
        self,
        inputs_sequence: list[dict[ModalityType, torch.Tensor]],
        reward_signals: list[float] | None = None,
        generate: bool = True,
    ) -> list[BrainOutput]:
        """Run a full simulation over a sequence of inputs.

        Args:
            inputs_sequence: list of input dicts, one per timestep
            reward_signals: optional reward per step (len must match inputs)
            generate: whether to produce cross-modal outputs

        Returns:
            List of BrainOutput per timestep
        """
        self.reset()

        if reward_signals is None:
            reward_signals = [0.0] * len(inputs_sequence)

        outputs = []
        for inputs, reward in zip(inputs_sequence, reward_signals):
            output = self.step(inputs, reward_signal=reward, generate=generate)
            outputs.append(output)

        return outputs

    def forward(
        self,
        inputs: dict[ModalityType, torch.Tensor],
        reward_signal: float = 0.0,
    ) -> BrainOutput:
        """Alias for step() — nn.Module compatibility."""
        return self.step(inputs, reward_signal=reward_signal)
