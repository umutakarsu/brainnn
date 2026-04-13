"""Demo: ADHD attention simulation — observe dopamine dynamics and focus shifts."""

import torch
from brainnn.core.config import ADHDConfig
from brainnn.core.state import BrainState
from brainnn.attention.adhd import ADHDAttention


def main():
    config = ADHDConfig(
        num_heads=8,
        baseline_dopamine=0.3,
        dopamine_decay=0.05,
        hyperfocus_threshold=0.8,
        max_noise_std=0.5,
    )

    attn = ADHDAttention(embed_dim=128, config=config)
    attn.train()

    state = BrainState(dopamine=config.baseline_dopamine)

    print("=== ADHD Attention Simulation ===\n")
    print(f"{'Step':>4} | {'Dopamine':>8} | {'Focus':>5} | {'SNR':>5} | {'Hyperfocus':>10} | {'Attn Entropy':>12}")
    print("-" * 65)

    x = torch.randn(1, 4, 128)  # 4 stimuli

    for step in range(30):
        # Simulate occasional interesting stimuli
        reward = 0.0
        if step == 10:
            reward = 0.7  # Very interesting stimulus!
        elif step == 20:
            reward = 0.3  # Mildly interesting

        state.update(dt=1.0, dopamine_decay=config.dopamine_decay, fatigue_rate=0.01)
        if reward > 0:
            state.receive_stimulus(reward)

        with torch.no_grad():
            out, weights = attn(x, state)

        # Calculate attention entropy (uniformity)
        entropy = -(weights * (weights + 1e-8).log()).sum(-1).mean().item()

        print(f"{step:>4} | {state.dopamine:>8.3f} | {state.focus:>5.3f} | {state.snr:>5.3f} | "
              f"{'  YES  ' if state.is_hyperfocused else '   no  ':>10} | {entropy:>12.3f}")

    print("\n=== Simulation Complete ===")
    print(f"Final fatigue: {state.fatigue:.3f}")
    print(f"Total steps: {state.step}")


if __name__ == "__main__":
    main()
