"""Demo: Full neurodivergent brain simulation with visualization."""

import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from brainnn.core.config import BrainConfig, ModalityType
from brainnn.core.brain import NeuroDivergentBrainSimulator
from brainnn.viz.dashboard import plot_brain_state_timeline, plot_simulation_dashboard
from brainnn.viz.connectivity import plot_connectivity_matrix


def main():
    # Configure a brain with strong synesthesia and moderate ADHD
    config = BrainConfig.default_three_modality()
    config.synesthesia.synesthesia_strength = 0.7
    config.adhd.baseline_dopamine = 0.35
    config.adhd.dopamine_decay = 0.03
    config.adhd.max_noise_std = 0.4

    brain = NeuroDivergentBrainSimulator(config)
    print(f"Brain created with {len(config.modalities)} modalities")
    print(f"Total parameters: {sum(p.numel() for p in brain.parameters()):,}")

    # Create a stimulation sequence with reward spikes
    num_steps = 40
    inputs_seq = []
    rewards = [0.0] * num_steps

    # Reward schedule: interesting events at certain steps
    rewards[5] = 0.6    # Moderately interesting
    rewards[15] = 0.9   # Very interesting → should trigger hyperfocus
    rewards[25] = 0.2   # Mildly interesting
    rewards[35] = 0.7   # Interesting again

    for _ in range(num_steps):
        inputs_seq.append({
            ModalityType.VISUAL: torch.randn(1, 3, 32, 32),
            ModalityType.AUDITORY: torch.randn(1, 1, 1024),
            ModalityType.TACTILE: torch.randn(1, 1, 16, 16),
        })

    # Run simulation
    print(f"\nRunning {num_steps}-step simulation...")
    with torch.no_grad():
        outputs = brain.simulate(inputs_seq, reward_signals=rewards, generate=True)

    # Print step-by-step summary
    print(f"\n{'Step':>4} | {'Dopamine':>8} | {'Focus':>5} | {'Fatigue':>7} | {'SNR':>5} | {'Hyperfocus':>10}")
    print("-" * 60)
    for i, out in enumerate(outputs):
        s = out.brain_state_snapshot
        marker = " ← reward!" if rewards[i] > 0 else ""
        print(f"{int(s['step']):>4} | {s['dopamine']:>8.3f} | {s['focus']:>5.3f} | "
              f"{s['fatigue']:>7.3f} | {s['snr']:>5.3f} | "
              f"{'  YES  ' if s['is_hyperfocused'] > 0.5 else '   no  '}{marker}")

    # Save visualizations
    print("\nGenerating visualizations...")

    fig1 = plot_brain_state_timeline(brain.state, title="Brain State Evolution")
    fig1.savefig("brain_state_timeline.png", dpi=150, bbox_inches="tight")
    print("  Saved: brain_state_timeline.png")

    fig2 = plot_simulation_dashboard(brain, outputs, step_idx=-1)
    fig2.savefig("brain_dashboard.png", dpi=150, bbox_inches="tight")
    print("  Saved: brain_dashboard.png")

    fig3 = plot_connectivity_matrix(brain.synesthetic_net, title="Synesthetic Connections")
    fig3.savefig("connectivity_matrix.png", dpi=150, bbox_inches="tight")
    print("  Saved: connectivity_matrix.png")

    plt.close("all")

    # Show generated outputs info
    last = outputs[-1]
    print("\nGenerated cross-modal outputs at final step:")
    for mod, tensor in last.generated.items():
        print(f"  {mod.value}: shape {tensor.shape}")

    print("\n=== Full Brain Demo Complete ===")


if __name__ == "__main__":
    main()
