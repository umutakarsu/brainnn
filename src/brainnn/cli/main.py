"""CLI entry point for the brainnn simulator."""

from __future__ import annotations

import sys

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
except ImportError:
    print("CLI dependencies not installed. Run: pip install brainnn[cli]")
    sys.exit(1)

import torch
import matplotlib.pyplot as plt

from brainnn.core.config import BrainConfig, ModalityType
from brainnn.core.brain import NeuroDivergentBrainSimulator
from brainnn.viz.dashboard import plot_brain_state_timeline, plot_simulation_dashboard

app = typer.Typer(
    name="brainnn",
    help="Neurodivergent Brain Simulator — synesthesia & ADHD modeling",
)
console = Console()


@app.command()
def simulate(
    steps: int = typer.Option(50, "--steps", "-s", help="Number of simulation steps"),
    dopamine: float = typer.Option(0.3, "--dopamine", "-d", help="Baseline dopamine [0-1]"),
    synesthesia_strength: float = typer.Option(0.5, "--synesthesia", help="Synesthesia strength [0-1]"),
    noise: float = typer.Option(0.5, "--noise", help="Max attention noise [0-1]"),
    reward_at: str = typer.Option("", "--reward-at", help="Steps with reward, e.g. '10:0.8,30:0.5'"),
    save: str = typer.Option("", "--save", help="Save dashboard to file path"),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots"),
) -> None:
    """Run a brain simulation with configurable parameters."""
    config = BrainConfig.default_three_modality()
    config.adhd.baseline_dopamine = dopamine
    config.synesthesia.synesthesia_strength = synesthesia_strength
    config.adhd.max_noise_std = noise

    brain = NeuroDivergentBrainSimulator(config)

    # Parse reward schedule
    rewards = [0.0] * steps
    if reward_at:
        for entry in reward_at.split(","):
            step_str, val_str = entry.split(":")
            rewards[int(step_str)] = float(val_str)

    # Generate random inputs for simulation
    console.print("[bold cyan]Starting simulation...[/bold cyan]")
    inputs_seq = []
    for _ in track(range(steps), description="Preparing inputs"):
        inputs_seq.append({
            ModalityType.VISUAL: torch.randn(1, 3, 64, 64),
            ModalityType.AUDITORY: torch.randn(1, 1, 16000),
            ModalityType.TACTILE: torch.randn(1, 1, 32, 32),
        })

    # Run simulation
    with torch.no_grad():
        outputs = brain.simulate(inputs_seq, reward_signals=rewards, generate=False)

    # Print state summary
    table = Table(title="Final Brain State")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    snapshot = outputs[-1].brain_state_snapshot
    for key, val in snapshot.items():
        table.add_row(key, f"{val:.4f}" if isinstance(val, float) else str(val))
    console.print(table)

    # Plot dashboard
    fig = plot_simulation_dashboard(brain, outputs)

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        console.print(f"[green]Dashboard saved to {save}[/green]")

    if show:
        plt.show()


@app.command()
def info() -> None:
    """Show simulator configuration info."""
    from brainnn import __version__
    config = BrainConfig.default_three_modality()

    table = Table(title=f"Brainnn v{__version__}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Modalities", ", ".join(m.value for m in config.modalities))
    table.add_row("Latent Dim", str(config.latent_dim))
    table.add_row("Synesthesia Layers", str(config.synesthesia.connection_layers))
    table.add_row("Synesthesia Strength", str(config.synesthesia.synesthesia_strength))
    table.add_row("ADHD Num Heads", str(config.adhd.num_heads))
    table.add_row("ADHD Baseline Dopamine", str(config.adhd.baseline_dopamine))
    table.add_row("ADHD Hyperfocus Threshold", str(config.adhd.hyperfocus_threshold))
    console.print(table)


if __name__ == "__main__":
    app()
