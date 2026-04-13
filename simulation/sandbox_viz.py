"""Sandbox visualization — generates plots for all simulation results."""

import sys
sys.path.insert(0, "/Users/umutakarsu/brainnn/src")
sys.path.insert(0, "/Users/umutakarsu/brainnn")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from brainnn.calibration.calibration_logic import CalibrationEngine, CorticalZone
from brainnn.synesthesia.synesthesia_engine import SynesthesiaEngine, SynesthesiaConfig, SpikeTrainEncoder
from brainnn.calibration.neuroplasticity import NeuroplasticityController
from simulation.sandbox import (
    SimulationSandbox, generate_synthetic_impedance,
    generate_synthetic_evoked, generate_synthetic_lfp,
)

# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------
sr = 44100
t = np.arange(sr) / sr

# Complex audio: 440Hz (A4) + 880Hz (A5) + 220Hz (bass)
audio = (0.4 * np.sin(2 * np.pi * 440 * t)
         + 0.3 * np.sin(2 * np.pi * 880 * t)
         + 0.3 * np.sin(2 * np.pi * 220 * t)
         + 0.05 * np.random.randn(sr))
audio /= np.max(np.abs(audio))

sandbox = SimulationSandbox(seed=42)
results = {}
for day in [1, 4, 7, 10, 14]:
    results[day] = sandbox.run(audio, onboarding_day=day)

# ---------------------------------------------------------------------------
# FIGURE 1: Onboarding Progress (Neuroplasticity Ramp)
# ---------------------------------------------------------------------------
fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle("SynapseFlow — Neural Simulation Sandbox Results", fontsize=16, fontweight="bold", y=0.98)

# 1a: Intensity Schedule (sigmoid curve)
ax = axes[0, 0]
schedule = NeuroplasticityController.default_schedule(14)
days = list(schedule.keys())
intensities = list(schedule.values())
ax.fill_between(days, intensities, alpha=0.3, color="#10B981")
ax.plot(days, intensities, "o-", color="#10B981", linewidth=2.5, markersize=8)
for d in [1, 7, 14]:
    ax.annotate(f"Day {d}: {schedule[d]:.0%}",
                (d, schedule[d]), textcoords="offset points",
                xytext=(10, 10), fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#333"))
ax.set_xlabel("Onboarding Day", fontsize=12)
ax.set_ylabel("Stimulation Intensity", fontsize=12)
ax.set_title("Neuroplasticity Onboarding Schedule (Sigmoid)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.set_xlim(0.5, 14.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(True, alpha=0.3)

# 1b: Cortical Response vs Day
ax = axes[0, 1]
resp_days = sorted(results.keys())
mean_resps = [np.mean(results[d].cortical_response) for d in resp_days]
ax.bar(resp_days, mean_resps, color=["#EF4444", "#F59E0B", "#10B981", "#3B82F6", "#8B5CF6"],
       width=0.8, edgecolor="white", linewidth=1.5)
for d, r in zip(resp_days, mean_resps):
    ax.text(d, r + 0.2, f"{r:.1f}", ha="center", fontweight="bold", fontsize=11)
ax.set_xlabel("Onboarding Day", fontsize=12)
ax.set_ylabel("Mean Cortical Response (a.u.)", fontsize=12)
ax.set_title("Brain Response Increases with Adaptation", fontsize=13, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)

# 1c: Audio → RGB Mapping (first 100 frames)
ax = axes[1, 0]
rgb_day14 = results[14].rgb_frames[:100]
# Show as color bar
for i in range(min(100, len(rgb_day14))):
    ax.axvspan(i, i+1, color=rgb_day14[i], alpha=0.9)
ax.set_xlim(0, 100)
ax.set_xlabel("Frame Index", fontsize=12)
ax.set_title("Synesthesia Output: Audio → RGB (Day 14, Full Intensity)", fontsize=13, fontweight="bold")
ax.set_yticks([])

# 1d: RGB channels over time
ax = axes[1, 1]
n_show = min(200, len(results[14].rgb_frames))
frames = results[14].rgb_frames[:n_show]
ax.plot(frames[:, 0], color="#EF4444", linewidth=1.5, label="Red (Bass: 20–250 Hz)")
ax.plot(frames[:, 1], color="#10B981", linewidth=1.5, label="Green (Mid: 250–4k Hz)")
ax.plot(frames[:, 2], color="#3B82F6", linewidth=1.5, label="Blue (Treble: 4k–20k Hz)")
ax.set_xlabel("Frame", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("RGB Channel Decomposition (Frequency Bands)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig("/Users/umutakarsu/brainnn/sandbox_results_1.png", dpi=150, bbox_inches="tight")
print("Saved: sandbox_results_1.png")

# ---------------------------------------------------------------------------
# FIGURE 2: Calibration & Electrode Map
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("SynapseFlow — Calibration & Electrode Analysis", fontsize=16, fontweight="bold", y=0.98)

cal = results[14].calibration

# 2a: Impedance heat map
ax = axes2[0, 0]
impedance_grid = np.zeros((32, 32))
for ep in cal.electrode_profiles:
    r, c = ep.grid_position
    impedance_grid[r, c] = ep.impedance_ohm / 1e6  # MΩ
im = ax.imshow(impedance_grid, cmap="RdYlGn_r", vmin=0, vmax=2.0)
ax.set_title("Electrode Impedance Map (MΩ)", fontsize=13, fontweight="bold")
ax.set_xlabel("Column")
ax.set_ylabel("Row")
plt.colorbar(im, ax=ax, label="Impedance (MΩ)")

# 2b: Cortical Zone Map
ax = axes2[0, 1]
zone_colors = {
    CorticalZone.VISUAL_V1: 0, CorticalZone.VISUAL_V4: 1,
    CorticalZone.AUDITORY_A1: 2, CorticalZone.SOMATOSENSORY_S1: 3,
    CorticalZone.MOTOR_M1: 4, CorticalZone.PREFRONTAL: 5,
    CorticalZone.UNKNOWN: 6,
}
zone_names = ["V1", "V4", "A1", "S1", "M1", "PFC", "?"]
zone_grid = np.full((32, 32), 6)
for ep in cal.electrode_profiles:
    r, c = ep.grid_position
    zone_grid[r, c] = zone_colors.get(ep.zone, 6)

cmap = plt.cm.get_cmap("Set2", 7)
im = ax.imshow(zone_grid, cmap=cmap, vmin=-0.5, vmax=6.5)
ax.set_title("Cortical Zone Classification", fontsize=13, fontweight="bold")
ax.set_xlabel("Column")
ax.set_ylabel("Row")
cbar = plt.colorbar(im, ax=ax, ticks=range(7))
cbar.ax.set_yticklabels(zone_names)

# 2c: Selectivity distribution
ax = axes2[1, 0]
selectivities = [ep.selectivity for ep in cal.active_electrodes]
ax.hist(selectivities, bins=30, color="#8B5CF6", edgecolor="white", alpha=0.8)
ax.axvline(np.mean(selectivities), color="#EF4444", linewidth=2, linestyle="--",
           label=f"Mean: {np.mean(selectivities):.3f}")
ax.set_xlabel("Selectivity Index", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Electrode Selectivity Distribution", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

# 2d: Spike train example (3 channels × 20ms)
ax = axes2[1, 1]
spikes = results[14].spike_trains[50]  # frame 50
colors = ["#EF4444", "#10B981", "#3B82F6"]
labels = ["Red (V1)", "Green (V4)", "Blue (A1)"]
for ch in range(3):
    spike_times = np.where(spikes[ch] == 1)[0]
    ax.eventplot([spike_times], lineoffsets=ch, linelengths=0.6, colors=colors[ch])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(labels)
ax.set_xlabel("Time (ms within 20ms window)", fontsize=12)
ax.set_title("Spike Train Encoding (Single Frame → Electrode Pulses)", fontsize=13, fontweight="bold")
ax.set_xlim(0, 20)
ax.grid(True, axis="x", alpha=0.3)

fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig("/Users/umutakarsu/brainnn/sandbox_results_2.png", dpi=150, bbox_inches="tight")
print("Saved: sandbox_results_2.png")

# ---------------------------------------------------------------------------
# FIGURE 3: Day comparison — RGB outputs at different onboarding stages
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(5, 1, figsize=(16, 10))
fig3.suptitle("SynapseFlow — Synesthesia Intensity Across Onboarding Days",
              fontsize=16, fontweight="bold", y=0.98)

for i, day in enumerate([1, 4, 7, 10, 14]):
    ax = axes3[i]
    rgb = results[day].rgb_frames[:200]
    for j in range(min(200, len(rgb))):
        ax.axvspan(j, j+1, color=rgb[j], alpha=0.9)
    ax.set_xlim(0, 200)
    ax.set_yticks([])
    intensity = sandbox.plasticity.schedule.get(day, 1.0)
    ax.set_ylabel(f"Day {day}\n({intensity:.0%})", fontsize=11, fontweight="bold", rotation=0, labelpad=50)

axes3[-1].set_xlabel("Frame", fontsize=12)
fig3.tight_layout(rect=[0, 0, 1, 0.96])
fig3.savefig("/Users/umutakarsu/brainnn/sandbox_results_3.png", dpi=150, bbox_inches="tight")
print("Saved: sandbox_results_3.png")

print("\nDone! Opening images...")
