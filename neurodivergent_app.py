"""Neurodivergent Brain Simulator — Web Dashboard"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from brainnn.core.config import BrainConfig, ModalityType
from brainnn.core.brain import NeuroDivergentBrainSimulator

st.set_page_config(page_title="Neurodivergent", layout="wide")

# --- Minimal custom styling ---
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #fafafa; }
    h1, h2, h3 { font-weight: 500; }
</style>
""", unsafe_allow_html=True)

st.title("Neurodivergent Brain Simulator")

# Muted, professional color palette
COLORS = {
    "dopamine": "#5a7d9a",
    "focus": "#2c3e50",
    "arousal": "#7f8c8d",
    "fatigue": "#bdc3c7",
    "accent": "#34495e",
    "reward": "#95a5a6",
}

# --- About section ---
with st.expander("About this simulation", expanded=False):
    st.markdown("""
This simulator models a **neurodivergent brain** processing three sensory channels
(vision, hearing, touch) simultaneously. It demonstrates two phenomena:

**ADHD Attention Model** —
The brain's ability to focus is governed by dopamine. When dopamine is low,
noise corrupts the attention signal and focus scatters. When dopamine spikes
above a threshold, the brain enters **hyperfocus** — locking onto a single target
while filtering everything else. Over time, dopamine decays and fatigue builds,
causing focus to collapse again.

**Synesthesia Model** —
In a typical brain, sensory channels are processed independently and merged only
at the final stage. In a synesthetic brain, **lateral connections** exist between
intermediate processing layers — a sound can automatically trigger a color percept,
or a texture can evoke an auditory sensation.

---

**How is synesthesia related?**

In a standard multimodal neural network, each sense is encoded separately and
combined only at the very end (late fusion):

```
Visual:    [V1] → [V2] → [V3] ──┐
                                  ├── Merge → Output
Auditory:  [A1] → [A2] → [A3] ──┤
                                  │
Tactile:   [T1] → [T2] → [T3] ──┘
```

This simulator adds **lateral connections** between the intermediate layers of
each encoder — so information "bleeds" across senses during processing, not
just at the end:

```
Visual:    [V1] → [V2] → [V3] ──┐
              ↕       ↕       ↕    ├── Merge → Output
Auditory:  [A1] → [A2] → [A3] ──┤
              ↕       ↕       ↕    │
Tactile:   [T1] → [T2] → [T3] ──┘
```

Each `↕` is a learnable projection with a gating mechanism. This mirrors the
neuroscience of synesthesia: synesthetic individuals have extra neural pathways
between sensory cortices that cause one sense to involuntarily activate another.

The **Synesthesia Strength** slider controls how much information flows through
these lateral connections. At 0, senses are fully independent (neurotypical).
At higher values, the internal representations of each sense start to resemble
each other — just as a synesthete might "see" colors when hearing music.

You can observe this in the **Modality Latents** chart: as synesthesia strength
increases, the activation patterns across modalities become more correlated.
The **Cross-Modal Generation** section shows the result — the brain can produce
a visual pattern from sound input, or a waveform from touch, because the fused
representation carries information from all senses simultaneously.

---

**Parameters**

| Parameter | What it controls | Low | High |
|---|---|---|---|
| Simulation Steps | Duration of the simulation in discrete timesteps | Short run | Long run |
| Baseline Dopamine | Starting dopamine level (typically low in ADHD) | Scattered attention | Focused start |
| Synesthesia Strength | Cross-modal connection intensity | Minimal blending | Strong sensory mixing |
| ADHD Noise | Noise injected into the attention signal | Calm focus | Highly distracted |

**Reward Events** are external stimuli that arrive at specific timesteps — think of
a sudden interesting sound or an unexpected visual. Each reward boosts dopamine by its
intensity value. A strong enough reward can trigger hyperfocus. The **step** is *when*
the stimulus arrives; the **intensity** is *how strong* it is (0 = none, 1 = maximum).

---

**Reading the charts**

- **Brain State Timeline** — Tracks dopamine, focus, arousal, and fatigue over time.
  Dashed lines mark reward events.
- **Attention Weights** — Shows how each attention head distributes focus across modalities.
- **Modality Latents** — Internal representation of each sensory channel at a given step.
- **Cross-Modal Generation** — What the brain produces after fusing all senses:
  a visual pattern, an audio waveform, and a tactile pressure map.
""")

# --- Sidebar controls ---
st.sidebar.header("Parameters")
steps = st.sidebar.slider("Simulation Steps", 10, 100, 40,
                           help="Number of discrete timesteps to simulate.")
dopamine = st.sidebar.slider("Baseline Dopamine", 0.0, 1.0, 0.35,
                              help="Starting dopamine level. Lower = more scattered attention.")
syn_strength = st.sidebar.slider("Synesthesia Strength", 0.0, 1.0, 0.7,
                                  help="Cross-modal connection strength. Higher = more sensory blending.")
noise = st.sidebar.slider("ADHD Noise", 0.0, 1.0, 0.4,
                           help="Noise magnitude in the attention signal. Higher = more distraction.")

st.sidebar.markdown("---")
st.sidebar.header("Reward Events")
st.sidebar.caption("External stimuli delivered during the simulation")
r1_step = st.sidebar.number_input("Reward 1 — Step", 0, steps - 1, min(5, steps - 1),
                                   help="Timestep when this stimulus arrives")
r1_val = st.sidebar.slider("Reward 1 — Intensity", 0.0, 1.0, 0.6,
                            help="Strength of the dopamine boost")
r2_step = st.sidebar.number_input("Reward 2 — Step", 0, steps - 1, min(15, steps - 1),
                                   help="Timestep when this stimulus arrives")
r2_val = st.sidebar.slider("Reward 2 — Intensity", 0.0, 1.0, 0.9,
                            help="Strength of the dopamine boost")

# --- Run simulation ---
if st.sidebar.button("Run Simulation", type="primary") or "outputs" not in st.session_state:
    config = BrainConfig.default_three_modality()
    config.adhd.baseline_dopamine = dopamine
    config.synesthesia.synesthesia_strength = syn_strength
    config.adhd.max_noise_std = noise

    brain = NeuroDivergentBrainSimulator(config)

    rewards = [0.0] * steps
    rewards[int(r1_step)] = r1_val
    rewards[int(r2_step)] = r2_val

    inputs_seq = []
    for _ in range(steps):
        inputs_seq.append({
            ModalityType.VISUAL: torch.randn(1, 3, 32, 32),
            ModalityType.AUDITORY: torch.randn(1, 1, 1024),
            ModalityType.TACTILE: torch.randn(1, 1, 16, 16),
        })

    with torch.no_grad():
        outputs = brain.simulate(inputs_seq, reward_signals=rewards, generate=True)

    st.session_state.outputs = outputs
    st.session_state.brain = brain
    st.session_state.rewards = rewards

outputs = st.session_state.get("outputs")
brain = st.session_state.get("brain")
rewards = st.session_state.get("rewards")

if outputs is None:
    st.info("Click **Run Simulation** to start.")
    st.stop()

# --- Extract state history ---
history = brain.state.history
x = list(range(len(history["dopamine"])))

# ============================================================
# 1. Brain State Timeline
# ============================================================
st.header("Brain State Timeline")

fig_state = make_subplots(rows=1, cols=1)

for key, color in COLORS.items():
    if key in history:
        fig_state.add_trace(go.Scatter(
            x=x, y=history[key], mode="lines", name=key.capitalize(),
            line=dict(color=color, width=2),
        ))

for i, r in enumerate(rewards):
    if r > 0:
        fig_state.add_vline(x=i, line_dash="dash", line_color=COLORS["reward"], line_width=1.5,
                            annotation_text=f"reward {r:.1f}", annotation_position="top",
                            annotation_font_size=10, annotation_font_color="#7f8c8d")

fig_state.update_layout(
    height=320,
    yaxis=dict(range=[0, 1.05], title="Value", gridcolor="#eee"),
    xaxis=dict(title="Step", gridcolor="#eee"),
    margin=dict(t=20, b=40, l=50, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig_state, use_container_width=True)

# ============================================================
# 2. Attention Weights + Modality Activations
# ============================================================
step_idx = st.slider("Inspect Step", 0, len(outputs) - 1, len(outputs) - 1)
out = outputs[step_idx]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Attention Weights")
    weights = out.attention_weights.detach().cpu().numpy()[0]
    if weights.ndim == 3:
        weights = weights.mean(axis=-1)
    fig_attn = go.Figure(go.Heatmap(
        z=weights, colorscale="Greys", showscale=True,
    ))
    fig_attn.update_layout(
        height=280, yaxis_title="Head", xaxis_title="Modality",
        margin=dict(t=10, b=40, l=50, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_attn, use_container_width=True)

with col2:
    st.subheader("Modality Latents")
    mod_names = []
    mod_data = []
    for mod, tensor in out.modality_latents.items():
        vals = tensor.detach().cpu().numpy()[0]
        mod_names.append(mod.value.capitalize())
        mod_data.append(vals)

    latent_colors = ["#5a7d9a", "#2c3e50", "#95a5a6"]
    fig_lat = go.Figure()
    for i, (name, data) in enumerate(zip(mod_names, mod_data)):
        fig_lat.add_trace(go.Bar(
            x=list(range(len(data))), y=data, name=name,
            opacity=0.6, marker_color=latent_colors[i % len(latent_colors)],
        ))
    fig_lat.update_layout(
        height=280, barmode="overlay",
        xaxis_title="Latent Dim", yaxis_title="Activation",
        margin=dict(t=10, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_lat, use_container_width=True)

# ============================================================
# 3. State Snapshot
# ============================================================
st.subheader("State Snapshot")
snap = out.brain_state_snapshot
cols = st.columns(6)
labels = ["Dopamine", "Focus", "Arousal", "Fatigue", "SNR", "Hyperfocus"]
values = [
    f"{snap['dopamine']:.3f}",
    f"{snap['focus']:.3f}",
    f"{snap['arousal']:.3f}",
    f"{snap['fatigue']:.3f}",
    f"{snap['snr']:.3f}",
    "Yes" if snap["is_hyperfocused"] > 0.5 else "No",
]
for col, label, val in zip(cols, labels, values):
    col.metric(label, val)

# ============================================================
# 4. Cross-modal generation
# ============================================================
st.header("Cross-Modal Generation")
st.caption("Outputs generated from the fused latent representation at the selected step")

gen_cols = st.columns(3)
gen_labels = {ModalityType.VISUAL: "Visual", ModalityType.AUDITORY: "Auditory", ModalityType.TACTILE: "Tactile"}

for col, (mod, label) in zip(gen_cols, gen_labels.items()):
    if mod in out.generated:
        tensor = out.generated[mod].detach().cpu().numpy()[0]
        with col:
            st.markdown(f"**{label}**")
            if mod == ModalityType.VISUAL:
                img = tensor.transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                st.image(img, use_container_width=True)
            elif mod == ModalityType.AUDITORY:
                wave = tensor[0]
                fig_w = go.Figure(go.Scatter(
                    y=wave, mode="lines",
                    line=dict(width=1, color=COLORS["dopamine"]),
                ))
                fig_w.update_layout(
                    height=180, margin=dict(t=10, b=30, l=40, r=10),
                    xaxis_title="Time", yaxis_title="Amp",
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig_w, use_container_width=True)
            elif mod == ModalityType.TACTILE:
                pressure = tensor[0]
                fig_p = go.Figure(go.Heatmap(z=pressure, colorscale="Greys", showscale=False))
                fig_p.update_layout(
                    height=180, margin=dict(t=10, b=30, l=40, r=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig_p, use_container_width=True)
