"""ADHD Attention Model — neuromodulator dynamics dashboard.

Interactive front-end for `brainnn`'s two-axis neuromodulated attention:
a precision axis (acetylcholine / norepinephrine) and a value axis (dopamine),
following Yu & Dayan (2005).

Note on scope: the network here is UNTRAINED. This dashboard visualises the
architecture's state dynamics, not learned perceptual content.
"""

import math
import os
import sys

# Kept for HuggingFace Spaces, where the package is not pip-installed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from brainnn.core.brain import NeuroDivergentBrainSimulator
from brainnn.core.config import BrainConfig, ModalityType
from brainnn.core.state import BrainState

st.set_page_config(
    page_title="ADHD Attention Model — Neuromodulator Dynamics",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #fafafa; }
    h1, h2, h3 { font-weight: 500; }
</style>
""", unsafe_allow_html=True)

st.title("ADHD Attention Model")
st.caption(
    "Two-axis neuromodulation of multi-head attention — precision (ACh / NE) "
    "and value (dopamine), after Yu & Dayan (2005)."
)

# ============================================================
# Colours
#   PALETTE — general UI accents, not tied to any data series.
#   *_SERIES — the plotted series. Kept separate from PALETTE on purpose:
#   the old code iterated the palette to decide what to plot, which silently
#   dropped acetylcholine and norepinephrine from every chart.
# ============================================================
PALETTE = {
    "accent": "#264653",
    "reward_marker": "#9aa7ad",
    "grid": "#eee",
}

# (history key, legend label, colour, dash)
NEUROMOD_SERIES = [
    ("dopamine", "Dopamine — value gating", "#e63946", "solid"),
    ("acetylcholine", "Acetylcholine — precision", "#2a9d8f", "solid"),
    ("norepinephrine", "Norepinephrine — arousal / surprise", "#f4a261", "solid"),
]

DERIVED_SERIES = [
    ("precision", "Precision (from ACh + NE) — how crisp attention is", "#2a9d8f", "solid"),
    ("value_gate", "Value gate (from dopamine) — how hard attention locks on", "#e63946", "solid"),
    ("focus", "Focus", "#457b9d", "dash"),
    ("arousal", "Arousal", "#8d99ae", "dot"),
    ("fatigue", "Fatigue", "#e9c46a", "dot"),
]

MODALITY_ORDER = [ModalityType.VISUAL, ModalityType.AUDITORY, ModalityType.TACTILE]
MODALITY_LABELS = ["Visual", "Auditory", "Tactile"]

# ============================================================
# Scope notice — always visible, never collapsed.
# ============================================================
st.warning(
    "**What this is.** The network weights are **untrained** — nothing here has learned "
    "to see or hear. What you are looking at is the *architecture's state dynamics*: how "
    "dopamine, acetylcholine and norepinephrine reshape multi-head attention over time. "
    "The attention maps and latents are the model's internal state under those "
    "neuromodulators, not recognised perceptual content. Runs are seeded, so the same "
    "settings always give the same result."
)

# ============================================================
# About — short, scannable, open by default.
# ============================================================
with st.expander("About this model", expanded=True):
    st.markdown("""
Attention in this model is steered by **two separate neuromodulatory axes**, not one.

**Precision axis — acetylcholine and norepinephrine.** This sets *how sharp* attention is.
ACh tracks **expected** uncertainty ("I know how noisy this is"); NE tracks **unexpected**
uncertainty — surprise and arousal — with an inverted-U (Yerkes–Dodson) shape, so both
drowsy and jittery degrade precision. Together they set the noise floor added to the raw
attention scores. Low precision looks like scattered, drifting focus.

**Value axis — dopamine.** This sets *how hard* attention commits. It is a cost–benefit
signal — "is this worth attending to?" — and it gates which attention heads get amplified
and which get suppressed. High dopamine locks onto a few heads (hyperfocus); low dopamine
gates weakly and heads drop out ("why bother"). Fatigue erodes both axes over time.

The split follows **Yu & Dayan (2005)** and was adopted after feedback from **Peter Dayan**.
The earlier version of this code collapsed both axes onto dopamine, which does not match the
neuromodulator literature. One practical consequence: the two axes are pharmacologically
dissociable in principle — cholinergic and noradrenergic drugs move precision, dopaminergic
drugs move value gain.

The encoders also carry cross-modal lateral connections (a synesthesia-style pathway); the
**Cross-modal coupling** slider controls their strength, and its main visible effect here is
how correlated the three modality latents become. That line of work is developed elsewhere.

*Reward events* are external stimuli at chosen timesteps: each bumps dopamine (value) and
norepinephrine (surprise), and a strong one can briefly relieve fatigue.
""")

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Parameters")
steps = st.sidebar.slider("Simulation steps", 10, 100, 40,
                          help="Number of discrete timesteps to simulate.")
dopamine = st.sidebar.slider("Baseline dopamine", 0.0, 1.0, 0.35,
                             help="Starting dopamine. Lower = weaker value gating, "
                                  "more scattered attention.")
syn_strength = st.sidebar.slider("Cross-modal coupling", 0.0, 1.0, 0.7,
                                 help="Strength of lateral connections between modality "
                                      "encoders.")
noise = st.sidebar.slider("Max attention noise", 0.0, 1.0, 0.4,
                          help="Ceiling on the ACh/NE-driven noise injected into attention "
                               "scores. Higher = more distractible at low precision.")
seed = st.sidebar.number_input("Random seed", 0, 9999, 0,
                               help="Fixed seed. Same settings + same seed = same run.")

st.sidebar.markdown("---")
st.sidebar.header("Reward events")
st.sidebar.caption("External stimuli delivered during the simulation")
r1_step = st.sidebar.number_input("Reward 1 — step", 0, steps - 1, min(5, steps - 1))
r1_val = st.sidebar.slider("Reward 1 — intensity", 0.0, 1.0, 0.6)
r2_step = st.sidebar.number_input("Reward 2 — step", 0, steps - 1, min(15, steps - 1))
r2_val = st.sidebar.slider("Reward 2 — intensity", 0.0, 1.0, 0.9)

st.sidebar.markdown("---")
rerun_clicked = st.sidebar.button("Re-run simulation", type="primary")


# ============================================================
# Inputs — structured, not pure noise.
# ============================================================
def make_inputs(step: int, total_steps: int) -> dict:
    """Structured stimuli: a drifting oriented grating, a chirping tone, a moving
    pressure blob. Deterministic given `step`; shapes match the encoders.

    These are stimuli, not a dataset — the point is that the input is a coherent
    signal rather than white noise, so attention has something structured to act on.
    """
    phase = step / max(1, total_steps)

    # Visual: 32x32 oriented grating whose orientation drifts across the run.
    theta = phase * math.pi
    yy, xx = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing="ij")
    grating = np.sin(8.0 * (xx * math.cos(theta) + yy * math.sin(theta)) + phase * 6.0)
    visual = np.stack([grating, grating * 0.6, grating * 0.3])[None].astype(np.float32)

    # Auditory: 1024-sample tone that sweeps upward in frequency.
    t = np.linspace(0.0, 1.0, 1024)
    f0 = 6.0 + 24.0 * phase
    audio = (np.sin(2 * np.pi * f0 * t) * np.exp(-2.0 * t)).astype(np.float32)
    audio = audio[None, None]

    # Tactile: 16x16 pressure blob that travels across the surface.
    gy, gx = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16), indexing="ij")
    cx = math.cos(phase * 2 * math.pi) * 0.5
    cy = math.sin(phase * 2 * math.pi) * 0.5
    blob = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / 0.15)
    tactile = blob[None, None].astype(np.float32)

    return {
        ModalityType.VISUAL: torch.from_numpy(visual),
        ModalityType.AUDITORY: torch.from_numpy(audio),
        ModalityType.TACTILE: torch.from_numpy(tactile),
    }


# ============================================================
# Model construction (cached) and simulation
# ============================================================
@st.cache_resource(max_entries=4, show_spinner=False)
def build_brain(baseline_dopamine: float, coupling: float, max_noise: float, seed: int):
    """Build the simulator once per parameter set.

    Weights are random and untrained; seeding only makes them reproducible.
    A warm-up step materialises the LazyLinear layers so that seeding immediately
    before `simulate()` yields an identical RNG stream on every run, cached or not.
    """
    torch.manual_seed(seed)
    config = BrainConfig.default_three_modality()
    config.adhd.baseline_dopamine = baseline_dopamine
    config.synesthesia.synesthesia_strength = coupling
    config.adhd.max_noise_std = max_noise

    brain = NeuroDivergentBrainSimulator(config)
    with torch.no_grad():
        brain.step(make_inputs(0, 1), reward_signal=0.0, generate=False)
    brain.reset()
    return brain


def run_simulation(params: dict):
    brain = build_brain(params["dopamine"], params["syn"], params["noise"], params["seed"])

    rewards = [0.0] * params["steps"]
    rewards[int(params["r1_step"])] = params["r1_val"]
    rewards[int(params["r2_step"])] = params["r2_val"]

    inputs_seq = [make_inputs(i, params["steps"]) for i in range(params["steps"])]

    torch.manual_seed(params["seed"])
    with torch.no_grad():
        outputs = brain.simulate(inputs_seq, reward_signals=rewards, generate=False)
    return brain, outputs, rewards


params = {
    "steps": steps,
    "dopamine": dopamine,
    "syn": syn_strength,
    "noise": noise,
    "seed": int(seed),
    "r1_step": r1_step,
    "r1_val": r1_val,
    "r2_step": r2_step,
    "r2_val": r2_val,
}
signature = tuple(sorted(params.items()))

# Auto-run on first load; re-run whenever any control changes (so what is on screen
# can never silently disagree with the sidebar) or the button is pressed.
if rerun_clicked or st.session_state.get("signature") != signature:
    with st.spinner("Running simulation..."):
        brain, outputs, rewards = run_simulation(params)
    st.session_state.signature = signature
    st.session_state.result = (brain, outputs, rewards)

brain, outputs, rewards = st.session_state.result

# ============================================================
# Series: recorded history + derived precision / value_gate
# ============================================================
history = brain.state.history
n_steps = len(history["dopamine"])
x = list(range(n_steps))


def derived_series(hist: dict) -> dict:
    """Recompute `precision` and `value_gate` from the recorded history.

    Uses BrainState's own properties rather than re-implementing the formulas,
    so this cannot drift from `src/brainnn/core/state.py`.
    """
    probe = BrainState()
    precision, value_gate, hyperfocus = [], [], []
    for i in range(len(hist["dopamine"])):
        probe.acetylcholine = hist["acetylcholine"][i]
        probe.norepinephrine = hist["norepinephrine"][i]
        probe.fatigue = hist["fatigue"][i]
        probe.dopamine = hist["dopamine"][i]
        probe.focus = hist["focus"][i]
        precision.append(probe.precision)
        value_gate.append(probe.value_gate)
        hyperfocus.append(probe.is_hyperfocused)
    return {"precision": precision, "value_gate": value_gate, "hyperfocused": hyperfocus}


series = dict(history)
series.update(derived_series(history))

missing = [k for k, *_ in NEUROMOD_SERIES + DERIVED_SERIES if k not in series]
if missing:
    st.error(f"Series missing from the simulation history: {', '.join(missing)}")


def timeline_figure(spec, title_y="Level"):
    fig = go.Figure()
    for key, label, color, dash in spec:
        if key not in series:
            continue
        fig.add_trace(go.Scatter(
            x=x, y=series[key], mode="lines", name=label,
            line=dict(color=color, width=2, dash=dash),
        ))
    for i, r in enumerate(rewards):
        if r > 0:
            fig.add_vline(
                x=i, line_dash="dash", line_color=PALETTE["reward_marker"], line_width=1.5,
                annotation_text=f"reward {r:.1f}", annotation_position="top",
                annotation_font_size=10, annotation_font_color="#7f8c8d",
            )
    fig.update_layout(
        height=340,
        yaxis=dict(range=[0, 1.05], title=title_y, gridcolor=PALETTE["grid"]),
        xaxis=dict(title="Step", gridcolor=PALETTE["grid"]),
        margin=dict(t=20, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ============================================================
# 1. Neuromodulators
# ============================================================
st.header("Neuromodulators over time")
st.caption(
    "The three signals that drive attention. Acetylcholine and norepinephrine form the "
    "**precision** axis (how sharp attention is); dopamine forms the **value** axis "
    "(how hard attention commits). Dashed vertical lines mark reward events — each one "
    "raises dopamine and, as a surprise signal, norepinephrine."
)
st.plotly_chart(timeline_figure(NEUROMOD_SERIES), width='stretch')

# ============================================================
# 2. Derived attention state
# ============================================================
st.header("What the two axes produce")
st.caption(
    "Precision is computed from acetylcholine and norepinephrine (never from dopamine); "
    "the value gate is computed from dopamine. Focus is driven by both, and fatigue "
    "erodes both as the run goes on."
)
st.plotly_chart(timeline_figure(DERIVED_SERIES), width='stretch')

# ============================================================
# 3. Step inspector
# ============================================================
st.header("Inspect a single step")
step_idx = st.slider("Step", 0, len(outputs) - 1, len(outputs) - 1)
out = outputs[step_idx]

metric_specs = [
    ("Dopamine", f"{series['dopamine'][step_idx]:.3f}"),
    ("Acetylcholine", f"{series['acetylcholine'][step_idx]:.3f}"),
    ("Norepinephrine", f"{series['norepinephrine'][step_idx]:.3f}"),
    ("Precision", f"{series['precision'][step_idx]:.3f}"),
    ("Value gate", f"{series['value_gate'][step_idx]:.3f}"),
    ("Focus", f"{series['focus'][step_idx]:.3f}"),
    ("Hyperfocus", "Yes" if series["hyperfocused"][step_idx] else "No"),
]
for col, (label, value) in zip(st.columns(len(metric_specs)), metric_specs):
    col.metric(label, value)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Attention weights")
    st.caption("Per-head attention across the three input channels, after precision noise "
               "and dopaminergic head gating.")
    weights = out.attention_weights.detach().cpu().numpy()[0]
    if weights.ndim == 3:
        weights = weights.mean(axis=-1)
    fig_attn = go.Figure(go.Heatmap(
        z=weights, colorscale="Greys", showscale=True,
        x=MODALITY_LABELS[:weights.shape[1]],
    ))
    fig_attn.update_layout(
        height=300, yaxis_title="Attention head", xaxis_title="Input channel",
        margin=dict(t=10, b=40, l=50, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_attn, width='stretch')

with col2:
    st.subheader("Modality latents")
    st.caption("Untrained encoder activations per channel. Cross-modal coupling makes these "
               "traces more similar to one another.")
    latent_colors = ["#5a7d9a", "#2c3e50", "#95a5a6"]
    fig_lat = go.Figure()
    for i, (mod, tensor) in enumerate(out.modality_latents.items()):
        fig_lat.add_trace(go.Bar(
            x=list(range(tensor.shape[-1])),
            y=tensor.detach().cpu().numpy()[0],
            name=mod.value.capitalize(),
            opacity=0.6,
            marker_color=latent_colors[i % len(latent_colors)],
        ))
    fig_lat.update_layout(
        height=300, barmode="overlay",
        xaxis_title="Latent dimension", yaxis_title="Activation",
        margin=dict(t=10, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_lat, width='stretch')

st.markdown("---")
st.caption(
    "Model: `brainnn` — ADHD attention with two-axis neuromodulation "
    "(precision: ACh/NE; value: dopamine), per Yu & Dayan (2005). "
    "Weights are untrained; this dashboard shows state dynamics, not learned representations."
)
