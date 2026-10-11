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

from brainnn.behavior import fit_exgaussian, simulate_session
from brainnn.core.phenotypes import MEDICATIONS, PHENOTYPES, get_phenotype
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
phenotype_key = st.sidebar.selectbox(
    "Attentional phenotype", list(PHENOTYPES.keys()),
    format_func=lambda k: PHENOTYPES[k].name,
    help="Sets tonic dopamine, phasic gain, cholinergic baseline, lapse rate and "
         "fatigue ceiling together. ADHD profiles have lower tonic dopamine, which "
         "disinhibits phasic bursts, and more frequent attentional lapses.")
medication_key = st.sidebar.selectbox(
    "Medication", list(MEDICATIONS.keys()),
    format_func=lambda k: "none" if MEDICATIONS[k] is None else MEDICATIONS[k].name,
    help="Mechanism-level approximation of each drug's primary action on the "
         "phenotype parameters. Directions are grounded in pharmacology; "
         "magnitudes are illustrative.")
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
@st.cache_resource(max_entries=6, show_spinner=False)
def build_brain(phenotype_key: str, medication_key: str, coupling: float,
                max_noise: float, seed: int):
    """Build the simulator once per parameter set.

    Weights are random and untrained; seeding only makes them reproducible.
    A warm-up step materialises the LazyLinear layers so that seeding immediately
    before `simulate()` yields an identical RNG stream on every run, cached or not.

    The phenotype is attached to the state *before* the warm-up and reset so that
    reset() initialises every neuromodulator from the phenotype's baselines. The
    config's baseline_dopamine is set to match the phenotype's tonic level because
    brain.reset() passes it through explicitly.
    """
    pheno = get_phenotype(phenotype_key, medication_key)

    torch.manual_seed(seed)
    config = BrainConfig.default_three_modality()
    config.adhd.baseline_dopamine = pheno.tonic_dopamine
    config.synesthesia.synesthesia_strength = coupling
    config.adhd.max_noise_std = max_noise

    brain = NeuroDivergentBrainSimulator(config)
    brain.state.phenotype = pheno
    with torch.no_grad():
        brain.step(make_inputs(0, 1), reward_signal=0.0, generate=False)
    brain.reset()
    return brain


def run_simulation(params: dict):
    brain = build_brain(params["phenotype"], params["medication"],
                        params["syn"], params["noise"], params["seed"])

    rewards = [0.0] * params["steps"]
    rewards[int(params["r1_step"])] = params["r1_val"]
    rewards[int(params["r2_step"])] = params["r2_val"]

    inputs_seq = [make_inputs(i, params["steps"]) for i in range(params["steps"])]

    # Lapses are sampled from the state's own generator; seed it alongside torch
    # so the lapse pattern is reproducible for a given seed.
    brain.state._rng = np.random.default_rng(params["seed"])
    torch.manual_seed(params["seed"])
    with torch.no_grad():
        outputs = brain.simulate(inputs_seq, reward_signals=rewards, generate=False)
    return brain, outputs, rewards


params = {
    "steps": steps,
    "phenotype": phenotype_key,
    "medication": medication_key,
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
    """Return `precision`, `value_gate` and `hyperfocused` series.

    v0.3 records precision and value_gate directly in the history, including the
    effect of attentional lapses and the phenotype's fatigue sensitivity. Those
    recorded values are authoritative — recomputing them from a default-phenotype
    probe (the v0.2 approach) would silently drop the lapse dips. The probe is
    kept only for the hyperfocus flag, which is not recorded.
    """
    n = len(hist["dopamine"])
    probe = BrainState()
    hyperfocus = []
    for i in range(n):
        probe.focus = hist["focus"][i]
        hyperfocus.append(probe.is_hyperfocused)

    out = {"hyperfocused": hyperfocus}
    for key in ("precision", "value_gate"):
        if key in hist and len(hist[key]) == n:
            out[key] = list(hist[key])
    return out


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
# 2b. Attentional lapses
# ============================================================
if "lapse" in series and any(series["lapse"]):
    lapse_steps = [i for i, v in enumerate(series["lapse"]) if v]
    st.caption(
        f"**Attentional lapses** on {len(lapse_steps)} of {n_steps} steps "
        f"(steps {', '.join(map(str, lapse_steps[:12]))}"
        f"{'…' if len(lapse_steps) > 12 else ''}). Each one is a transient collapse "
        f"of precision — the dips you can see in the precision trace above. Lapses are "
        f"the mechanism behind the slow tail in the reaction-time section below."
    )

# ============================================================
# 2c. Reaction-time signature — the behavioural prediction
# ============================================================
st.header("What this predicts about behaviour")
st.caption(
    "The same state machinery, run through a sustained-attention task for 400 trials, "
    "produces a reaction-time distribution. The most replicated behavioural finding in "
    "ADHD is that this distribution gains a **heavy right tail** — occasional very slow "
    "responses from attentional lapses — rather than shifting uniformly. In the "
    "ex-Gaussian decomposition that tail is **τ (tau)**; the Gaussian spread is **σ**; "
    "the centre is **μ**. The published group difference is largest in τ, moderate in σ, "
    "smallest in μ (Leth-Steensen et al. 2000; Kofler et al. 2013 meta-analysis)."
)


@st.cache_data(show_spinner=False)
def rt_session_summary(phenotype_key: str, medication_key: str, seed: int):
    """One simulated participant's RTs plus the ex-Gaussian fit.

    Returns plain arrays and floats so the result caches cleanly.
    """
    pheno = get_phenotype(phenotype_key, medication_key)
    sess = simulate_session(pheno, n_trials=400, seed=seed)
    fit = fit_exgaussian(sess["rt"])
    return {
        "rt": np.asarray(sess["rt"], dtype=float),
        "lapse_rate": float(np.mean(sess["lapse"])),
        "mu": fit.mu, "sigma": fit.sigma, "tau": fit.tau,
        "name": pheno.name,
    }


selected = rt_session_summary(phenotype_key, medication_key, int(seed))
reference = rt_session_summary("neurotypical", "none", int(seed))

rt_col, fit_col = st.columns([3, 2])

with rt_col:
    fig_rt = go.Figure()
    bins = dict(start=150, end=1500, size=20)
    fig_rt.add_trace(go.Histogram(
        x=reference["rt"], xbins=bins, histnorm="probability density",
        name=reference["name"], marker_color="#2a9d8f", opacity=0.55,
    ))
    if selected["name"] != reference["name"]:
        fig_rt.add_trace(go.Histogram(
            x=selected["rt"], xbins=bins, histnorm="probability density",
            name=selected["name"], marker_color="#e63946", opacity=0.55,
        ))
    fig_rt.update_layout(
        barmode="overlay", height=320,
        xaxis=dict(title="reaction time (ms)", range=[150, 1500], gridcolor="#eee"),
        yaxis=dict(title="density", gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=40, l=50, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_rt, width='stretch')

with fit_col:
    st.markdown("**Ex-Gaussian fit (ms)**")
    rows = [("μ  centre", "mu"), ("σ  spread", "sigma"), ("τ  tail", "tau")]
    header = f"| | {reference['name']} | {selected['name']} |\n|---|---:|---:|\n"
    body = "".join(
        f"| {label} | {reference[k]:.0f} | {selected[k]:.0f} |\n" for label, k in rows
    )
    body += (f"| lapse rate | {reference['lapse_rate']:.1%} | "
             f"{selected['lapse_rate']:.1%} |\n")
    st.markdown(header + body)
    if selected["name"] != reference["name"]:
        ratio = selected["tau"] / max(reference["tau"], 1e-9)
        st.caption(
            f"τ is **{ratio:.1f}×** the neurotypical value here. Published ADHD/control "
            f"τ ratios are roughly 1.5–2.5×, so this model overproduces the tail — the "
            f"*ordering* τ > σ > μ is reproduced, the *magnitude* is not. Likely cause: "
            f"decision time is hyperbolic in precision, which amplifies lapses. "
            f"See `scripts/validate_adhd.py` for the group-level comparison."
        )
    else:
        st.caption("Select an ADHD phenotype in the sidebar to see the contrast.")

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
