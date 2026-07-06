"""BCI Pipeline Dashboard — cross-subject EEG transformer + neuromodulator conditioning.

Interactive Streamlit dashboard that walks a user through the full pipeline:
  1. Pick a subject and epoch → view raw EEG signal
  2. Show frequency spectrum with the mu/beta motor imagery bands highlighted
  3. Show model's per-layer transformer attention weights
  4. Show baseline prediction from the cross-subject transformer
  5. Toggle neuromodulator state (ACh, NE, DA) sliders → see how conditioning shifts
     the prediction confidence (Dayan-correction integration)

This is Project 4 of the sprint: makes the abstract "cross-subject foundation model"
concrete and inspectable, and demonstrates the neuromodulator conditioning
architecture proposed in the Dayan follow-up correspondence.
"""

import sys
from pathlib import Path

# Make imports work
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "src"))

import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="BCI Cross-Subject Transformer + Neuromodulator Conditioning",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    .stage {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 4px solid #10B981;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .stage-num {
        display: inline-block;
        background: #10B981;
        color: white;
        width: 32px; height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: bold;
        margin-right: 10px;
    }
    .explanation {
        background: #eff6ff;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 14px;
        color: #1e40af;
        margin: 8px 0 16px 0;
    }
    .confidence-bar {
        height: 24px;
        border-radius: 6px;
        overflow: hidden;
        background: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("# 🧠 Cross-Subject EEG Transformer")
st.markdown("### With Neuromodulator-Conditioned Decoding (Dayan-Correction Extension)")
st.markdown("""
<div class="explanation">
This dashboard visualizes an end-to-end BCI pipeline: a small transformer trained on
8 subjects, evaluated zero-shot on a held-out 9th subject (BNCI2014_001 motor imagery, 4 classes).
The decoder is wrapped with FiLM-style neuromodulator conditioning (acetylcholine, norepinephrine, dopamine)
following the Yu &amp; Dayan (2005) framework. Toggle the neuromodulator sliders below to see how the model's
predictions shift with the user's inferred cognitive state.
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data + model loading (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading BNCI subject...")
def load_subject_cached(subject_id: int):
    from brainnn.bci.datasets import load_bnci_subject
    return load_bnci_subject(subject_id=subject_id)


@st.cache_resource(show_spinner="Loading trained model checkpoint...")
def load_model_and_decoder():
    from brainnn.bci.models import EEGTransformer, EEGTransformerConfig
    from brainnn.bci.neuromod_decoder import NeuromodConditionedDecoder

    ckpt_path = _root / "checkpoints" / "eeg_transformer_subj1-8.pt"

    if not ckpt_path.exists():
        # Fall back to random-init model with a warning banner
        st.warning(f"No trained checkpoint at {ckpt_path.name} — using random-init model for demo. "
                   f"Run `python scripts/run_full_loso.py` to train.")
        cfg = EEGTransformerConfig(n_subjects=8)
        base_model = EEGTransformer(cfg)
        class_names = ["feet", "left_hand", "right_hand", "tongue"]
    else:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        mcfg = ckpt["model_cfg"]
        cfg = EEGTransformerConfig(
            n_channels=mcfg["n_channels"],
            n_timepoints=mcfg["n_timepoints"],
            n_classes=mcfg["n_classes"],
            n_subjects=mcfg["n_subjects"],
            embed_dim=mcfg["embed_dim"],
        )
        base_model = EEGTransformer(cfg)
        base_model.load_state_dict(ckpt["model_state"])
        class_names = ckpt["class_names"]

    base_model.eval()
    decoder = NeuromodConditionedDecoder(base_model)
    decoder.eval()
    return base_model, decoder, class_names


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.title("Controls")
subject_id = st.sidebar.slider("Held-out subject to explore", 1, 9, 9)
st.sidebar.caption("Subject 9 was held out during training (zero-shot).")

# Load
rec = load_subject_cached(subject_id)
base_model, decoder, class_names = load_model_and_decoder()

epoch_idx = st.sidebar.slider("Epoch (trial) index", 0, rec.n_epochs - 1, 0)

st.sidebar.divider()
st.sidebar.markdown("**Neuromodulator state (Dayan-correction)**")
ach = st.sidebar.slider("Acetylcholine (ACh) — precision", 0.0, 1.0, 0.5, 0.05,
                         help="Expected uncertainty / precision sharpening. High = crisp attention.")
ne = st.sidebar.slider("Norepinephrine (NE) — arousal", 0.0, 1.0, 0.5, 0.05,
                       help="Unexpected uncertainty / arousal. Inverted-U (Yerkes-Dodson) — optimal near 0.5.")
da = st.sidebar.slider("Dopamine (DA) — value gate", 0.0, 1.0, 0.5, 0.05,
                       help="Cost-benefit of attention. High = strong head selection.")

st.sidebar.divider()
st.sidebar.caption(f"Loaded subject {subject_id}: **{rec.n_epochs}** trials, "
                    f"**{rec.n_channels}** channels @ **{rec.sfreq}** Hz")


# ----------------------------------------------------------------------------
# Get the selected epoch
# ----------------------------------------------------------------------------
x = rec.X[epoch_idx]  # (channels, timepoints)
y_true = rec.y[epoch_idx]
label_true = class_names[y_true]

# Convert to tensor
x_tensor = torch.from_numpy(x).float().unsqueeze(0)  # (1, C, T)


# ----------------------------------------------------------------------------
# Section 1 — Raw signal
# ----------------------------------------------------------------------------
st.markdown("""
<div class="stage">
    <span class="stage-num">1</span>
    <b>RAW EEG</b> — one trial from the held-out subject
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="explanation">
Subject <b>{subject_id}</b>, trial <b>{epoch_idx}/{rec.n_epochs-1}</b>, true class = <b>{label_true}</b>.
The 22 EEG channels are z-scored per-channel (this is the network's actual input).
Motor imagery is 3-second window starting 500 ms after cue.
</div>
""", unsafe_allow_html=True)

t_axis = np.arange(rec.n_timepoints) / rec.sfreq  # seconds

fig_eeg = go.Figure()
for c in range(rec.n_channels):
    fig_eeg.add_trace(go.Scatter(
        x=t_axis, y=x[c] + c * 5,  # vertical stagger
        mode="lines", line=dict(width=0.8, color="#334155"),
        hovertemplate=f"Ch {c}<br>t=%{{x:.2f}}s<br>z=%{{y:.2f}}<extra></extra>",
        showlegend=False,
    ))
fig_eeg.update_layout(
    title=f"22-channel EEG (z-scored, stacked)",
    xaxis_title="Time (s)",
    yaxis=dict(showticklabels=False, title="Channel (stacked)"),
    height=420, template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
)
st.plotly_chart(fig_eeg, use_container_width=True)


# ----------------------------------------------------------------------------
# Section 2 — Frequency spectrum
# ----------------------------------------------------------------------------
st.markdown("""
<div class="stage">
    <span class="stage-num">2</span>
    <b>FREQUENCY CONTENT</b> — where the motor-imagery signal lives
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
Motor imagery produces changes in the <b>mu (8-13 Hz)</b> and <b>beta (13-30 Hz)</b> rhythms
over sensorimotor cortex. Both bands are highlighted below on the average across channels.
</div>
""", unsafe_allow_html=True)

fft = np.abs(np.fft.rfft(x, axis=-1))
freqs = np.fft.rfftfreq(rec.n_timepoints, d=1 / rec.sfreq)
spec_mean = fft.mean(axis=0)

mask = freqs < 50
fig_fft = go.Figure()
fig_fft.add_trace(go.Scatter(
    x=freqs[mask], y=spec_mean[mask],
    fill="tozeroy", fillcolor="rgba(99,102,241,0.15)",
    line=dict(color="#6366F1", width=2),
))
fig_fft.add_vrect(x0=8, x1=13, fillcolor="rgba(239,68,68,0.15)",
                  annotation_text="mu (8-13 Hz)", annotation_position="top left")
fig_fft.add_vrect(x0=13, x1=30, fillcolor="rgba(16,185,129,0.15)",
                  annotation_text="beta (13-30 Hz)", annotation_position="top left")
fig_fft.update_layout(
    title="Average spectrum across channels",
    xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
    height=280, template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
)
st.plotly_chart(fig_fft, use_container_width=True)


# ----------------------------------------------------------------------------
# Section 3 — Model prediction (baseline) + attention + layer-lens
# ----------------------------------------------------------------------------
st.markdown("""
<div class="stage">
    <span class="stage-num">3</span>
    <b>BASELINE PREDICTION</b> — cross-subject transformer, no neuromodulator conditioning
</div>
""", unsafe_allow_html=True)

with torch.no_grad():
    # Full forward pass with all extras
    logits_base, cls_feat_base, attentions, layer_feats = base_model(
        x_tensor,
        subject_id=None,
        return_features=True,
        return_attention=True,
        return_layer_features=True,
    )
    probs_base = torch.softmax(logits_base, dim=-1)[0].numpy()

fig_base = go.Figure()
colors = ["#EF4444" if class_names[i] == label_true else "#94A3B8"
          for i in range(len(class_names))]
fig_base.add_trace(go.Bar(
    x=class_names, y=probs_base * 100,
    marker_color=colors,
    text=[f"{p*100:.1f}%" for p in probs_base],
    textposition="outside",
))
pred_idx = int(np.argmax(probs_base))
correct = pred_idx == int(y_true)
outcome = "✅ correct" if correct else "❌ wrong"
fig_base.update_layout(
    title=f"Baseline predicted: <b>{class_names[pred_idx]}</b> ({outcome}, true={label_true})",
    yaxis_title="Probability (%)",
    yaxis_range=[0, 105],
    height=320, template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
)
st.plotly_chart(fig_base, use_container_width=True)


# ----------------------------------------------------------------------------
# Section 3b — Attention heatmap (what time regions the model uses)
# ----------------------------------------------------------------------------
st.markdown("""
<div class="stage">
    <span class="stage-num">3b</span>
    <b>ATTENTION MAP</b> — where does the transformer look during classification?
</div>
""", unsafe_allow_html=True)

# Each attention tensor is (batch, n_heads, n_tokens, n_tokens).
# Token 0 is CLS, tokens 1..n_patches are time patches.
# We want the CLS row: "how much does the class token attend to each time patch?"
# per layer, per head.

n_layers = len(attentions)
n_heads = attentions[0].shape[1]
n_patches = attentions[0].shape[-1] - 1  # exclude CLS

# Convert time patches back to their center time in seconds
patch_stride = 16  # from config
patch_time = 32
patch_centers_samples = np.arange(n_patches) * patch_stride + patch_time / 2
patch_centers_seconds = patch_centers_samples / rec.sfreq

st.markdown(f"""
<div class="explanation">
Each row below is one <b>transformer layer</b>. The <b>4 columns are attention heads</b> (different feature "specialists").
Colors show how much the <b>classification token (CLS)</b> attends to each time patch of the input.
Brighter = more attention. Motor imagery activity is expected between 0.5-3.5 s post-cue —
if the model uses this window, it has learned meaningful physiological features.
</div>
""", unsafe_allow_html=True)

# Build a compact grid: layers × heads, each cell is a 1D heatmap of attention to patches
fig_attn = make_subplots(
    rows=n_layers, cols=n_heads,
    subplot_titles=[f"Head {h}" for h in range(n_heads)] * n_layers,
    shared_xaxes=True, shared_yaxes=True,
    vertical_spacing=0.08, horizontal_spacing=0.03,
)
for layer_idx in range(n_layers):
    a = attentions[layer_idx][0]  # (n_heads, n_tokens, n_tokens)
    for head_idx in range(n_heads):
        # CLS token's attention to each time patch (drop CLS→CLS self-attn)
        cls_attn = a[head_idx, 0, 1:].numpy()  # (n_patches,)
        fig_attn.add_trace(
            go.Heatmap(
                z=cls_attn.reshape(1, -1),
                x=patch_centers_seconds,
                colorscale="Viridis",
                showscale=False,
                zmin=0, zmax=cls_attn.max() * 1.05 if cls_attn.max() > 0 else 1,
            ),
            row=layer_idx + 1, col=head_idx + 1,
        )
fig_attn.update_yaxes(showticklabels=False)
fig_attn.update_xaxes(title_text="Time (s)", row=n_layers)
for r in range(n_layers):
    fig_attn.update_yaxes(title_text=f"Layer {r+1}", row=r+1, col=1)
fig_attn.update_layout(
    height=140 * n_layers + 60, template="plotly_white",
    margin=dict(l=60, r=20, t=40, b=40),
    title="CLS token attention to time patches",
)
st.plotly_chart(fig_attn, use_container_width=True)


# ----------------------------------------------------------------------------
# Section 3c — Logit Lens: prediction confidence at each layer
# ----------------------------------------------------------------------------
st.markdown("""
<div class="stage">
    <span class="stage-num">3c</span>
    <b>LOGIT LENS</b> — at which depth does the model "make up its mind"?
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
We apply the final classifier to the CLS features at <b>every layer</b> — not just the last one.
This shows the trajectory of the model's belief as it deepens. If confidence rises steadily, features
are being progressively refined. If final-layer only shows commitment, earlier layers are still
doing generic feature extraction.
</div>
""", unsafe_allow_html=True)

# Compute per-layer probabilities via the same classifier head
with torch.no_grad():
    layer_probs = []
    for feat in layer_feats:
        logits_l = base_model.head(feat)
        layer_probs.append(torch.softmax(logits_l, dim=-1)[0].numpy())
    # Append the final (post-norm) prediction for completeness
    # (already included as last layer feature, so no extra needed)

fig_lens = go.Figure()
x_layers = list(range(1, len(layer_probs) + 1))
for cls_idx, name in enumerate(class_names):
    trace_color = "#EF4444" if name == label_true else "#94A3B8"
    fig_lens.add_trace(go.Scatter(
        x=x_layers,
        y=[p[cls_idx] * 100 for p in layer_probs],
        mode="lines+markers",
        line=dict(color=trace_color, width=3 if name == label_true else 1.5),
        marker=dict(size=10 if name == label_true else 6),
        name=name + (" (true)" if name == label_true else ""),
    ))
fig_lens.add_hline(y=25, line_dash="dash", line_color="#CBD5E1",
                   annotation_text="chance", annotation_position="top left")
fig_lens.update_layout(
    title="Per-layer class probabilities (applied classifier head at each depth)",
    xaxis_title="Transformer layer",
    yaxis_title="Probability (%)",
    yaxis_range=[0, 100],
    height=320, template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
)
st.plotly_chart(fig_lens, use_container_width=True)


# ----------------------------------------------------------------------------
# Section 4 — Neuromodulator-conditioned prediction
# ----------------------------------------------------------------------------
st.markdown("""
<div class="stage">
    <span class="stage-num">4</span>
    <b>NEUROMODULATOR-CONDITIONED PREDICTION</b> — FiLM-modulated decoder
</div>
""", unsafe_allow_html=True)

from brainnn.core.state import BrainState
state = BrainState(acetylcholine=ach, norepinephrine=ne, dopamine=da)

with torch.no_grad():
    logits_cond, gamma, beta = decoder(x_tensor, subject_id=None, neuromod=state,
                                        return_gamma_beta=True)
    probs_cond = torch.softmax(logits_cond, dim=-1)[0].numpy()

col_left, col_right = st.columns([2, 1])

with col_left:
    fig_cond = go.Figure()
    fig_cond.add_trace(go.Bar(
        x=class_names, y=probs_base * 100,
        marker_color="#94A3B8", name="Baseline",
        opacity=0.6,
    ))
    fig_cond.add_trace(go.Bar(
        x=class_names, y=probs_cond * 100,
        marker_color="#10B981", name="Conditioned",
    ))
    pred_cond = int(np.argmax(probs_cond))
    correct_cond = pred_cond == int(y_true)
    outcome_cond = "✅ correct" if correct_cond else "❌ wrong"
    fig_cond.update_layout(
        title=f"Conditioned predicted: <b>{class_names[pred_cond]}</b> ({outcome_cond})",
        barmode="group", yaxis_title="Probability (%)",
        yaxis_range=[0, 105], height=320, template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_cond, use_container_width=True)

with col_right:
    # Derived quantities
    st.markdown("**Derived quantities:**")
    st.markdown(f"- **Precision** (ACh + NE): `{state.precision:.3f}`")
    st.markdown(f"- **Value gate** (DA - fatigue): `{state.value_gate:.3f}`")

    # Interpretation
    if state.precision > 0.7:
        p_label = "🔍 crisp attention"
    elif state.precision > 0.4:
        p_label = "🌫 moderate precision"
    else:
        p_label = "😴 scattered attention"

    if state.value_gate > 0.7:
        v_label = "⚡ strong gating"
    elif state.value_gate > 0.4:
        v_label = "🌊 mild gating"
    else:
        v_label = "🌙 weak gating"

    st.info(f"{p_label}\n\n{v_label}")

    st.markdown("**FiLM parameters:**")
    st.markdown(f"- gamma range: `[{gamma.min():.3f}, {gamma.max():.3f}]`")
    st.markdown(f"- beta range: `[{beta.min():.3f}, {beta.max():.3f}]`")

# Note about untrained FiLM
if not (_root / "checkpoints" / "eeg_transformer_subj1-8.pt").exists():
    st.warning("⚠️ FiLM head is untrained and zero-initialized — conditioned prediction will "
                "match baseline. This demonstrates the *architecture* only.")


# ----------------------------------------------------------------------------
# Section 4b — Focused vs tired: side-by-side conditioning comparison
# ----------------------------------------------------------------------------
st.markdown("""
<div class="stage">
    <span class="stage-num">4b</span>
    <b>PHARMACOLOGICAL DISSOCIATION</b> — focused user vs tired user
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
The same EEG trial, decoded under two different inferred neuromodulator states:
<b>focused</b> (high ACh, balanced NE, moderate DA) vs <b>tired</b> (low everything + high fatigue).
This is the falsifiability test suggested to Peter Dayan — a cholinergic vs dopaminergic
manipulation should produce distinct decoder outputs on identical neural input.
Watch how the prediction shifts as adaptation state changes.
</div>
""", unsafe_allow_html=True)

# Two contrasting states
state_focused = BrainState(
    acetylcholine=0.85, norepinephrine=0.55, dopamine=0.7, fatigue=0.05,
)
state_tired = BrainState(
    acetylcholine=0.25, norepinephrine=0.25, dopamine=0.2, fatigue=0.75,
)

with torch.no_grad():
    logits_focused = decoder(x_tensor, subject_id=None, neuromod=state_focused)
    logits_tired = decoder(x_tensor, subject_id=None, neuromod=state_tired)
    probs_focused = torch.softmax(logits_focused, dim=-1)[0].numpy()
    probs_tired = torch.softmax(logits_tired, dim=-1)[0].numpy()

col_focus, col_tired = st.columns(2)

with col_focus:
    st.markdown(f"""
    <div style="text-align:center; background:#f0fdf4; border:1px solid #10B981;
                border-radius:8px; padding:8px 4px; margin-bottom:8px;">
        <b style="color:#065F46">🎯 FOCUSED</b><br>
        <span style="font-size:11px; color:#065F46">
            ACh={state_focused.acetylcholine:.2f} · NE={state_focused.norepinephrine:.2f} ·
            DA={state_focused.dopamine:.2f} · fatigue={state_focused.fatigue:.2f}
        </span><br>
        <span style="font-size:11px; color:#065F46">
            precision={state_focused.precision:.2f}, value_gate={state_focused.value_gate:.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)
    pred_f = int(np.argmax(probs_focused))
    ok_f = "✅" if pred_f == int(y_true) else "❌"
    fig_f = go.Figure()
    fig_f.add_trace(go.Bar(
        x=class_names, y=probs_focused * 100,
        marker_color=["#10B981" if class_names[i] == label_true else "#94A3B8"
                       for i in range(len(class_names))],
        text=[f"{p*100:.1f}%" for p in probs_focused],
        textposition="outside",
    ))
    fig_f.update_layout(
        title=f"pred: <b>{class_names[pred_f]}</b> {ok_f}",
        yaxis_range=[0, 105], height=280, template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_f, use_container_width=True)

with col_tired:
    st.markdown(f"""
    <div style="text-align:center; background:#fef2f2; border:1px solid #EF4444;
                border-radius:8px; padding:8px 4px; margin-bottom:8px;">
        <b style="color:#7F1D1D">😴 TIRED</b><br>
        <span style="font-size:11px; color:#7F1D1D">
            ACh={state_tired.acetylcholine:.2f} · NE={state_tired.norepinephrine:.2f} ·
            DA={state_tired.dopamine:.2f} · fatigue={state_tired.fatigue:.2f}
        </span><br>
        <span style="font-size:11px; color:#7F1D1D">
            precision={state_tired.precision:.2f}, value_gate={state_tired.value_gate:.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)
    pred_t = int(np.argmax(probs_tired))
    ok_t = "✅" if pred_t == int(y_true) else "❌"
    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(
        x=class_names, y=probs_tired * 100,
        marker_color=["#EF4444" if class_names[i] == label_true else "#94A3B8"
                       for i in range(len(class_names))],
        text=[f"{p*100:.1f}%" for p in probs_tired],
        textposition="outside",
    ))
    fig_t.update_layout(
        title=f"pred: <b>{class_names[pred_t]}</b> {ok_t}",
        yaxis_range=[0, 105], height=280, template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_t, use_container_width=True)

# Delta chart — how much does conditioning shift each class?
delta = (probs_focused - probs_tired) * 100
fig_delta = go.Figure()
fig_delta.add_trace(go.Bar(
    x=class_names, y=delta,
    marker_color=["#10B981" if d > 0 else "#EF4444" for d in delta],
    text=[f"{d:+.1f}pp" for d in delta],
    textposition="outside",
))
fig_delta.add_hline(y=0, line_color="#94A3B8")
fig_delta.update_layout(
    title="Difference: focused - tired (percentage points per class)",
    height=240, template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
)
st.plotly_chart(fig_delta, use_container_width=True)


# ----------------------------------------------------------------------------
# Section 5 — LOSO results summary (if available)
# ----------------------------------------------------------------------------
st.markdown("""
<div class="stage">
    <span class="stage-num">5</span>
    <b>CROSS-SUBJECT TRANSFER RESULTS</b> — leave-one-subject-out evaluation
</div>
""", unsafe_allow_html=True)

loso_path = _root / "checkpoints" / "loso_results.json"
if loso_path.exists():
    with open(loso_path) as f:
        summary = json.load(f)

    st.markdown(f"""
    <div class="explanation">
    Model trained on 8 subjects, evaluated <b>zero-shot</b> (subject_id=None) on the held-out 9th.
    Chance for 4-class = 25%. Any performance above chance means the model
    learned generalizable neural patterns, not just per-subject idiosyncrasies.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean zero-shot acc", f"{summary['zero_shot_mean']*100:.1f}%",
              f"{(summary['zero_shot_mean']-0.25)*100:+.1f} vs chance")
    c2.metric("Best fold", f"{summary['zero_shot_max']*100:.1f}%")
    c3.metric("Worst fold", f"{summary['zero_shot_min']*100:.1f}%")
    c4.metric("Std dev", f"{summary['zero_shot_std']*100:.1f}%")

    per_fold = summary.get("per_fold", [])
    if per_fold:
        fig_loso = go.Figure()
        fig_loso.add_trace(go.Bar(
            x=[f"S{f['subject']}" for f in per_fold],
            y=[f["acc"] * 100 for f in per_fold],
            marker_color=["#10B981" if f["acc"] > 0.25 else "#EF4444" for f in per_fold],
            text=[f"{f['acc']*100:.1f}%" for f in per_fold],
            textposition="outside",
        ))
        fig_loso.add_hline(y=25, line_dash="dash", line_color="#94A3B8",
                            annotation_text="chance (25%)", annotation_position="top left")
        fig_loso.update_layout(
            title="Zero-shot accuracy per held-out subject",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, max(50, max(f["acc"] for f in per_fold) * 100 + 10)],
            height=320, template="plotly_white",
        )
        st.plotly_chart(fig_loso, use_container_width=True)
else:
    st.info("LOSO results not yet available. Run `python scripts/run_full_loso.py` "
             "to produce `checkpoints/loso_results.json`.")


# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:12px; padding:20px 0;">
    Built on BNCI2014_001 (Brunner et al. 2008) via MOABB.
    Neuromodulator framework: Yu &amp; Dayan (2005).
    Architecture: convolutional patch embed → 3-layer transformer → FiLM conditioning.<br>
    Code: <a href="https://github.com/umutakarsu/brainnn">github.com/umutakarsu/brainnn</a>
</div>
""", unsafe_allow_html=True)
