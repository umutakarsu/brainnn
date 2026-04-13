"""SynapseFlow — Interactive Neural Simulation Dashboard (v2: Visual Storytelling)"""

import sys
sys.path.insert(0, "/Users/umutakarsu/brainnn/src")
sys.path.insert(0, "/Users/umutakarsu/brainnn")

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from brainnn.calibration.calibration_logic import CalibrationEngine, CorticalZone
from brainnn.synesthesia.synesthesia_engine import (
    SynesthesiaEngine, SynesthesiaConfig, SpikeTrainEncoder,
)
from brainnn.calibration.neuroplasticity import NeuroplasticityController
from simulation.sandbox import SimulationSandbox

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SynapseFlow",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .step-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 4px solid #10B981;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin: 12px 0;
    }
    .step-number {
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
    .arrow-down {
        text-align: center;
        font-size: 28px;
        color: #10B981;
        margin: 4px 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
    }
    .explanation {
        background: #eff6ff;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 14px;
        color: #1e40af;
        margin: 8px 0 16px 0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Audio generators
# ---------------------------------------------------------------------------
SR = 44100
DURATION = 1.0

def make_audio(name: str) -> tuple[np.ndarray, str, str]:
    """Returns (audio_array, short_description, emoji)."""
    t = np.arange(int(SR * DURATION)) / SR
    if name == "Pure Tone (La - 440 Hz)":
        audio = np.sin(2 * np.pi * 440 * t)
        return audio / np.max(np.abs(audio)), "A single 440 Hz sine wave — the purest possible sound", "🎵"
    elif name == "Chord (Bass + Mid + Treble)":
        audio = 0.4 * np.sin(2*np.pi*220*t) + 0.35 * np.sin(2*np.pi*440*t) + 0.25 * np.sin(2*np.pi*880*t)
        return audio / np.max(np.abs(audio)), "Three notes layered: 220 Hz (bass) + 440 Hz (mid) + 880 Hz (treble)", "🎶"
    elif name == "Rising Sweep (20 Hz to 8 kHz)":
        freq = np.linspace(20, 8000, len(t))
        audio = np.sin(2 * np.pi * np.cumsum(freq / SR))
        return audio / np.max(np.abs(audio)), "Frequency rises from deep bass to high treble over 1 second", "📈"
    elif name == "Heartbeat":
        audio = np.zeros_like(t)
        for beat_start in np.arange(0, DURATION, 0.8):
            idx = int(beat_start * SR)
            if idx + 2000 < len(audio):
                env = np.exp(-np.arange(2000) / 300)
                audio[idx:idx+2000] += 0.8 * env * np.sin(2*np.pi*60*np.arange(2000)/SR)
            idx2 = idx + int(0.2 * SR)
            if idx2 + 1500 < len(audio):
                env2 = np.exp(-np.arange(1500) / 200)
                audio[idx2:idx2+1500] += 0.5 * env2 * np.sin(2*np.pi*80*np.arange(1500)/SR)
        return audio / (np.max(np.abs(audio)) + 1e-8), "Simulated heartbeat rhythm at ~75 BPM", "💓"
    else:
        audio = np.random.randn(len(t))
        return audio / np.max(np.abs(audio)), "Random noise — all frequencies at once", "📻"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🧠 SynapseFlow")
st.markdown("### How a Brain Chip Turns Sound Into Color")
st.markdown("""
<div class="explanation">
<b>What is this?</b> This is a simulation of <b>artificial synesthesia</b> — a brain implant that lets you
<em>see sounds as colors</em>. The chip picks up audio, breaks it into frequency bands, converts each band
to a color (bass→red, mid→green, treble→blue), and stimulates visual cortex electrodes so the brain
perceives those colors. This dashboard lets you watch each step of that process.
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Controls — simple, inline, no sidebar
# ---------------------------------------------------------------------------
st.markdown("### Configure Your Simulation")

col_a, col_b, col_c = st.columns(3)

with col_a:
    audio_choice = st.selectbox(
        "🔊 Choose a sound",
        ["Pure Tone (La - 440 Hz)", "Chord (Bass + Mid + Treble)",
         "Rising Sweep (20 Hz to 8 kHz)", "Heartbeat", "White Noise"],
        index=1,
    )

with col_b:
    onboarding_day = st.select_slider(
        "📅 Brain adaptation day",
        options=list(range(1, 15)),
        value=1,
        help="Day 1 = brain just got the implant (10% power). Day 14 = fully adapted (100% power).",
    )

with col_c:
    master_gain = st.select_slider(
        "🔧 Signal strength",
        options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        value=1.0,
    )

run_sim = st.button("▶  Run Simulation", type="primary", use_container_width=True)

if not run_sim:
    st.divider()
    st.markdown("### 👆 Press the button above to start")
    st.markdown("""
    <div class="explanation">
    <b>Tip:</b> Try changing the <b>adaptation day</b> — on Day 1 the brain barely perceives the new colors
    (10% intensity). By Day 14, it's fully adapted (100%). This models real neuroplasticity:
    the brain needs time to learn a new sense.
    </div>
    """, unsafe_allow_html=True)

    # Preview: onboarding curve
    ctrl = NeuroplasticityController()
    schedule = ctrl.default_schedule(14)

    fig = go.Figure()
    days = list(schedule.keys())
    intensities = [schedule[d] * 100 for d in days]

    fig.add_trace(go.Bar(
        x=days, y=intensities,
        marker_color=[
            "#EF4444" if d == onboarding_day else (
                "#10B981" if d <= onboarding_day else "#e2e8f0"
            ) for d in days
        ],
        text=[f"{v:.0f}%" for v in intensities],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.add_annotation(
        x=onboarding_day, y=schedule[onboarding_day] * 100 + 8,
        text=f"YOU ARE HERE<br>Day {onboarding_day}: {schedule[onboarding_day]:.0%}",
        showarrow=True, arrowhead=2, arrowcolor="#EF4444",
        font=dict(color="#EF4444", size=12, family="Arial Black"),
    )
    fig.update_layout(
        title=dict(text="Brain Adaptation Schedule (14-Day Onboarding)", font=dict(size=16)),
        xaxis_title="Day", yaxis_title="Stimulation Power (%)",
        yaxis_range=[0, 115], height=380,
        template="plotly_white", showlegend=False,
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
audio, audio_desc, audio_emoji = make_audio(audio_choice)

progress = st.progress(0, text="Starting simulation...")
time.sleep(0.2)

config = SynesthesiaConfig(fft_window=256, hop_size=128, master_gain=master_gain)
sandbox = SimulationSandbox(synesthesia_config=config, seed=42)
spike_encoder = SpikeTrainEncoder(max_rate_hz=300)

progress.progress(20, text="Step 1/4: Calibrating 1024 electrodes...")
time.sleep(0.3)

result = sandbox.run(audio, onboarding_day=onboarding_day)

progress.progress(50, text="Step 2/4: Breaking sound into frequency bands...")
time.sleep(0.3)

progress.progress(75, text="Step 3/4: Converting colors to brain pulses...")
spike_trains = []
for frame in result.rgb_frames:
    spike_trains.append(spike_encoder.encode(frame, duration_ms=20.0))
time.sleep(0.2)

progress.progress(100, text="Step 4/4: Simulating brain response — Done!")
time.sleep(0.3)
progress.empty()

rgb = result.rgb_frames
intensity = sandbox.plasticity.schedule.get(onboarding_day, 1.0)
cal = result.calibration

# ---------------------------------------------------------------------------
# PIPELINE VISUALIZATION — Step by step
# ---------------------------------------------------------------------------

# ======================== STEP 1: THE SOUND ========================
st.divider()
st.markdown(f"""
<div class="step-box">
    <span class="step-number">1</span>
    <b>THE SOUND</b> — What the microphone picks up
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="explanation">
{audio_emoji} <b>{audio_choice}</b>: {audio_desc}.
The chip's microphone captures this audio and needs to convert it into something the visual cortex can understand.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    # Waveform
    show_n = min(int(0.05 * SR), len(audio))
    t_ms = np.arange(show_n) / SR * 1000
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=t_ms, y=audio[:show_n],
        line=dict(color="#6366F1", width=1.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
    ))
    fig_wave.update_layout(
        title="Audio Waveform (first 50 milliseconds)",
        xaxis_title="Time (ms)", yaxis_title="Amplitude",
        height=280, template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_wave, use_container_width=True)

with col2:
    # Frequency spectrum with colored bands
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/SR)
    mask = freqs < 10000
    fig_fft = go.Figure()
    # Color-coded bands
    for flo, fhi, color, name, alpha in [
        (20, 250, "#EF4444", "Bass → RED", 0.25),
        (250, 4000, "#10B981", "Mid → GREEN", 0.25),
        (4000, 10000, "#3B82F6", "Treble → BLUE", 0.25),
    ]:
        band_mask = (freqs >= flo) & (freqs < fhi)
        fig_fft.add_trace(go.Scatter(
            x=freqs[band_mask], y=spectrum[band_mask],
            fill="tozeroy", fillcolor=f"rgba({','.join(str(int(c)) for c in [239,68,68] if color=='#EF4444')},{alpha})" if color == "#EF4444"
                else f"rgba(16,185,129,{alpha})" if color == "#10B981"
                else f"rgba(59,130,246,{alpha})",
            line=dict(color=color, width=1.5),
            name=name,
        ))
    fig_fft.update_layout(
        title="Frequency Bands → Color Mapping",
        xaxis_title="Frequency (Hz)", yaxis_title="Energy",
        xaxis_type="log", height=280, template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.12, font=dict(size=11)),
    )
    st.plotly_chart(fig_fft, use_container_width=True)


# ======================== ARROW ========================
st.markdown('<div class="arrow-down">⬇</div>', unsafe_allow_html=True)


# ======================== STEP 2: THE COLOR ========================
st.markdown(f"""
<div class="step-box">
    <span class="step-number">2</span>
    <b>THE COLOR</b> — Sound converted to visual information
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="explanation">
Each audio frame is split into 3 frequency bands. Bass energy becomes <b style="color:#EF4444">Red</b>,
mid becomes <b style="color:#10B981">Green</b>, treble becomes <b style="color:#3B82F6">Blue</b>.
Together they form a color that represents what the brain will "see."
On <b>Day {onboarding_day}</b>, intensity is at <b>{intensity:.0%}</b> —
{"the brain is just starting to learn this new sense." if intensity < 0.4 else "the brain is well-adapted to perceive these colors." if intensity > 0.7 else "adaptation is progressing nicely."}
</div>
""", unsafe_allow_html=True)

# Color strip as an image (much faster than 300 bar traces)
n_show = min(250, len(rgb))
strip_data = np.zeros((40, n_show, 3), dtype=np.uint8)
for i in range(n_show):
    r, g, b = rgb[i]
    strip_data[:, i] = [int(r * 255), int(g * 255), int(b * 255)]

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

fig_strip, ax = plt.subplots(figsize=(14, 1.8))
ax.imshow(strip_data, aspect="auto", extent=[0, n_show, 0, 1])
ax.set_xlabel("Frame (time →)", fontsize=11)
ax.set_yticks([])
ax.set_title("Color Stream: What the visual cortex would perceive", fontsize=12, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig_strip)
plt.close()

# RGB channel breakdown
col1, col2 = st.columns([2, 1])
with col1:
    fig_rgb = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=["🔴 Red Channel (Bass: 20-250 Hz)",
                                            "🟢 Green Channel (Mid: 250-4000 Hz)",
                                            "🔵 Blue Channel (Treble: 4000+ Hz)"])
    for i, (color, name) in enumerate([
        ("#EF4444", "Red"), ("#10B981", "Green"), ("#3B82F6", "Blue")
    ]):
        fig_rgb.add_trace(go.Scatter(
            y=rgb[:n_show, i], line=dict(color=color, width=1.5),
            fill="tozeroy", showlegend=False,
        ), row=i+1, col=1)
    fig_rgb.update_layout(height=350, template="plotly_white",
                          margin=dict(l=50, r=20, t=30, b=30))
    fig_rgb.update_xaxes(title_text="Frame", row=3, col=1)
    st.plotly_chart(fig_rgb, use_container_width=True)

with col2:
    # Big color swatch — average color
    avg_r, avg_g, avg_b = np.mean(rgb, axis=0)
    hex_avg = f"#{int(avg_r*255):02x}{int(avg_g*255):02x}{int(avg_b*255):02x}"
    st.markdown(f"""
    <div style="text-align:center; margin-top:20px;">
        <p style="font-size:13px; color:#64748b; margin-bottom:8px;">Average perceived color</p>
        <div style="width:160px; height:160px; background:{hex_avg};
             border-radius:50%; margin:0 auto; border:4px solid #e2e8f0;
             box-shadow: 0 0 40px {hex_avg}60;"></div>
        <p style="font-size:20px; font-weight:700; margin-top:12px; color:#1e293b;">{hex_avg}</p>
        <p style="font-size:12px; color:#64748b;">
            R: {avg_r:.0%} &nbsp; G: {avg_g:.0%} &nbsp; B: {avg_b:.0%}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Energy pie
    fig_pie = go.Figure(go.Pie(
        labels=["Bass (Red)", "Mid (Green)", "Treble (Blue)"],
        values=[avg_r, avg_g, avg_b],
        marker_colors=["#EF4444", "#10B981", "#3B82F6"],
        hole=0.5, textinfo="label+percent",
        textfont=dict(size=11),
    ))
    fig_pie.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ======================== ARROW ========================
st.markdown('<div class="arrow-down">⬇</div>', unsafe_allow_html=True)


# ======================== STEP 3: THE BRAIN CHIP ========================
st.markdown(f"""
<div class="step-box">
    <span class="step-number">3</span>
    <b>THE BRAIN CHIP</b> — 1024 electrodes mapped to brain regions
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="explanation">
Before stimulation begins, the chip runs a <b>calibration</b>: it tests each of its 1024 electrodes to
find out which brain region each one can activate.
Result: <b>{len(cal.active_electrodes)}</b> electrodes are healthy and selective
(<b>{cal.overall_quality:.0%}</b> quality). The colored map below shows which electrode connects to which brain zone.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Zone map with clear labels
    zone_info = {
        CorticalZone.VISUAL_V1: ("V1 — Primary Visual", "#EF4444", 0),
        CorticalZone.VISUAL_V4: ("V4 — Color Vision", "#F59E0B", 1),
        CorticalZone.AUDITORY_A1: ("A1 — Primary Auditory", "#10B981", 2),
        CorticalZone.SOMATOSENSORY_S1: ("S1 — Touch/Feeling", "#3B82F6", 3),
        CorticalZone.MOTOR_M1: ("M1 — Movement", "#8B5CF6", 4),
        CorticalZone.PREFRONTAL: ("PFC — Thinking/Planning", "#EC4899", 5),
        CorticalZone.UNKNOWN: ("Unknown / Dead", "#d1d5db", 6),
    }

    zone_grid = np.full((32, 32), 6)
    for ep in cal.electrode_profiles:
        r, c = ep.grid_position
        zone_grid[r, c] = zone_info.get(ep.zone, ("?", "#999", 6))[2]

    fig_zone = go.Figure(go.Heatmap(
        z=zone_grid,
        colorscale=[
            [0, "#EF4444"], [0.167, "#F59E0B"], [0.333, "#10B981"],
            [0.5, "#3B82F6"], [0.667, "#8B5CF6"], [0.833, "#EC4899"], [1.0, "#d1d5db"],
        ],
        zmin=0, zmax=6, showscale=False,
    ))
    # Add zone labels as annotations
    zone_positions = {
        0: (5, 8, "V1\nVision"), 1: (5, 24, "V4\nColor"),
        2: (12, 16, "A1\nHearing"), 3: (19, 8, "S1\nTouch"),
        4: (19, 24, "M1\nMotor"), 5: (28, 16, "PFC\nThinking"),
    }
    for idx, (row, col, label) in zone_positions.items():
        fig_zone.add_annotation(x=col, y=row, text=label,
                                showarrow=False, font=dict(color="white", size=11, family="Arial Black"))

    fig_zone.update_layout(
        title=dict(text="Electrode → Brain Region Map (32×32 grid)", font=dict(size=14)),
        xaxis_title="Column", yaxis_title="Row",
        height=420, template="plotly_white",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_zone, use_container_width=True)

with col2:
    # Impedance map — healthy vs dead
    imp_grid = np.zeros((32, 32))
    for ep in cal.electrode_profiles:
        r, c = ep.grid_position
        imp_grid[r, c] = ep.impedance_ohm / 1e6

    fig_imp = go.Figure(go.Heatmap(
        z=imp_grid, colorscale="RdYlGn_r",
        zmin=0, zmax=2.5,
        colorbar=dict(title=dict(text="MΩ", side="right")),
    ))
    fig_imp.update_layout(
        title=dict(text="Electrode Health (Impedance) — Green = Good, Red = Bad", font=dict(size=14)),
        xaxis_title="Column", yaxis_title="Row",
        height=420, template="plotly_white",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# Quick stats
st.markdown(f"""
| Metric | Value | Meaning |
|--------|-------|---------|
| Active Electrodes | **{len(cal.active_electrodes)} / 1024** | {len(cal.active_electrodes)} electrodes are working properly |
| Dead Electrodes | **{1024 - len(cal.active_electrodes)}** | Too high impedance — these are skipped |
| Calibration Quality | **{cal.overall_quality:.1%}** | Overall health score of the implant |
| Processing Latency | **{result.latency_ms:.1f} ms** | Must stay under 10ms for the brain to sync sound+color |
""")


# ======================== ARROW ========================
st.markdown('<div class="arrow-down">⬇</div>', unsafe_allow_html=True)


# ======================== STEP 4: ELECTRICAL PULSES ========================
st.markdown(f"""
<div class="step-box">
    <span class="step-number">4</span>
    <b>THE PULSES</b> — Colors encoded as electrical brain stimulation
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="explanation">
Each color frame is converted to a <b>spike train</b> — a pattern of tiny electrical pulses sent to electrodes.
Brighter color = more pulses per second. The visual cortex reads these pulses and interprets them as color perception.
Use the slider to explore different moments in the audio.
</div>
""", unsafe_allow_html=True)

frame_idx = st.slider("🎬 Explore frame by frame", 0, len(spike_trains) - 1,
                       len(spike_trains) // 4, help="Each frame = ~6ms of audio")

spikes = spike_trains[frame_idx]
rgb_val = rgb[frame_idx]

col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    # Input color
    r_val, g_val, b_val = rgb_val
    hex_color = f"#{int(r_val*255):02x}{int(g_val*255):02x}{int(b_val*255):02x}"
    st.markdown(f"""
    <div style="text-align:center;">
        <p style="font-size:12px; color:#64748b;">INPUT COLOR</p>
        <div style="width:90px; height:90px; background:{hex_color};
             border-radius:12px; margin:0 auto; border:2px solid #333;
             box-shadow: 0 0 20px {hex_color}80;"></div>
        <p style="margin-top:6px; font-size:11px; color:#64748b;">Frame {frame_idx}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Spike raster plot
    fig_spike = go.Figure()
    colors = ["#EF4444", "#10B981", "#3B82F6"]
    labels = ["Red → V1 (Vision)", "Green → V4 (Color)", "Blue → A1 (Hearing)"]

    for ch in range(3):
        spike_times = np.where(spikes[ch] == 1)[0]
        if len(spike_times) > 0:
            fig_spike.add_trace(go.Scatter(
                x=spike_times, y=[ch] * len(spike_times),
                mode="markers",
                marker=dict(symbol="line-ns", size=18, line_width=2, color=colors[ch]),
                name=f"{labels[ch]}  ({len(spike_times)} pulses)",
            ))

    fig_spike.update_layout(
        title=f"Electrical Pulse Pattern (20ms window)",
        xaxis_title="Time within window (ms)",
        yaxis=dict(tickvals=[0, 1, 2], ticktext=["V1 (Vision)", "V4 (Color)", "A1 (Hearing)"]),
        height=220, template="plotly_white",
        margin=dict(l=100, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
        xaxis_range=[-0.5, 20.5],
    )
    st.plotly_chart(fig_spike, use_container_width=True)

with col3:
    # Spike rates
    st.markdown("""<div style="text-align:center;">
        <p style="font-size:12px; color:#64748b;">PULSE RATES</p>
    </div>""", unsafe_allow_html=True)
    for ch, (name, color) in enumerate(zip(["Bass", "Mid", "Treble"], colors)):
        rate = np.sum(spikes[ch]) / 0.02
        max_rate = 300
        pct = min(100, rate / max_rate * 100)
        st.markdown(f"""
        <div style="margin:8px 0;">
            <span style="font-size:11px; color:{color}; font-weight:700;">{name}</span>
            <div style="background:#e2e8f0; border-radius:4px; height:12px; margin-top:2px;">
                <div style="background:{color}; width:{pct}%; height:100%; border-radius:4px;"></div>
            </div>
            <span style="font-size:11px; color:#64748b;">{rate:.0f} Hz</span>
        </div>
        """, unsafe_allow_html=True)


# ======================== ARROW ========================
st.markdown('<div class="arrow-down">⬇</div>', unsafe_allow_html=True)


# ======================== STEP 5: BRAIN RESPONSE ========================
st.markdown(f"""
<div class="step-box">
    <span class="step-number">5</span>
    <b>THE BRAIN'S RESPONSE</b> — How the cortex reacts to stimulation
</div>
""", unsafe_allow_html=True)

day_label = {1: "just implanted — barely notices", 4: "starting to sense patterns",
             7: "forming new neural pathways", 10: "getting comfortable",
             14: "fully adapted — perceives colors naturally"}
closest_label = day_label.get(onboarding_day,
    "early adaptation" if onboarding_day < 5 else
    "mid adaptation" if onboarding_day < 10 else "nearly adapted")

st.markdown(f"""
<div class="explanation">
<b>Day {onboarding_day} of 14</b> — Stimulation at <b>{intensity:.0%}</b> power.
The brain is {closest_label}.
The chart shows how strongly the cortex responds to each frame of stimulation.
Higher response = the brain is successfully interpreting the signal as color.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    fig_cortex = go.Figure()
    fig_cortex.add_trace(go.Scatter(
        y=result.cortical_response,
        line=dict(color="#F59E0B", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.15)",
        name="Cortical Response",
    ))
    fig_cortex.update_layout(
        title=f"Cortical Response Over Time (Day {onboarding_day})",
        xaxis_title="Frame", yaxis_title="Response Strength (a.u.)",
        height=300, template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_cortex, use_container_width=True)

with col2:
    mean_resp = np.mean(result.cortical_response)
    max_resp = np.max(result.cortical_response)

    # Visual gauge
    gauge_color = "#EF4444" if mean_resp < 2 else "#F59E0B" if mean_resp < 5 else "#10B981"
    gauge_label = "Weak" if mean_resp < 2 else "Moderate" if mean_resp < 5 else "Strong"

    st.markdown(f"""
    <div style="text-align:center; margin-top:20px;">
        <p style="font-size:13px; color:#64748b;">Brain Response Level</p>
        <p style="font-size:48px; font-weight:800; color:{gauge_color}; margin:0;">{mean_resp:.1f}</p>
        <p style="font-size:16px; font-weight:600; color:{gauge_color};">{gauge_label}</p>
        <p style="font-size:11px; color:#94a3b8; margin-top:12px;">
            Peak: {max_resp:.1f}<br>
            Day {onboarding_day} / 14<br>
            Power: {intensity:.0%}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================== FINAL: ONBOARDING COMPARISON ========================
st.divider()
st.markdown("### 📊 How Adaptation Changes Everything")
st.markdown("""
<div class="explanation">
Compare the same sound processed at different stages of brain adaptation.
On Day 1 the colors are faint (brain can't handle full power yet).
By Day 14 the same sound produces vivid, rich colors.
</div>
""", unsafe_allow_html=True)

comparison_days = [1, 4, 7, 10, 14]
fig_compare = make_subplots(rows=5, cols=1, vertical_spacing=0.02,
                            subplot_titles=[f"Day {d} — {sandbox.plasticity.schedule.get(d, 1):.0%} power"
                                            for d in comparison_days])

for idx, day in enumerate(comparison_days):
    res = sandbox.run(audio, onboarding_day=day)
    n = min(200, len(res.rgb_frames))
    frames = res.rgb_frames[:n]

    # Create a heatmap-style color strip using 3 rows for R, G, B
    for ch, color in enumerate(["rgb(239,68,68)", "rgb(16,185,129)", "rgb(59,130,246)"]):
        fig_compare.add_trace(go.Scatter(
            x=list(range(n)), y=frames[:, ch],
            fill="tozeroy",
            line=dict(width=0),
            fillcolor=f"rgba({','.join(color[4:-1].split(','))},0.5)",
            showlegend=False,
        ), row=idx + 1, col=1)

    fig_compare.update_yaxes(range=[0, 1.1], showticklabels=False, row=idx + 1, col=1)

fig_compare.update_layout(
    height=600, template="plotly_white",
    margin=dict(l=50, r=20, t=30, b=30),
    showlegend=False,
)
fig_compare.update_xaxes(title_text="Frame", row=5, col=1)
st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:12px; padding:20px 0;">
    SynapseFlow Neural Simulation Sandbox — Built by Umut Akarsu<br>
    Simulating artificial synesthesia without hardware
</div>
""", unsafe_allow_html=True)
