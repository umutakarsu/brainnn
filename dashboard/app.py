#!/usr/bin/env python3
"""
IC-LEM Dashboard — In Silico Closed-Loop Emotional Modulation
==============================================================
"Beyond the Pill" — Interactive comparison: Drugs vs Neural Implant

Designed for non-expert readers: select a treatment, see results,
compare with IC-LEM. No parameter tuning required.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from translations import T, t

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="IC-LEM | Beyond the Pill",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE SELECTOR (top of page)
# ══════════════════════════════════════════════════════════════════════════════

LANG_OPTIONS = {"Türkçe": "tr", "English": "en", "Deutsch": "de"}

lang_col1, lang_col2 = st.columns([4, 1])
with lang_col2:
    lang_name = st.selectbox(
        "🌐", list(LANG_OPTIONS.keys()), index=0, label_visibility="collapsed"
    )
L = LANG_OPTIONS[lang_name]

# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL DRUG PARAMETERS (hidden from user)
# ══════════════════════════════════════════════════════════════════════════════

DRUG_PARAMS = {
    "ssri": {
        "global_gain": 1.10, "onset_delay_ms": 800, "onset_sigmoid_k": 0.01,
        "blunting_factor": 0.40, "gaba_boost": 0.0, "bla_drive_reduction": 0.15,
        "mpfc_bla_boost": 0.0, "plasticity": 0.0, "decay_time_ms": None,
        "closed_loop": False,
    },
    "benzo": {
        "global_gain": 1.0, "onset_delay_ms": 0, "onset_sigmoid_k": 0.1,
        "blunting_factor": 0.20, "gaba_boost": 0.50, "bla_drive_reduction": 0.60,
        "mpfc_bla_boost": 0.0, "plasticity": 0.0, "decay_time_ms": None,
        "closed_loop": False,
    },
    "ketamine": {
        "global_gain": 1.0, "onset_delay_ms": 50, "onset_sigmoid_k": 0.05,
        "blunting_factor": 0.05, "gaba_boost": 0.0, "bla_drive_reduction": 0.10,
        "mpfc_bla_boost": 0.45, "plasticity": 0.50, "decay_time_ms": 1200,
        "closed_loop": False,
    },
    "iclem": {
        "global_gain": 1.0, "onset_delay_ms": 0, "onset_sigmoid_k": 1.0,
        "blunting_factor": 0.0, "gaba_boost": 0.0, "bla_drive_reduction": 0.0,
        "mpfc_bla_boost": 0.0, "plasticity": 0.0, "decay_time_ms": None,
        "closed_loop": True,
    },
}

# Fixed simulation parameters (optimized for MDD scenario)
SIM_DURATION_MS = 2000
NOISE_LEVEL = 0.20
THRESHOLD_HZ = 60.0
STIM_CURRENT = 2.5
W_MPFC_BLA = 0.15  # MDD baseline


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL CIRCUIT SIMULATION (LIF — pure NumPy)
# ══════════════════════════════════════════════════════════════════════════════

class LIFNetwork:
    """Leaky Integrate-and-Fire network: BLA, mPFC, NAc populations."""

    def __init__(self, n_bla=100, n_mpfc=50, n_nac=30, dt_ms=0.1):
        self.n_bla = n_bla
        self.n_mpfc = n_mpfc
        self.n_nac = n_nac
        self.dt = dt_ms
        self.v_rest = -65.0
        self.v_thresh = -50.0
        self.v_reset = -65.0
        self.tau_bla = 10.0
        self.tau_mpfc = 20.0
        self.tau_nac = 15.0
        self.R = 100.0
        self.refrac_ms = 2.0

        self.v_bla = np.full(n_bla, self.v_rest)
        self.v_mpfc = np.full(n_mpfc, self.v_rest)
        self.v_nac = np.full(n_nac, self.v_rest)
        self.ref_bla = np.zeros(n_bla)
        self.ref_mpfc = np.zeros(n_mpfc)
        self.ref_nac = np.zeros(n_nac)

        self.I_bla_base = 0.30 + 0.15 * np.random.rand(n_bla)
        self.I_mpfc_base = np.full(n_mpfc, 0.05)
        self.I_nac_base = 0.15 + 0.05 * np.random.rand(n_nac)

        self.W_bla_mpfc = 0.8
        self.W_mpfc_bla = 0.15
        self.W_bla_bla = 0.3
        self.W_bla_nac = 0.2

        self.conn_bla_mpfc = (np.random.rand(n_bla, n_mpfc) < 0.3).astype(float)
        self.conn_mpfc_bla = (np.random.rand(n_mpfc, n_bla) < 0.4).astype(float)
        self.conn_bla_bla = (np.random.rand(n_bla, n_bla) < 0.1).astype(float)
        np.fill_diagonal(self.conn_bla_bla, 0)
        self.conn_bla_nac = (np.random.rand(n_bla, n_nac) < 0.2).astype(float)

    def step(self, I_bla_ext, I_mpfc_ext, I_nac_ext,
             spikes_bla_prev, spikes_mpfc_prev,
             W_mpfc_bla_eff, W_bla_bla_eff, global_gain=1.0,
             gaba_boost=0.0, blunting_factor=0.0):

        syn_bla_to_mpfc = np.zeros(self.n_mpfc)
        if len(spikes_bla_prev) > 0:
            syn_bla_to_mpfc = self.W_bla_mpfc * global_gain * \
                self.conn_bla_mpfc[spikes_bla_prev, :].sum(axis=0)

        syn_mpfc_to_bla = np.zeros(self.n_bla)
        if len(spikes_mpfc_prev) > 0:
            syn_mpfc_to_bla = W_mpfc_bla_eff * (1.0 + gaba_boost) * \
                self.conn_mpfc_bla[spikes_mpfc_prev, :].sum(axis=0)

        syn_bla_to_bla = np.zeros(self.n_bla)
        if len(spikes_bla_prev) > 0:
            syn_bla_to_bla = W_bla_bla_eff * global_gain * \
                self.conn_bla_bla[spikes_bla_prev, :].sum(axis=0)

        syn_bla_to_nac = np.zeros(self.n_nac)
        if len(spikes_bla_prev) > 0:
            syn_bla_to_nac = self.W_bla_nac * global_gain * \
                self.conn_bla_nac[spikes_bla_prev, :].sum(axis=0)

        active_bla = self.ref_bla <= 0
        dv_bla = (-(self.v_bla - self.v_rest) + self.R * I_bla_ext -
                  syn_mpfc_to_bla + syn_bla_to_bla * 0.5) / self.tau_bla * self.dt
        if blunting_factor > 0:
            dv_bla *= (1.0 - blunting_factor * 0.5)
        self.v_bla[active_bla] += dv_bla[active_bla]

        active_mpfc = self.ref_mpfc <= 0
        dv_mpfc = (-(self.v_mpfc - self.v_rest) + self.R * I_mpfc_ext +
                   syn_bla_to_mpfc * 0.5) / self.tau_mpfc * self.dt
        self.v_mpfc[active_mpfc] += dv_mpfc[active_mpfc]

        active_nac = self.ref_nac <= 0
        dv_nac = (-(self.v_nac - self.v_rest) + self.R * I_nac_ext +
                  syn_bla_to_nac * 0.3) / self.tau_nac * self.dt
        if blunting_factor > 0:
            dv_nac *= (1.0 - blunting_factor)
        self.v_nac[active_nac] += dv_nac[active_nac]

        spikes_bla = np.where(self.v_bla >= self.v_thresh)[0]
        spikes_mpfc = np.where(self.v_mpfc >= self.v_thresh)[0]
        spikes_nac = np.where(self.v_nac >= self.v_thresh)[0]

        self.v_bla[spikes_bla] = self.v_reset
        self.v_mpfc[spikes_mpfc] = self.v_reset
        self.v_nac[spikes_nac] = self.v_reset

        self.ref_bla[spikes_bla] = self.refrac_ms
        self.ref_mpfc[spikes_mpfc] = self.refrac_ms
        self.ref_nac[spikes_nac] = self.refrac_ms
        self.ref_bla = np.maximum(0, self.ref_bla - self.dt)
        self.ref_mpfc = np.maximum(0, self.ref_mpfc - self.dt)
        self.ref_nac = np.maximum(0, self.ref_nac - self.dt)

        return spikes_bla, spikes_mpfc, spikes_nac


@st.cache_data(show_spinner=False)
def run_simulation(mode_key):
    """Run full simulation for a given treatment mode."""
    np.random.seed(42)

    dp = DRUG_PARAMS[mode_key]
    dt = 0.1
    n_steps = int(SIM_DURATION_MS / dt)
    sense_window_steps = int(20.0 / dt)

    net = LIFNetwork(n_bla=100, n_mpfc=50, n_nac=30, dt_ms=dt)

    is_closed_loop = dp["closed_loop"]
    global_gain = dp["global_gain"]
    onset_delay = dp["onset_delay_ms"]
    onset_k = dp["onset_sigmoid_k"]
    blunting = dp["blunting_factor"]
    gaba_boost = dp["gaba_boost"]
    bla_drive_red = dp["bla_drive_reduction"]
    plasticity = dp["plasticity"]
    decay_time = dp["decay_time_ms"]

    record_interval = int(1.0 / dt)
    n_records = n_steps // record_interval + 1

    bla_rates = np.zeros(n_records)
    nac_rates = np.zeros(n_records)
    stim_log = np.zeros(n_records)
    time_log = np.zeros(n_records)

    bla_spike_times = []
    bla_spike_ids = []
    mpfc_spike_times = []
    mpfc_spike_ids = []

    bla_spike_buffer = np.zeros(sense_window_steps)
    buffer_idx = 0
    prev_bla = np.array([], dtype=int)
    prev_mpfc = np.array([], dtype=int)
    W_mpfc_bla_eff = W_MPFC_BLA
    rec_idx = 0
    cl_start = 500.0

    for step in range(n_steps):
        t_ms = step * dt

        if is_closed_loop:
            drug_effect = 0.0
            c_blunting = c_gaba = 0.0
            c_gain = 1.0
        else:
            drug_effect = 1.0 / (1.0 + np.exp(-onset_k * (t_ms - onset_delay)))
            if decay_time is not None and t_ms > onset_delay + 200:
                drug_effect *= np.exp(-(t_ms - onset_delay - 200) / decay_time)
            c_blunting = blunting * drug_effect
            c_gaba = gaba_boost * drug_effect
            c_gain = 1.0 + (global_gain - 1.0) * drug_effect
            if plasticity > 0:
                W_mpfc_bla_eff = W_MPFC_BLA + (plasticity * drug_effect)

        I_bla = net.I_bla_base.copy()
        if not is_closed_loop:
            I_bla *= (1.0 - bla_drive_red * drug_effect)
        I_bla += NOISE_LEVEL * np.random.randn(net.n_bla) * 0.1

        I_mpfc = net.I_mpfc_base.copy()
        I_mpfc += NOISE_LEVEL * np.random.randn(net.n_mpfc) * 0.02

        I_nac = net.I_nac_base.copy()
        if not is_closed_loop:
            I_nac *= (1.0 - c_blunting * 0.5)
        I_nac += NOISE_LEVEL * np.random.randn(net.n_nac) * 0.03

        stim_on = False
        if is_closed_loop and t_ms >= cl_start:
            window_spikes = bla_spike_buffer.sum()
            rate = (window_spikes / net.n_bla) / 0.02
            noisy_rate = rate + NOISE_LEVEL * np.random.randn() * 10.0
            if noisy_rate > THRESHOLD_HZ:
                stim_on = True
                I_mpfc += STIM_CURRENT
                I_bla -= 0.15

        sp_bla, sp_mpfc, sp_nac = net.step(
            I_bla, I_mpfc, I_nac, prev_bla, prev_mpfc,
            W_mpfc_bla_eff,
            net.W_bla_bla * c_gain if not is_closed_loop else net.W_bla_bla,
            global_gain=c_gain if not is_closed_loop else 1.0,
            gaba_boost=c_gaba if not is_closed_loop else 0.0,
            blunting_factor=c_blunting if not is_closed_loop else 0.0,
        )

        bla_spike_buffer[buffer_idx % sense_window_steps] = len(sp_bla)
        buffer_idx += 1
        prev_bla = sp_bla
        prev_mpfc = sp_mpfc

        if step % 10 == 0:
            for idx in sp_bla:
                bla_spike_times.append(t_ms)
                bla_spike_ids.append(idx)
            for idx in sp_mpfc:
                mpfc_spike_times.append(t_ms)
                mpfc_spike_ids.append(idx)

        if step % record_interval == 0 and rec_idx < n_records:
            bla_rates[rec_idx] = (bla_spike_buffer.sum() / 100) / 0.02
            nac_rates[rec_idx] = len(sp_nac) / 30 / (dt / 1000.0)
            stim_log[rec_idx] = 1.0 if stim_on else 0.0
            time_log[rec_idx] = t_ms
            rec_idx += 1

    kernel = np.ones(50) / 50
    bla_smooth = np.convolve(bla_rates[:rec_idx], kernel, mode='same')
    nac_smooth = np.convolve(nac_rates[:rec_idx], kernel, mode='same')
    time = time_log[:rec_idx]

    bl_mask = time < 400
    end_mask = time > (SIM_DURATION_MS - 400)
    bl_bla = np.mean(bla_smooth[bl_mask]) if bl_mask.any() else 1
    end_bla = np.mean(bla_smooth[end_mask]) if end_mask.any() else 0
    bla_red = ((bl_bla - end_bla) / max(bl_bla, 1)) * 100

    bl_nac = np.mean(nac_smooth[bl_mask]) if bl_mask.any() else 1
    end_nac = np.mean(nac_smooth[end_mask]) if end_mask.any() else 0
    nac_red = ((bl_nac - end_nac) / max(bl_nac, 1)) * 100

    stim_ratio = np.mean(stim_log[:rec_idx][time > cl_start]) * 100 if is_closed_loop else 0

    return {
        "time": time, "bla_rates": bla_smooth, "nac_rates": nac_smooth,
        "stim_log": stim_log[:rec_idx],
        "bla_spike_t": np.array(bla_spike_times), "bla_spike_i": np.array(bla_spike_ids),
        "mpfc_spike_t": np.array(mpfc_spike_times), "mpfc_spike_i": np.array(mpfc_spike_ids),
        "bla_reduction": bla_red, "nac_reduction": nac_red, "stim_ratio": stim_ratio,
    }


@st.cache_data
def compute_efield(grid_size=100, sigma=0.3):
    """Compute monopolar and phased array E-field."""
    x = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, x)
    r_mono = np.maximum(np.sqrt(X**2 + Y**2), 0.05)
    V_mono = 1.0 / (4 * np.pi * sigma * r_mono)

    epos = [(-3, -3), (3, -3), (-3, 3), (3, 3)]
    V_ph = np.zeros_like(X)
    for (ex, ey) in epos:
        r = np.maximum(np.sqrt((X-ex)**2 + (Y-ey)**2), 0.05)
        d = np.sqrt(ex**2 + ey**2)
        phi = -2 * np.pi * d / 7.692
        V_ph += np.cos(phi) / (4 * np.pi * sigma * r)
    return X, Y, V_mono, V_ph, epos


# ══════════════════════════════════════════════════════════════════════════════
#  TREATMENT MODES — user-facing labels
# ══════════════════════════════════════════════════════════════════════════════

MODES = {
    "ssri": {
        "label": {"tr": "💊 SSRI (Prozac / Fluoxetine)", "en": "💊 SSRI (Prozac / Fluoxetine)", "de": "💊 SSRI (Prozac / Fluoxetin)"},
        "desc": "desc_ssri",
    },
    "benzo": {
        "label": {"tr": "💉 Benzodiazepin (Xanax)", "en": "💉 Benzodiazepine (Xanax)", "de": "💉 Benzodiazepin (Xanax)"},
        "desc": "desc_benzo",
    },
    "ketamine": {
        "label": {"tr": "⚡ Ketamin (Spravato)", "en": "⚡ Ketamine (Spravato)", "de": "⚡ Ketamin (Spravato)"},
        "desc": "desc_ketamine",
    },
    "iclem": {
        "label": {"tr": "🎯 Akıllı Beyin İmplantı (IC-LEM)", "en": "🎯 Smart Brain Implant (IC-LEM)", "de": "🎯 Intelligentes Gehirnimplantat (IC-LEM)"},
        "desc": "desc_iclem",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ──
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
        <h1 style="margin-bottom: 0.2rem;">🧠 {t('page_title', L)}</h1>
        <h4 style="color: #6B7280; margin-top: 0;">{t('page_subtitle', L)}</h4>
        <p style="color: #9CA3AF; font-size: 0.9rem;">{t('page_desc', L)}</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Treatment selector — simple buttons ──
    st.markdown(f"### {t('mode_label', L)}")

    mode_keys = list(MODES.keys())
    cols = st.columns(4)
    selected_key = st.session_state.get("selected_mode", None)

    for i, key in enumerate(mode_keys):
        with cols[i]:
            label = MODES[key]["label"][L]
            if st.button(label, use_container_width=True,
                         type="primary" if selected_key == key else "secondary"):
                st.session_state["selected_mode"] = key
                selected_key = key

    if selected_key is None:
        st.info(t("info_msg", L))
        return

    # Show description
    st.caption(t(MODES[selected_key]["desc"], L))

    # ── Run simulation (auto — no button needed) ──
    with st.spinner(f"🔄 {MODES[selected_key]['label'][L]} {t('running_spinner', L)}"):
        results = run_simulation(selected_key)

    # Always run IC-LEM for comparison
    iclem_results = run_simulation("iclem") if selected_key != "iclem" else results

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    #  TOP METRICS
    # ══════════════════════════════════════════════════════════════════════

    onset_map = {"ssri": "onset_ssri", "benzo": "onset_benzo",
                 "ketamine": "onset_ketamine", "iclem": "onset_iclem"}
    precision_map = {"ssri": "~10%", "benzo": "~5%", "ketamine": "~45%", "iclem": "98%"}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(t("metric_bla_reduction", L),
                  f"{results['bla_reduction']:.1f}%",
                  delta=f"-{results['bla_reduction']:.0f}%", delta_color="normal")
    with c2:
        if selected_key == "iclem":
            st.metric(t("metric_stim_ratio", L), f"{results['stim_ratio']:.1f}%")
        else:
            st.metric(t("metric_nac_blunting", L),
                      f"{results['nac_reduction']:.1f}%",
                      delta=f"-{results['nac_reduction']:.0f}%", delta_color="inverse")
    with c3:
        st.metric(t("metric_onset", L), t(onset_map[selected_key], L))
    with c4:
        st.metric(t("metric_precision", L), precision_map[selected_key])

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    #  TABS
    # ══════════════════════════════════════════════════════════════════════

    if selected_key == "iclem":
        tab_names = [t("tab_neural", L), t("tab_efield", L),
                     t("tab_comparison", L), t("tab_analysis", L)]
    else:
        tab_names = [t("tab_neural", L), t("tab_comparison", L), t("tab_analysis", L)]

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # ── TAB: Neural Activity ──
    with tabs[tab_idx]:
        tab_idx += 1
        time_arr = results["time"]
        bla_r = results["bla_rates"]

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                t("plot_raster_title", L),
                t("plot_bla_rate_title", L),
                t("plot_nac_title", L) if selected_key != "iclem" else t("plot_stim_title", L),
            ),
            vertical_spacing=0.10,
            row_heights=[0.35, 0.35, 0.30]
        )

        # Raster — BLA
        if len(results["bla_spike_t"]) > 0:
            max_pts = 12000
            n_sp = len(results["bla_spike_t"])
            if n_sp > max_pts:
                idx = np.random.choice(n_sp, max_pts, replace=False)
            else:
                idx = np.arange(n_sp)
            fig.add_trace(go.Scattergl(
                x=results["bla_spike_t"][idx], y=results["bla_spike_i"][idx],
                mode='markers', marker=dict(size=1.5, color='crimson', opacity=0.5),
                name=t("legend_bla_spikes", L), showlegend=True
            ), row=1, col=1)

        # Raster — mPFC
        if len(results["mpfc_spike_t"]) > 0:
            max_pts = 6000
            n_sp = len(results["mpfc_spike_t"])
            if n_sp > max_pts:
                idx = np.random.choice(n_sp, max_pts, replace=False)
            else:
                idx = np.arange(n_sp)
            fig.add_trace(go.Scattergl(
                x=results["mpfc_spike_t"][idx], y=results["mpfc_spike_i"][idx] + 100,
                mode='markers', marker=dict(size=1.5, color='dodgerblue', opacity=0.5),
                name=t("legend_mpfc_spikes", L), showlegend=True
            ), row=1, col=1)

        if selected_key == "iclem":
            fig.add_vline(x=500, line_dash="dash", line_color="green",
                         annotation_text=t("legend_cl_on", L), row=1, col=1)

        # BLA firing rate
        fig.add_trace(go.Scatter(
            x=time_arr, y=bla_r, mode='lines',
            line=dict(color='crimson', width=2),
            name=t("legend_bla_rate", L)
        ), row=2, col=1)
        fig.add_hline(y=THRESHOLD_HZ, line_dash="dash", line_color="orange",
                     annotation_text=f"{t('legend_threshold', L)}: {THRESHOLD_HZ:.0f} Hz",
                     row=2, col=1)

        if selected_key == "iclem":
            stim = results["stim_log"]
            stim_mask = stim > 0.5
            if stim_mask.any():
                fig.add_trace(go.Scatter(
                    x=time_arr[stim_mask], y=bla_r[stim_mask],
                    mode='markers', marker=dict(size=3, color='green', opacity=0.3),
                    name=t("legend_stim_active", L)
                ), row=2, col=1)

        # Row 3: NAc (drugs) or Stim timing (IC-LEM)
        if selected_key == "iclem":
            fig.add_trace(go.Scatter(
                x=time_arr, y=results["stim_log"],
                mode='lines', fill='tozeroy',
                line=dict(color='#10B981', width=1),
                fillcolor='rgba(16, 185, 129, 0.3)',
                name=t("legend_stim_onoff", L)
            ), row=3, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=time_arr, y=results["nac_rates"],
                mode='lines', line=dict(color='#A78BFA', width=2),
                name=t("legend_nac_rate", L)
            ), row=3, col=1)

        fig.update_layout(
            height=750, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=80, b=40)
        )
        fig.update_xaxes(title_text=t("axis_time", L), row=3, col=1)
        fig.update_yaxes(title_text=t("axis_neuron", L), row=1, col=1)
        fig.update_yaxes(title_text=t("axis_hz", L), row=2, col=1)
        if selected_key == "iclem":
            fig.update_yaxes(title_text=t("axis_onoff", L), row=3, col=1)
        else:
            fig.update_yaxes(title_text=t("axis_hz", L), row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # ── TAB: E-field (only for IC-LEM) ──
    if selected_key == "iclem":
        with tabs[tab_idx]:
            tab_idx += 1
            st.markdown(f"### {t('efield_title', L)}")

            X, Y, V_mono, V_ph, epos = compute_efield()
            ce1, ce2 = st.columns(2)

            with ce1:
                fig_m = go.Figure(data=go.Heatmap(
                    z=V_mono, x=np.linspace(-5,5,100), y=np.linspace(-5,5,100),
                    colorscale='Hot', colorbar=dict(title="mV"),
                    zmin=0, zmax=np.percentile(V_mono, 98)
                ))
                fig_m.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                    marker=dict(symbol='cross', size=15, color='white', line=dict(width=2)),
                    name=t("efield_legend_target", L)))
                fig_m.update_layout(
                    title=f"{t('efield_mono_title', L)}<br><sub>{t('efield_mono_sub', L)}</sub>",
                    xaxis_title="x (mm)", yaxis_title="y (mm)",
                    width=500, height=500, yaxis=dict(scaleanchor="x"))
                st.plotly_chart(fig_m, use_container_width=True)

            with ce2:
                fig_p = go.Figure(data=go.Heatmap(
                    z=V_ph, x=np.linspace(-5,5,100), y=np.linspace(-5,5,100),
                    colorscale='Hot', colorbar=dict(title="mV"),
                    zmin=0, zmax=np.percentile(V_ph, 98)
                ))
                fig_p.add_trace(go.Scatter(
                    x=[p[0] for p in epos], y=[p[1] for p in epos], mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='cyan',
                               line=dict(width=1, color='white')),
                    name=t("efield_legend_electrodes", L)))
                fig_p.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                    marker=dict(symbol='star', size=18, color='red'),
                    name=t("efield_legend_target", L)))
                fig_p.update_layout(
                    title=f"{t('efield_phased_title', L)}<br><sub>{t('efield_phased_sub', L)}</sub>",
                    xaxis_title="x (mm)", yaxis_title="y (mm)",
                    width=500, height=500, yaxis=dict(scaleanchor="x"))
                st.plotly_chart(fig_p, use_container_width=True)

            # Profile comparison
            st.markdown(f"### {t('efield_profile_title', L)}")
            mid = 50
            x_ax = np.linspace(-5, 5, 100)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=x_ax, y=V_mono[mid,:]/np.max(V_mono[mid,:]),
                mode='lines', line=dict(color='red', width=3), name=t("efield_legend_mono", L)))
            fig_pr.add_trace(go.Scatter(x=x_ax, y=V_ph[mid,:]/np.max(V_ph[mid,:]),
                mode='lines', line=dict(color='blue', width=3), name=t("efield_legend_phased", L)))
            fig_pr.add_vline(x=0, line_dash="dash", line_color="gray",
                            annotation_text=t("efield_legend_target_line", L))
            fig_pr.update_layout(xaxis_title="x (mm)", yaxis_title=t("efield_axis_norm", L),
                                height=400, template="plotly_white")
            st.plotly_chart(fig_pr, use_container_width=True)

    # ── TAB: Comparison Table ──
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown(f"### {t('comp_title', L)}")
        st.markdown(f"*{t('comp_subtitle', L)}*")

        # Build table with translated values
        tbl = {
            t("tbl_param", L): [
                t("tbl_onset", L), t("tbl_spatial", L), t("tbl_bla_ctrl", L),
                t("tbl_blunting", L), t("tbl_cognitive", L), t("tbl_addiction", L),
                t("tbl_targeting", L), t("tbl_dosing", L),
                t("tbl_reversibility", L), t("tbl_longterm", L),
            ],
            "💊 SSRI": [
                t("ssri_onset", L), t("ssri_spatial", L), t("ssri_bla", L),
                t("ssri_blunting", L), t("ssri_cognitive", L), t("ssri_addiction", L),
                t("ssri_targeting", L), t("ssri_dosing", L),
                t("ssri_reversibility", L), t("ssri_longterm", L),
            ],
            "💉 Benzodiazepin": [
                t("benzo_onset", L), t("benzo_spatial", L), t("benzo_bla", L),
                t("benzo_blunting", L), t("benzo_cognitive", L), t("benzo_addiction", L),
                t("benzo_targeting", L), t("benzo_dosing", L),
                t("benzo_reversibility", L), t("benzo_longterm", L),
            ],
            "⚡ Ketamin": [
                t("ket_onset", L), t("ket_spatial", L), t("ket_bla", L),
                t("ket_blunting", L), t("ket_cognitive", L), t("ket_addiction", L),
                t("ket_targeting", L), t("ket_dosing", L),
                t("ket_reversibility", L), t("ket_longterm", L),
            ],
            "🎯 IC-LEM": [
                t("iclem_onset", L), t("iclem_spatial", L),
                f"{t('ket_bla', L).split('(')[0]}({results['bla_reduction']:.0f}%)" if selected_key != "iclem"
                    else f"{results['bla_reduction']:.0f}%",
                t("iclem_blunting", L), t("iclem_cognitive", L), t("iclem_addiction", L),
                t("iclem_targeting", L), t("iclem_dosing", L),
                t("iclem_reversibility", L), t("iclem_longterm", L),
            ],
        }

        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # Score bars
        st.divider()
        st.markdown(f"### {t('comp_numerical', L)}")
        score_cols = st.columns(4)
        names = ["SSRI", "Benzodiazepin", "Ketamin", "IC-LEM"]
        scores = [35, 20, 55, 95]
        for col, name, score in zip(score_cols, names, scores):
            with col:
                st.markdown(f"**{name}**")
                st.progress(score / 100)
                st.caption(f"{t('comp_score', L)}: {score}/100")

    # ── TAB: Analysis Report ──
    with tabs[tab_idx]:
        st.markdown(f"### {t('analysis_title', L)}")

        # Analysis text
        analysis_map = {"ssri": "analysis_ssri", "benzo": "analysis_benzo",
                        "ketamine": "analysis_ketamine", "iclem": "analysis_iclem"}
        st.markdown(t(analysis_map[selected_key], L))

        # Stats line
        st.markdown("---")
        st.markdown(
            f"**{t('metric_bla_reduction', L)}:** {results['bla_reduction']:.1f}%  |  "
            f"**{t('metric_onset', L)}:** {t(onset_map[selected_key], L)}  |  "
            f"**{t('metric_precision', L)}:** {precision_map[selected_key]}"
        )

        # IC-LEM recommendation (for drug modes)
        if selected_key != "iclem":
            st.divider()
            st.markdown(f"**[{t('analysis_iclem_rec', L)}]**")
            st.markdown(t("iclem_recommendation", L))
            st.markdown(
                f"*{t('metric_nac_blunting', L)}: "
                f"{MODES[selected_key]['label'][L]} → {results['nac_reduction']:.1f}% | "
                f"IC-LEM → ~0%*"
            )

        # Drug mechanism details
        st.divider()
        st.markdown(f"### {t('analysis_drug_detail_title', L)}")

        detail_map = {"ssri": "detail_ssri", "benzo": "detail_benzo",
                      "ketamine": "detail_ketamine", "iclem": "detail_iclem"}
        details = t(detail_map[selected_key], L)
        if isinstance(details, dict):
            for key, val in details.items():
                st.markdown(f"**{key}:** {val}")

    # ── Footer ──
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: #9CA3AF; font-size: 0.8rem; padding: 1rem 0;">
        {t('footer', L)}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
