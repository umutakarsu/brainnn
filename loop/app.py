"""Loop — map a behavioural loop, see the mechanism (§4 four-screen flow).

Run: streamlit run app.py

No PyTorch, no ML, no database, no auth, no payments (§8 "Do NOT"). Just the
curve model + Streamlit. All six presets get identical UI treatment (§3.2).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from loop import copy as C
from loop import simulate as sim
from loop.narrate import narrate
from loop.presets import all_selectable, get_preset, PRESETS
from loop.copy import PRESET_SUBTITLES

# --- Palette (theme-neutral, colour-blind-safe pairing) --------------------
COLOR_DOPAMINE = "#2E86DE"   # blue
COLOR_CORTISOL = "#E67E22"   # amber
COLOR_BASELINE = "#8395A7"   # grey
COLOR_VULN = "rgba(231, 76, 60, 0.10)"   # faint red shade for vulnerable window
PHASE_COLORS = {
    "trigger": "#8E44AD",
    "peak": "#2E86DE",
    "crash": "#C0392B",
    "vulnerable": "#E67E22",
}

st.set_page_config(page_title=C.APP_NAME, page_icon="🔁", layout="centered")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("screen", 1)
    ss.setdefault("preset_key", None)
    ss.setdefault("frequency", "daily")
    ss.setdefault("duration", "months")
    ss.setdefault("intensity", 6)


def _goto(screen: int) -> None:
    st.session_state.screen = screen


def _episode_count() -> float:
    return sim.episodes_from_frequency_duration(
        st.session_state.frequency, st.session_state.duration
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def timeline_figure(timeline: dict, phases: list) -> go.Figure:
    t = timeline["t"]
    fig = go.Figure()

    # Vulnerable window shading (the key insight, §4 ④).
    vuln = next((p for p in phases if p.key == "vulnerable"), None)
    if vuln is not None:
        fig.add_vrect(x0=vuln.t_start, x1=vuln.t_end, fillcolor=COLOR_VULN,
                      line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=t, y=timeline["baseline"], name=C.t("legend_baseline"),
        line=dict(color=COLOR_BASELINE, width=1.5, dash="dash"),
        hovertemplate="baseline<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=timeline["dopamine"], name=C.t("legend_dopamine"),
        line=dict(color=COLOR_DOPAMINE, width=3),
        hovertemplate="dopamine %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=timeline["cortisol"], name=C.t("legend_cortisol"),
        line=dict(color=COLOR_CORTISOL, width=2, dash="dot"),
        hovertemplate="cortisol %{y:.2f}<extra></extra>",
    ))

    # Phase markers. Stagger label heights so fast loops (trigger/peak nearly
    # coincident) don't collide.
    label_heights = [1.10, 1.02, 1.10, 1.02]
    for i, p in enumerate(phases):
        fig.add_annotation(
            x=p.t_marker, y=label_heights[i % len(label_heights)], yref="paper",
            showarrow=False, text=f"<b>{p.label}</b>",
            font=dict(color=PHASE_COLORS[p.key], size=12),
        )
        fig.add_vline(x=p.t_marker, line=dict(color=PHASE_COLORS[p.key],
                      width=1, dash="dot"), opacity=0.4)

    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=56, b=10),
        xaxis_title=C.t("axis_time"), yaxis_title=C.t("axis_level"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
        hovermode="x unified",
    )
    return fig


def recovery_figure(rec: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=1.0, line=dict(color=COLOR_BASELINE, width=1.5, dash="dash"),
                  annotation_text="intact baseline")
    fig.add_trace(go.Scatter(
        x=rec["days"], y=rec["baseline"], name="baseline",
        line=dict(color=COLOR_DOPAMINE, width=3), fill="tozeroy",
        fillcolor="rgba(46,134,222,0.08)",
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Days with no repetition", yaxis_title="Baseline",
        yaxis=dict(range=[0, 1.1]), showlegend=False,
    )
    return fig


def delay_figure(cmp: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=cmp["baseline"], line=dict(color=COLOR_BASELINE, width=1.2,
                  dash="dash"))
    fig.add_trace(go.Scatter(x=cmp["t"], y=cmp["now"], name="now",
                  line=dict(color=COLOR_DOPAMINE, width=3)))
    fig.add_trace(go.Scatter(x=cmp["t"], y=cmp["delayed"],
                  name=f"delayed {cmp['delay_hours']:.0f}h",
                  line=dict(color=COLOR_CORTISOL, width=3, dash="dot")))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=C.t("axis_time"), yaxis_title=C.t("axis_level"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
    )
    return fig


def frequency_bar(cmp: dict) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Current frequency", "Half the frequency"],
        y=[cmp["current_baseline"], cmp["reduced_baseline"]],
        marker_color=[COLOR_CORTISOL, COLOR_DOPAMINE],
        text=[f"{cmp['current_baseline']:.2f}", f"{cmp['reduced_baseline']:.2f}"],
        textposition="outside",
    ))
    fig.add_hline(y=1.0, line=dict(color=COLOR_BASELINE, width=1.2, dash="dash"),
                  annotation_text="intact baseline")
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 1.1], title="Resulting baseline"), showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Persistent chrome (disclaimer + resources always visible, §6)
# ---------------------------------------------------------------------------

def footer() -> None:
    st.divider()
    st.caption("⚕️ " + C.t("disclaimer"))
    st.caption("📞 " + C.t("resources"))
    with st.expander(C.t("about_title")):
        st.markdown(C.t("about_body"))


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def screen_pick() -> None:
    st.title(C.APP_NAME)
    st.markdown(f"**{C.TAGLINE}**")
    st.subheader(C.t("s1_title"))
    st.caption(C.t("s1_subtitle"))

    options = all_selectable()
    cols = st.columns(2)
    for i, (key, label) in enumerate(options):
        with cols[i % 2]:
            subtitle = PRESET_SUBTITLES.get(key, "")
            if st.button(f"**{label}**\n\n{subtitle}", key=f"preset_{key}",
                         use_container_width=True):
                st.session_state.preset_key = key
                _goto(2)
                st.rerun()


def screen_customise() -> None:
    st.subheader(C.t("s2_title"))
    st.caption(C.t("s2_subtitle"))

    freq_opts = C.frequency_options()
    dur_opts = C.duration_options()

    freq_labels = [lab for _, lab in freq_opts]
    freq_keys = [k for k, _ in freq_opts]
    cur_freq = st.session_state.frequency
    freq_idx = freq_keys.index(cur_freq) if cur_freq in freq_keys else 1
    freq_choice = st.radio(C.t("q_frequency"), freq_labels, index=freq_idx)
    st.session_state.frequency = freq_keys[freq_labels.index(freq_choice)]

    dur_labels = [lab for _, lab in dur_opts]
    dur_keys = [k for k, _ in dur_opts]
    cur_dur = st.session_state.duration
    dur_idx = dur_keys.index(cur_dur) if cur_dur in dur_keys else 1
    dur_choice = st.radio(C.t("q_duration"), dur_labels, index=dur_idx)
    st.session_state.duration = dur_keys[dur_labels.index(dur_choice)]

    st.session_state.intensity = st.slider(
        C.t("q_intensity"), 1, 10, st.session_state.intensity,
        help=f"1 = {C.t('intensity_min')}, 10 = {C.t('intensity_max')}",
    )

    c1, c2 = st.columns(2)
    if c1.button(C.t("cta_back"), use_container_width=True):
        _goto(1); st.rerun()
    if c2.button(C.t("cta_next"), type="primary", use_container_width=True):
        _goto(3); st.rerun()


def screen_narrated() -> None:
    preset = get_preset(st.session_state.preset_key)
    st.subheader(C.t("s3_title"))
    st.caption(f"{preset['label']} — {C.t('s3_subtitle')}")

    timeline = sim.simulate_timeline(
        preset, intensity=st.session_state.intensity, n_episodes=_episode_count()
    )
    phases = narrate(preset, timeline)
    st.plotly_chart(timeline_figure(timeline, phases), use_container_width=True)

    for p in phases:
        st.markdown(
            f"<span style='color:{PHASE_COLORS[p.key]}'>●</span> "
            f"**{p.label}** — {p.text}", unsafe_allow_html=True,
        )

    drift = timeline["drift"]
    if drift < -0.01:
        st.caption(
            f"After the repetitions you described, the baseline this curve is "
            f"measured against has drifted down by {abs(drift) * 100:.0f}%. "
            f"That is the allostatic shift — the same peak now lands on lower ground."
        )

    c1, c2 = st.columns(2)
    if c1.button(C.t("cta_back"), use_container_width=True):
        _goto(2); st.rerun()
    if c2.button(C.t("cta_whatif"), type="primary", use_container_width=True):
        _goto(4); st.rerun()


def screen_whatif() -> None:
    preset = get_preset(st.session_state.preset_key)
    n = _episode_count()
    intensity = st.session_state.intensity
    st.subheader(C.t("s4_title"))
    st.caption(C.t("s4_subtitle"))

    # The recovery curve first — the most motivating (§4).
    st.markdown(f"**{C.t('whatif_recovery')}**")
    st.plotly_chart(recovery_figure(sim.simulate_recovery(preset, n, days=30)),
                    use_container_width=True)
    st.caption(C.t("whatif_recovery_note"))

    st.markdown(f"**{C.t('whatif_delay')}**")
    st.plotly_chart(delay_figure(sim.compare_delay(preset, intensity, n)),
                    use_container_width=True)
    st.caption(C.t("whatif_delay_note"))

    st.markdown(f"**{C.t('whatif_frequency')}**")
    st.plotly_chart(frequency_bar(sim.compare_frequency(preset, intensity, n)),
                    use_container_width=True)
    st.caption(C.t("whatif_frequency_note"))

    c1, c2 = st.columns(2)
    if c1.button(C.t("cta_back"), use_container_width=True):
        _goto(3); st.rerun()
    if c2.button(C.t("cta_restart"), use_container_width=True):
        _goto(1); st.session_state.preset_key = None; st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    screen = st.session_state.screen
    if screen == 1 or st.session_state.preset_key is None:
        screen_pick()
    elif screen == 2:
        screen_customise()
    elif screen == 3:
        screen_narrated()
    elif screen == 4:
        screen_whatif()
    footer()


if __name__ == "__main__":
    main()
