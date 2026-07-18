"""Phase detection + narration (§5, and the SCREEN 3 text from §4).

Given a generated timeline, detect the four phases and return narration strings.
Tone follows §6 strictly: mechanism language, second-person descriptive, no
shame, no prescription, no success/failure framing. The narration's whole job is
the §4 insight — that the crash below baseline is the mechanism, it is
predictable, and it is not a moral failure.

Phase-narration templates live here (not copy.py) because they are filled by the
detection logic below — see DECISIONS.md D4. copy.py holds the static chrome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Per-preset "key insight" sentence, appended to the VULNERABLE phase. These are
# the genuinely-interesting mechanisms that distinguish each loop.
INSIGHT_TEXT: dict[str, str] = {
    "anticipation_exceeds_reward": (
        "For this loop the anticipation peak is higher than the peak at "
        "consumption. The wanting is bigger than the liking — that gap is the "
        "engine of the loop."
    ),
    "novelty_escalation": (
        "This loop escalates through novelty: the peak needs something newer "
        "each time to reach the same height (the Coolidge effect), and a "
        "prolactin surge right after the peak flattens things fast. That is why "
        "it climbs, not why you are weak."
    ),
    "withdrawal_not_pleasure": (
        "After the receptors adjust, most of what a hit does is relieve the "
        "withdrawal the last hit created. The curve is chasing baseline, not "
        "reaching for pleasure."
    ),
    "no_satiety_signal": (
        "The spikes are small but there is no satiety signal to end them — no "
        "natural full point. That missing stop cue is the mechanism, not a lack "
        "of discipline."
    ),
    "gaba_rebound_anxiety": (
        "The calm early on is GABA. Hours later glutamate rebounds past where it "
        "started — that overshoot is the anxiety the next morning. Same curve, "
        "predictable timing."
    ),
    "borrowed_not_created": (
        "This one blocks adenosine rather than adding dopamine, so no energy is "
        "created — it is borrowed. The crash is the deferred fatigue arriving, "
        "on schedule."
    ),
    "generic_dopamine_loop": (
        "The shape is the general dopamine loop: a rise, a peak, and a dip below "
        "where you started. The dip is the part that gets blamed on willpower."
    ),
}

# The four-phase templates (§4). {n}/{m} are filled with the vulnerable window.
PHASE_TEMPLATES: dict[str, str] = {
    "trigger": (
        "Cue fires. Dopamine rises before you've done anything — this is "
        "wanting, not liking."
    ),
    "peak": "The spike. Shorter than you remember it being.",
    "crash": (
        "Dopamine drops below where you started. This is the part that gets "
        "blamed on willpower."
    ),
    "vulnerable": (
        "Hours {n}–{m}: your baseline is at its lowest. This is when the "
        "next trigger has the most power. The loop is not a character flaw — "
        "it's this curve."
    ),
}

PHASE_LABELS: dict[str, str] = {
    "trigger": "Trigger",
    "peak": "Peak",
    "crash": "Crash",
    "vulnerable": "Vulnerable",
}


@dataclass
class Phase:
    key: str
    label: str
    t_start: float   # hours
    t_end: float     # hours
    t_marker: float  # hours — where to anchor the annotation
    text: str


def _first_true(mask: np.ndarray) -> int | None:
    idx = np.nonzero(mask)[0]
    return int(idx[0]) if idx.size else None


def _last_true(mask: np.ndarray) -> int | None:
    idx = np.nonzero(mask)[0]
    return int(idx[-1]) if idx.size else None


def detect_phases(timeline: dict, key_insight: str | None = None) -> list[Phase]:
    """Detect the four phases from a simulate_timeline() output.

    - trigger:    onset -> peak (the anticipatory rise)
    - peak:       the global maximum
    - crash:      first below-baseline crossing after the peak -> the trough
    - vulnerable: the trough -> baseline recovery (or window end if still low)
    """
    t = np.asarray(timeline["t"], dtype=float)
    dopamine = np.asarray(timeline["dopamine"], dtype=float)
    baseline = float(timeline["effective_baseline"])

    peak_idx = int(np.argmax(dopamine))
    peak_t = float(t[peak_idx])

    # Below-baseline region strictly after the peak.
    after_peak = np.arange(t.size) > peak_idx
    below = (dopamine < baseline) & after_peak

    crash_start_i = _first_true(below)
    crash_start_t = float(t[crash_start_i]) if crash_start_i is not None else peak_t

    # Trough: minimum after the peak.
    if after_peak.any():
        rel = np.where(after_peak, dopamine, np.inf)
        trough_idx = int(np.argmin(rel))
    else:
        trough_idx = int(np.argmin(dopamine))
    trough_t = float(t[trough_idx])

    recovery_i = _last_true(below)
    recovery_t = float(t[recovery_i]) if recovery_i is not None else float(t[-1])

    n = int(round(crash_start_t))
    m = int(round(recovery_t))
    if m <= n:
        m = n + 1

    vulnerable_text = PHASE_TEMPLATES["vulnerable"].format(n=n, m=m)
    if key_insight and key_insight in INSIGHT_TEXT:
        vulnerable_text = f"{vulnerable_text} {INSIGHT_TEXT[key_insight]}"

    return [
        Phase("trigger", PHASE_LABELS["trigger"], 0.0, peak_t, peak_t * 0.5,
              PHASE_TEMPLATES["trigger"]),
        Phase("peak", PHASE_LABELS["peak"], max(0.0, peak_t - 0.2), peak_t + 0.2,
              peak_t, PHASE_TEMPLATES["peak"]),
        Phase("crash", PHASE_LABELS["crash"], crash_start_t, trough_t, trough_t,
              PHASE_TEMPLATES["crash"]),
        Phase("vulnerable", PHASE_LABELS["vulnerable"], trough_t, recovery_t,
              (trough_t + recovery_t) / 2.0, vulnerable_text),
    ]


def narrate(preset: dict, timeline: dict) -> list[Phase]:
    """Convenience: detect phases using the preset's key insight."""
    return detect_phases(timeline, key_insight=preset.get("key_insight"))
