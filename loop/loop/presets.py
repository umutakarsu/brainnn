"""Preset library — the neurochemical profiles that ARE the product (§3).

Each preset is an educational approximation grounded in published pharmacology.
They are NOT clinical measurements (see the disclaimer in copy.py, surfaced in
the UI per §6). Values are transcribed verbatim from HANDOFF §3.1.

Ordering follows §3 v1 table. Umut has final say on ordering (§9); changing it
is a matter of reordering PRESET_ORDER below.
"""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "binge_eating": {
        "label": "Binge eating",
        "anticipation_spike": 1.6,      # dopamine ×baseline during anticipation
        "consumption_spike": 1.3,       # dopamine at consumption (LOWER than anticipation —
                                        # this is the key insight: wanting > liking)
        "opioid_release": 0.8,          # β-endorphin at consumption (the "relief")
        "time_to_peak_min": 15,
        "crash_depth": 0.55,            # dopamine undershoot as fraction of baseline
        "crash_onset_min": 60,
        "cortisol_rebound": 0.75,       # post-episode shame/stress spike
        "recovery_hours": 18,
        "tolerance_per_episode": 0.03,  # baseline erosion per repetition
        "key_insight": "anticipation_exceeds_reward",
    },

    "porn": {
        "label": "Porn",
        "anticipation_spike": 1.9,      # novelty-driven, very high
        "consumption_spike": 2.2,       # orgasm dopamine peak
        "prolactin_surge": 0.9,         # post-orgasm prolactin → anhedonia, refractory
        "time_to_peak_min": 20,
        "crash_depth": 0.45,            # deeper undershoot than most loops
        "crash_onset_min": 20,          # very fast — prolactin is immediate
        "cortisol_rebound": 0.6,
        "recovery_hours": 36,           # longer than most
        "tolerance_per_episode": 0.05,  # fastest escalation of the six —
                                        # novelty requirement compounds (Coolidge effect)
        "key_insight": "novelty_escalation",
    },

    "nicotine": {
        "label": "Nicotine / vaping",
        "anticipation_spike": 1.3,
        "consumption_spike": 1.9,
        "time_to_peak_sec": 10,         # fastest of all — 7-10 seconds to brain
        "half_life_hours": 2.0,
        "crash_depth": 0.6,
        "crash_onset_min": 45,
        "withdrawal_trough_hours": 3,   # the "I need one" point
        "recovery_hours": 72,
        "tolerance_per_episode": 0.02,
        "receptor_upregulation": True,  # nAChR density increases → higher baseline need
        "key_insight": "withdrawal_not_pleasure",  # smoking relieves withdrawal it caused
    },

    "doomscrolling": {
        "label": "Doomscrolling",
        "anticipation_spike": 1.15,     # small spikes...
        "consumption_spike": 1.25,      # ...but MANY of them
        "spike_frequency_per_min": 4,   # variable-ratio schedule
        "satiety_signal": 0.0,          # THE KEY: no natural stopping point
        "time_to_peak_min": 2,
        "crash_depth": 0.7,
        "crash_onset_min": 90,
        "cortisol_rebound": 0.8,        # negative content → sustained stress
        "recovery_hours": 8,
        "tolerance_per_episode": 0.015,
        "key_insight": "no_satiety_signal",
    },

    "alcohol": {
        "label": "Alcohol",
        "anticipation_spike": 1.4,
        "consumption_spike": 1.7,
        "gaba_agonism": 0.85,           # the anxiolytic effect
        "time_to_peak_min": 45,
        "glutamate_rebound": 0.9,       # THE hangxiety mechanism
        "rebound_onset_hours": 8,
        "crash_depth": 0.5,
        "cortisol_rebound": 0.85,
        "recovery_hours": 48,
        "tolerance_per_episode": 0.025,
        "key_insight": "gaba_rebound_anxiety",
    },

    "caffeine": {
        "label": "Caffeine",
        "adenosine_blockade": 0.8,      # NOT primarily dopaminergic — important nuance
        "anticipation_spike": 1.1,
        "consumption_spike": 1.3,
        "time_to_peak_min": 45,
        "half_life_hours": 5.5,
        "crash_depth": 0.75,
        "crash_onset_hours": 5,
        "adenosine_debt": True,         # blocked adenosine accumulates → harder crash
        "recovery_hours": 12,
        "tolerance_per_episode": 0.01,
        "key_insight": "borrowed_not_created",  # you don't gain energy, you defer fatigue
    },
}

# Display / selection order for the six v1 presets (§3 table).
PRESET_ORDER: list[str] = [
    "binge_eating",
    "porn",
    "nicotine",
    "doomscrolling",
    "alcohol",
    "caffeine",
]

# "Something else" → generic dopamine template (§4, SCREEN 1).
GENERIC_PRESET: dict = {
    "label": "Something else",
    "anticipation_spike": 1.4,
    "consumption_spike": 1.6,
    "time_to_peak_min": 20,
    "crash_depth": 0.55,
    "crash_onset_min": 60,
    "cortisol_rebound": 0.6,
    "recovery_hours": 24,
    "tolerance_per_episode": 0.02,
    "key_insight": "generic_dopamine_loop",
}

GENERIC_KEY = "generic"


def get_preset(key: str) -> dict:
    """Return a preset by key, including the generic template."""
    if key == GENERIC_KEY:
        return GENERIC_PRESET
    return PRESETS[key]


def all_selectable() -> list[tuple[str, str]]:
    """(key, label) pairs in display order, with the generic option last (§4)."""
    pairs = [(k, PRESETS[k]["label"]) for k in PRESET_ORDER]
    pairs.append((GENERIC_KEY, GENERIC_PRESET["label"]))
    return pairs
