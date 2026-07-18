"""Every static user-facing string, in one place (§5 layout, §6 tone).

Tone rules (§6), enforced by tests/test_copy_tone.py:
  Never: addiction/addict, clean, sober, relapse, failure, quit; "you should",
         "you need to", "try to"; success/failure framing; streaks/badges/scores.
  Always: loop / pattern / cycle / episode; mechanism language; second-person
          descriptive not prescriptive; agency-restoring ("this is predictable").

Generated phase narration lives in narrate.py (coupled to detection logic — see
DECISIONS.md D4). Everything else — chrome, questions, disclaimer, resources,
about — is here. The structure is i18n-ready: strings hang off the COPY dict
under a language key; swap "en" for another locale without touching app.py.

Branding: APP_NAME is the single source of the product name (§2). Changing the
name is a one-line edit here.
"""

from __future__ import annotations

# --- Branding (one-line change per §2) ------------------------------------
APP_NAME = "Loop"
TAGLINE = "Stop guessing why you keep doing this. See the mechanism."

COPY: dict[str, dict] = {
    "en": {
        # Screen 1 — pick your loop
        "s1_title": "Which loop do you want to understand?",
        "s1_subtitle": "Pick the one you want to see the shape of.",

        # Screen 2 — make it yours
        "s2_title": "Make it yours",
        "s2_subtitle": "Three questions. They shape the curve, nothing is stored.",
        "q_frequency": "How often does it happen?",
        "q_duration": "How long has this pattern been running?",
        "q_intensity": "How strong is the pull on a bad day?",

        # Screen 3 — your loop, narrated
        "s3_title": "Your loop, narrated",
        "s3_subtitle": "The same shape every time. Here is what each part is.",
        "axis_time": "Hours since the cue",
        "axis_level": "Level (relative to your baseline)",
        "legend_dopamine": "Dopamine",
        "legend_cortisol": "Cortisol (stress)",
        "legend_baseline": "Your baseline",

        # Screen 4 — what if
        "s4_title": "What if?",
        "s4_subtitle": "Same mechanism, different inputs. Watch the curve move.",
        "whatif_delay": "What if you delayed 2 hours?",
        "whatif_frequency": "What if you cut frequency in half?",
        "whatif_recovery": "What does 30 days of no repetition do to your baseline?",
        "whatif_delay_note": (
            "Delaying doesn't erase the loop — it moves the crash. Seeing where "
            "it lands is the point."
        ),
        "whatif_frequency_note": (
            "Fewer repetitions, less the baseline erodes. This is the curve, not "
            "a target."
        ),
        "whatif_recovery_note": (
            "With no repetition, the baseline drifts back up toward where it "
            "started. This is the most hopeful curve here, and it is just the "
            "mechanism running in reverse."
        ),

        # Navigation
        "cta_next": "Show me the curve",
        "cta_whatif": "What if?",
        "cta_restart": "Map a different loop",
        "cta_back": "Back",

        # Disclaimer (verbatim, §6 — must be visible, not buried)
        "disclaimer": (
            "This is an educational model, not a medical tool. The curves are "
            "approximations based on published pharmacology — they show general "
            "mechanisms, not measurements of your brain. If a pattern is "
            "affecting your life, a doctor or therapist can help in ways a "
            "simulation cannot."
        ),

        # Resources (one line, country-agnostic, always present, §6)
        "resources": (
            "If a pattern is affecting your life: SAMHSA (US) 1-800-662-4357 · "
            "Samaritans (UK & IE) 116 123 · or search your local helpline."
        ),

        # About / scientific grounding (§5)
        "about_title": "About this model",
        "about_body": (
            "Every episode in this tool is built from the same idea: a fast "
            "response up, and a slower opposite response that outlasts it and "
            "pulls you below where you started. With repetition, the whole "
            "baseline drifts down.\n\n"
            "The frame is Solomon & Corbit (1974), the opponent-process theory "
            "of motivation, and Koob & Le Moal (2001), the allostatic model of "
            "the reward set-point shifting down over time. The split between "
            "'wanting' (the dopamine anticipation) and 'liking' (the relief at "
            "consumption) follows Berridge & Robinson.\n\n"
            "The numbers per loop are educational approximations grounded in "
            "published pharmacology. They are not measurements of any individual "
            "brain."
        ),

        # Frequency options: (key, label)
        "frequency_options": [
            ("several_per_day", "Several times a day"),
            ("daily", "About once a day"),
            ("few_per_week", "A few times a week"),
            ("bursts", "In bursts"),
        ],
        # Duration options: (key, label)
        "duration_options": [
            ("weeks", "Weeks"),
            ("months", "Months"),
            ("years", "Years"),
        ],
        "intensity_min": "Barely there",
        "intensity_max": "Overwhelming",
    }
}

# Per-preset one-line card subtitle — mechanism tease, no judgment (§3.2, §6).
PRESET_SUBTITLES: dict[str, str] = {
    "binge_eating": "The wanting is bigger than the reward. See the gap.",
    "porn": "Novelty escalation and the drop right after the peak, as a curve.",
    "nicotine": "Mostly relieving the withdrawal the last one created.",
    "doomscrolling": "Small hits, many of them, and no signal to stop.",
    "alcohol": "Calm now; the glutamate rebound arrives hours later.",
    "caffeine": "Energy borrowed, not created — the crash is the bill.",
    "generic": "Map any dopamine loop and see its shape.",
}


def t(key: str, lang: str = "en") -> str:
    """Look up a chrome string."""
    return COPY[lang][key]


def frequency_options(lang: str = "en") -> list[tuple[str, str]]:
    return COPY[lang]["frequency_options"]


def duration_options(lang: str = "en") -> list[tuple[str, str]]:
    return COPY[lang]["duration_options"]
