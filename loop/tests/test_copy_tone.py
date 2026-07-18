"""§6 tone guard: no shame/prescription/gamification language in user-facing text.

Scans every string the user can see — chrome (copy.py), preset subtitles, and
the generated narration templates + per-preset insights (narrate.py).
"""

import re

import pytest

from loop.copy import COPY, PRESET_SUBTITLES, TAGLINE
from loop.narrate import PHASE_TEMPLATES, INSIGHT_TEXT

# Banned as whole words (case-insensitive). "quit" would also catch nothing
# legitimate here; word boundaries avoid false hits like "cleanly" (none exist).
BANNED_WORDS = [
    "addiction", "addict", "sober", "relapse",
    "failure", "quit", "willpower is", "loser", "dirty",
]
BANNED_PHRASES = [
    "you should", "you need to", "try to", "you must", "you have to",
    "streak", "badge", "score", "level up", "points",
]


def _user_facing_strings():
    out = []
    for lang in COPY:
        for key, val in COPY[lang].items():
            if isinstance(val, str):
                out.append((f"copy[{lang}][{key}]", val))
            elif isinstance(val, list):
                for k, label in val:
                    out.append((f"copy[{lang}][{key}]", label))
    for k, v in PRESET_SUBTITLES.items():
        out.append((f"subtitle[{k}]", v))
    for k, v in PHASE_TEMPLATES.items():
        out.append((f"phase[{k}]", v))
    for k, v in INSIGHT_TEXT.items():
        out.append((f"insight[{k}]", v))
    out.append(("tagline", TAGLINE))
    return out


ALL_STRINGS = _user_facing_strings()


@pytest.mark.parametrize("word", BANNED_WORDS)
def test_no_banned_words(word):
    pat = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    hits = [name for name, s in ALL_STRINGS if pat.search(s)]
    assert not hits, f"banned word '{word}' in: {hits}"


@pytest.mark.parametrize("phrase", BANNED_PHRASES)
def test_no_banned_phrases(phrase):
    hits = [name for name, s in ALL_STRINGS if phrase.lower() in s.lower()]
    assert not hits, f"banned phrase '{phrase}' in: {hits}"


def test_disclaimer_present_and_verbatim():
    d = COPY["en"]["disclaimer"]
    assert "educational model, not a medical tool" in d
    assert "doctor or therapist" in d


def test_resources_present():
    assert COPY["en"]["resources"].strip()
