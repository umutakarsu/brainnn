# HANDOFF SPEC — "Loop" consumer product pivot

**Status:** Spec written 2026-07-07. Implementation NOT started.
**Written by:** Claude session A (low on tokens) → handing to session B.
**Read this whole file before writing code. It is designed so you do not need to ask questions.**

---

## 0. What you are picking up

This repo (`brainnn`) currently contains:

1. **A research project** — neuromodulator-conditioned BCI decoding. Cross-subject EEG transformer, FiLM conditioning on ACh/NE/DA, 40% zero-shot LOSO result. Peter Dayan reviewed the architecture. **Do not touch this.** It is grant-critical and already pushed. Files: `src/brainnn/bci/`, `bci_dashboard.py`, `README.md`, `docs/`.

2. **A simulation engine** — `src/brainnn/core/state.py` (BrainState with ACh/NE/DA), `src/brainnn/attention/` (ADHD attention model). This is the engine you will reuse.

3. **An old dashboard** — `neurodivergent_app.py`. Deployed to HuggingFace Spaces as `umuutakarsu/neurodivergent-brain-sim`. This is what you are **replacing**.

**Your job:** build the consumer product described below.

---

## 1. The pivot

### What's wrong with the current app

`neurodivergent_app.py` is framed as "simulate a neurodivergent brain — ADHD attention + synesthesia + cross-modal generation." Problems:

- **No clear user.** Who opens this and why?
- **Synesthesia is unrelated.** It was a research artifact. It confuses the story. **Remove it entirely.**
- **Too technical.** "Modality Latents", "Attention Weights", "Cross-Modal Generation" — nobody outside a lab cares.
- **No commercial path.**

### What it becomes

**A tool where anyone can map their own behavioural loop and see what is happening in their brain.**

The origin insight: Umut mapped a friend's binge-eating cycle using this engine and showed them the dopamine curve. The friend's awareness increased measurably — because seeing "this is a neurochemical loop with a predictable shape" is different from "I have no self-control." **Shame drops, agency rises.** That is the product.

**Target user:** anyone stuck in a dopamine-driven loop who wants to understand it, not be lectured at.

**Core promise:** *"Stop guessing why you keep doing this. See the mechanism."*

---

## 2. Product name

**Primary recommendation: `Loop`**

- Short, memorable, ownable in this context
- Users already say "I'm stuck in a loop"
- Neutral — not clinical, not moralising, not recovery-coded
- Works as a verb ("map your loop")

**Repo:** create a **new repo** `umutakarsu/loop`. Do not build this inside `brainnn` — the research repo must stay clean for grant reviewers. Copy the ~200 lines of engine you need (see §5).

Alternates if `Loop` feels wrong: `Cravescope`, `The Loop`, `Understand`.
**Umut has final say on name.** If unsure, build with `Loop` and leave branding in one config constant so it is a one-line change.

---

## 3. Presets — the preset library IS the product

Each preset is a neurochemical profile. The user picks their loop, tunes 2-3 sliders, and gets a personalised timeline with narration.

### Ship these six in v1

| # | Preset | Why it's in v1 |
|---|---|---|
| 1 | **Binge eating** | The origin story. Proven to work on a real person. |
| 2 | **Porn** | **Explicitly requested by Umut. Do not omit.** Huge underserved audience (NoFap, therapy-adjacent). Almost no tool explains the neurochemistry non-judgmentally. Handle clinically — see §3.2. |
| 3 | **Nicotine / vaping** | 1.3B users worldwide. Cleanest pharmacology to model. |
| 4 | **Doomscrolling** | Gen Z hook. Variable-ratio reinforcement is fascinating when visualised. |
| 5 | **Alcohol** | Culturally discussable, "hangxiety" is a strong aha-moment. |
| 6 | **Caffeine** | Low-stakes entry point. Lets a curious user try the tool without disclosing anything heavy. |

Later: cannabis, gambling, stimulants, shopping, gaming.

### 3.1 Neurochemical parameters per preset

These are the modelling parameters. They are grounded in real pharmacology but are **educational approximations, not clinical measurements** — this must be stated in the UI (see §6).

```python
PRESETS = {
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
```

### 3.2 Handling the porn preset

This is a legitimate behavioural-health topic — ICD-11 recognises Compulsive Sexual Behaviour Disorder, and there is a large audience (NoFap, therapy-adjacent apps) with almost no non-judgmental neuroscience tooling.

**Rules:**
- The tool shows **curves and mechanisms only**. No explicit content, no imagery, no descriptions of acts.
- **Clinical register, zero moralising.** Not "porn is destroying your brain." Instead: "This is what the dopamine curve looks like, and here is why the escalation pattern happens."
- The distinguishing mechanism is **novelty escalation** (Coolidge effect) and **prolactin-driven post-peak anhedonia** — these are genuinely interesting and rarely explained well.
- No shame language anywhere. The whole product thesis is that shame makes loops worse.
- Same UI treatment as every other preset — do not hide it, do not flag it, do not add warnings that other presets lack. Differential treatment implies judgment.

---

## 4. UX flow

Replace the current "tune parameters → run simulation" engineering flow with a narrative one.

```
┌─ SCREEN 1: Pick your loop ────────────────────────┐
│  "Which loop do you want to understand?"          │
│  [6 preset cards, plain labels, no icons that     │
│   read as judgmental]                             │
│  + "Something else" → generic dopamine template   │
└───────────────────────────────────────────────────┘
                    ↓
┌─ SCREEN 2: Make it yours (2-3 questions max) ─────┐
│  "How often does it happen?"                      │
│    → several times a day / daily / few times a    │
│      week / in bursts                             │
│  "How long has this pattern been running?"        │
│    → weeks / months / years                       │
│      (drives tolerance accumulation)              │
│  "How strong is the pull on a bad day?"           │
│    → slider 1-10                                  │
└───────────────────────────────────────────────────┘
                    ↓
┌─ SCREEN 3: Your loop, narrated ───────────────────┐
│  Timeline chart (dopamine, cortisol, baseline     │
│  drift) with FOUR annotated phases:               │
│                                                   │
│  ① TRIGGER    "Cue fires. Dopamine rises before   │
│                you've done anything — this is     │
│                wanting, not liking."              │
│  ② PEAK       "The spike. Shorter than you        │
│                remember it being."                │
│  ③ CRASH      "Dopamine drops BELOW where you     │
│                started. This is the part that     │
│                gets blamed on willpower."         │
│  ④ VULNERABLE "Hours N-M: your baseline is at its │
│                lowest. This is when the next      │
│                trigger has the most power. The    │
│                loop is not a character flaw —     │
│                it's this curve."                  │
└───────────────────────────────────────────────────┘
                    ↓
┌─ SCREEN 4: What if? (the retention hook) ─────────┐
│  Interactive comparison, side-by-side curves:     │
│    "What if you delayed 2 hours?"                 │
│    "What if you cut frequency in half?"           │
│    "What does 30 days of no repetition do to      │
│     your baseline?"  ← baseline recovery curve,   │
│                        the most motivating chart  │
│  Show the CURVES changing, not advice text.       │
└───────────────────────────────────────────────────┘
```

**The insight in ④ is the product.** Everything else supports it. If a user leaves understanding only one thing, it should be: *the crash below baseline is the mechanism, and it is predictable, and it is not a moral failure.*

---

## 5. Architecture

### Reuse from `brainnn`

Copy (don't import — new repo should be standalone):

- `src/brainnn/core/state.py` → the `BrainState` dataclass. It already has ACh/NE/DA separated correctly (post-Dayan refactor). **Extend it** with `cortisol`, `opioid`, and `baseline_drift` fields for this product.
- Nothing else is required. The ADHD attention transformer is overkill here — this product needs a **pharmacokinetic curve model**, not a neural network.

### What to build

```
loop/
├── app.py                  # Streamlit entry point
├── loop/
│   ├── presets.py          # The PRESETS dict from §3.1
│   ├── state.py            # Extended BrainState (copy + extend from brainnn)
│   ├── simulate.py         # Curve generation — see below
│   ├── narrate.py          # Phase detection + narration text generation
│   └── copy.py             # All user-facing strings in ONE file (i18n-ready)
├── requirements.txt        # streamlit, numpy, plotly ONLY. No torch.
└── README.md
```

**Critical: no PyTorch.** This is a curve model, not a learned model. Keep the dependency footprint tiny so it deploys anywhere (HF Spaces free tier, Streamlit Cloud) in seconds.

### Simulation model

Do NOT reuse the discrete-timestep `BrainState.update()` loop from brainnn — it is too coarse. Write a continuous pharmacokinetic model:

```python
# Per-episode dopamine response — sum of:
#   1. Anticipation: sigmoid rise before consumption
#   2. Consumption: sharp peak
#   3. Clearance: exponential decay with preset half-life
#   4. Undershoot: opponent-process rebound below baseline
#      (Solomon & Corbit opponent-process theory — cite this,
#       it is the correct theoretical frame for the whole product)
#   5. Baseline drift: tolerance accumulation across repeated episodes

def episode_curve(t_hours, preset, intensity):
    anticipation = preset["anticipation_spike"] * sigmoid(...)
    peak         = preset["consumption_spike"] * gaussian(...)
    clearance    = exp_decay(half_life=preset["half_life_hours"])
    undershoot   = -preset["crash_depth"] * opponent_process(...)
    return baseline + anticipation + peak * clearance + undershoot
```

Theoretical grounding: **Solomon & Corbit (1974) opponent-process theory of motivation**, and **Koob & Le Moal (2001) allostatic model of addiction** — the "baseline shifts down with repetition" mechanism. Cite both in the About section. This gives the product real scientific credibility rather than pop-neuroscience vibes.

---

## 6. Copy and tone — non-negotiable rules

The entire product thesis is that **shame worsens loops** (cortisol elevation → lower prefrontal control → higher relapse probability). The copy must never add shame.

**Never write:**
- "addiction", "addict", "clean", "sober", "relapse", "failure", "quit"
- "you should", "you need to", "try to"
- Any success/failure framing
- Streaks, badges, scores, gamification

**Always write:**
- "loop", "pattern", "cycle", "episode"
- Mechanism language: "this is what happens", "the curve shows"
- Second-person descriptive, not prescriptive: "your dopamine drops below baseline here"
- Agency-restoring: "this is predictable" > "you're out of control"

**Required disclaimer** (must be visible, not buried in an expander):

> This is an educational model, not a medical tool. The curves are approximations based on published pharmacology — they show general mechanisms, not measurements of your brain. If a pattern is affecting your life, a doctor or therapist can help in ways a simulation cannot.

Also add a resources link block (country-agnostic: SAMHSA-equivalent, Samaritans, local helplines). Do not make it heavy — one line, always present.

---

## 7. Commercial model

Build v1 as **free and public**. Instrument it, learn what people use, then monetise.

Path, in order of realism for a solo 18-year-old in Berlin:

1. **Free public tool** (week 1) — HF Spaces or Streamlit Cloud. Get it in front of people. This is the priority.
2. **Freemium** (month 1-2) — 3 presets free, full library + save/compare + personal report export at ~€4/mo.
3. **Therapist tool** (month 3-6) — €25/mo for clinicians, per-client loop maps as session material. Requires talking to actual therapists first — do not build this speculatively.
4. **Integration** (month 6+) — license the engine to existing recovery apps (Reframe, Sunnyside, Quitzilla). Highest ceiling, longest sales cycle.

Do not build payments in v1. Ship the free version, see if anyone cares, then decide.

---

## 8. Concrete task list for session B

Work in this order. Commit after each numbered item so progress is recoverable.

1. Create new repo `umutakarsu/loop`. Init, `.gitignore`, MIT license.
2. Copy + extend `BrainState` from `brainnn/src/brainnn/core/state.py`. Add `cortisol`, `opioid`, `baseline_drift`.
3. Write `loop/presets.py` with the six presets from §3.1 verbatim.
4. Write `loop/simulate.py` — continuous curve model per §5. Unit-test that each preset produces: a peak above baseline, an undershoot below baseline, and monotonic baseline erosion with repetition.
5. Write `loop/narrate.py` — detect the four phases from a generated curve, return narration strings.
6. Write `loop/copy.py` — every user-facing string, following §6 rules.
7. Write `app.py` — the four-screen Streamlit flow from §4.
8. Deploy to HuggingFace Spaces (Umut already has an account: `umuutakarsu`). Free tier is fine — no torch means fast cold starts.
9. Write `README.md` with a screenshot and the scientific grounding (Solomon & Corbit, Koob & Le Moal).
10. Report back: deployed URL + a screenshot of the binge-eating and porn presets.

### Do NOT

- Touch `src/brainnn/bci/`, `bci_dashboard.py`, `README.md`, or `docs/` in the brainnn repo. Grant-critical.
- Add PyTorch or any ML dependency to the new repo.
- Add synesthesia, cross-modal generation, or attention-weight visualisation. Those belong to the research project.
- Add gamification, streaks, or any success/failure framing.
- Treat the porn preset differently from the others in the UI.
- Build payments, auth, or a database in v1.

---

## 9. Coordination between sessions

Session A (this one) and session B coordinate through **git only**.

- Session B: work in the new `loop` repo. Commit often with clear messages.
- If you change anything in `brainnn`, note it in a commit message starting with `[handoff]`.
- If you hit a decision this spec does not cover: **make the reasonable call, note it in `DECISIONS.md` in the new repo**, and keep moving. Do not block waiting for Umut.
- Umut has final say on: product name, preset ordering, pricing, anything that reads as clinical advice.

---

## 10. Context on Umut (so tone is right)

18, Turkish, currently in Berlin, starting a BSc September 2026. Solo — no lab, no advisor yet, on the research team of Prof. Surjo Soekadar at Charité. Technical: writes Python fluently, understands the neuroscience, has correspondence with Peter Dayan. Building a portfolio while applying for grants and a German student visa.

He does not need hand-holding or encouragement padding. He needs work that is finished, honest about what is uncertain, and shipped. If something in this spec is wrong, say so directly and fix it.

This product matters to him personally — it started because it genuinely helped a friend. Build it like that friend is the user.
