---
title: Clinical Triage Navigator
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# Clinical Triage Navigator

An OpenEnv environment that simulates emergency department triage. Agents work as triage nurses: assess each patient, assign an ESI level, request resources, route to the right care area, and escalate sentinel conditions like STEMI, stroke, or sepsis.

---

## Why This Environment

Emergency triage is one of the highest-stakes decisions in medicine. A nurse sees dozens of patients per shift with incomplete information and must decide in seconds who goes to resus and who waits. The cost of getting it wrong is asymmetric: under-triage can kill, over-triage wastes resources. This asymmetry is built directly into the reward function.

Most agent benchmarks test language and reasoning in low-stakes domains. This environment tests something different: **risk-calibrated decision-making under uncertainty** with real clinical stakes.

### Why Reinforcement Learning Specifically

This environment is designed for RL because the structure of the problem maps directly onto what RL does well:

**Asymmetric cost structure.** Getting ESI 1 wrong by under-triaging is 5x worse than over-triaging. A supervised model trained on accuracy would not learn this asymmetry. RL with a shaped reward function can.

**Delayed signal.** The episode-end bonus for zero critical misses only appears after all 10 patients are triaged. An agent that misses a STEMI early gets punished at the end. This requires credit assignment over a trajectory, not just per-step prediction.

**Exploration-exploitation tradeoff.** Over-triaging is safe but wastes resources. Under-triaging is dangerous. An RL agent must learn the right level of caution from reward signal rather than copying labeled examples.

**Distributional shift.** Vital signs are perturbed with seed-locked noise on every reset. PMH is randomly omitted. The agent must learn to act on the underlying clinical pattern, not memorize patient-specific presentations.

---

## Environment Description

Each episode presents a queue of 10 synthetic patients. For each patient the agent sees:
- Chief complaint and history of present illness
- Past medical history (randomly omitted 40% per item per episode)
- Vital signs with seed-locked noise (+/-8 bpm HR, +/-10 mmHg BP, +/-1.5% SpO2, +/-0.3C temp)
- Medications, allergies, and any additional clinical context

Some patients have hidden sentinel conditions (STEMI, stroke, sepsis, PE, etc.) that require explicit escalation independent of ESI level. A patient can have the correct ESI assigned but still miss the sentinel — and vice versa. These are scored separately.

---

## Action Space

```json
{
  "esi_level": 1,
  "resources_requested": ["IV_ACCESS", "LABS", "ECG"],
  "disposition": "resus_bay",
  "escalation": "ACTIVATE_TEAM",
  "clinical_reasoning": "SBP 82, SpO2 88%, GCS 13 — haemodynamic compromise with STEMI on paramedic ECG."
}
```

| Field | Type | Values |
|---|---|---|
| `esi_level` | int | 1 (resuscitation) to 5 (non-urgent) |
| `resources_requested` | list[str] | `IV_ACCESS`, `LABS`, `ECG`, `IMAGING_XRAY`, `IMAGING_CT`, `IMAGING_MRI`, `MEDICATION_IV`, `MEDICATION_ORAL`, `SPECIALIST_CONSULT`, `BEDSIDE_MONITORING`, `PROCEDURE`, `NONE` |
| `disposition` | str | `resus_bay`, `trauma_bay`, `monitored_bed`, `fast_track`, `waiting_room` |
| `escalation` | str | `NONE`, `ACTIVATE_TEAM`, `ALERT_SENIOR` |
| `clinical_reasoning` | str | Free-text rationale (scored on hard task) |

**ESI levels:**
- **ESI 1** - Resuscitation: immediately life-threatening (GCS <=10, SBP <90, SpO2 <90%)
- **ESI 2** - Emergent: high-risk, time-critical, not in physiologic collapse yet
- **ESI 3** - Urgent: stable, needs 2 or more resources, can wait 30-60 min
- **ESI 4** - Less-urgent: needs exactly 1 resource
- **ESI 5** - Non-urgent: history and exam alone sufficient

**Escalation:**
- `ACTIVATE_TEAM` - immediate team activation: STEMI, stroke, sepsis, cardiac arrest, ectopic, overdose
- `ALERT_SENIOR` - high suspicion not confirmed: PE, SAH, aortic dissection
- `NONE` - standard triage

---

## Observation Space

Returned after every `reset()` and `step()`:

```json
{
  "current_patient": { "patient_id": "H003", "age": 52, "..." : "..." },
  "last_action_feedback": "UNDER-TRIAGE: ESI 3 assigned, correct ESI 1. SENTINEL MISSED: CARDIAC_ARREST.",
  "last_esi_correct": false,
  "last_esi_assigned": 3,
  "last_esi_ground_truth": 1,
  "patients_remaining": 7,
  "patients_triaged": 3,
  "step_reward": -8.01,
  "cumulative_reward": -4.23,
  "done": false,
  "task_name": "hard",
  "task_description": "..."
}
```

`last_action_feedback` gives rich per-step feedback covering ESI accuracy, missed/unnecessary resources, disposition, and sentinel result. This gives a dense learning signal within the episode.

---

## Tasks

### easy
10 patients with clear, unambiguous presentations covering all ESI levels 1-5. Vital signs perturbed on each reset. 3 sentinel conditions: STEMI, SAH, STROKE.

**Grader:** `0.7 x ESI accuracy + 0.3 x sentinel accuracy`
**Pass threshold:** 0.60 | **Baseline (Qwen2.5-72B):** 0.86

### medium
10 patients with higher acuity mix. Includes atypical presentations and competing priorities. PMH randomly omitted. 3 sentinel conditions: NSTEMI, ectopic pregnancy, PE.

**Grader:** `0.6 x ESI accuracy + 0.4 x sentinel accuracy`
**Pass threshold:** 0.55 | **Baseline (Qwen2.5-72B):** 0.75

### hard
10-patient mass casualty scenario designed to challenge frontier models. What makes it hard:
- Patients with deceptive presentations: elderly MI presenting as fatigue and jaw pain (no chest pain), overdose with normal initial vitals, sepsis in an immunosuppressed patient with no fever
- Incomplete histories and unknown medication lists
- Ethical complexity: DNR patients in cardiac arrest, unconscious patients with no ID
- Clinical reasoning is scored by keyword relevance (not length)
- 3 sentinel conditions: cardiac arrest, overdose, sepsis shock
- Episode-end bonus (+0.5) for zero critical misses

**Grader:** `0.55 x ESI accuracy + 0.45 x sentinel accuracy + episode_bonus`
**Pass threshold:** 0.45 | **Baseline (Qwen2.5-72B):** 0.89

---

## Reward Function

The reward is asymmetric — under-triage is clinically dangerous and penalised 5x harder than over-triage.

| Component | Value | Notes |
|---|---|---|
| ESI exact match | +1.00 | |
| Over-triage by 1 (cautious) | +0.50 | Safe heuristic, rewarded |
| Over-triage by 2 | -0.20 | |
| Over-triage by 3+ | -0.50 | |
| Under-triage by 1 | -1.00 | Dangerous |
| Under-triage by 2 | -2.50 | |
| Under-triage by 3+ | -5.00 | Critical miss |
| Sentinel correctly escalated | +0.40 | |
| Sentinel missed | -3.00 | |
| False escalation (ACTIVATE_TEAM) | -0.20 | |
| False escalation (ALERT_SENIOR) | -0.10 | |
| Vital concordance bonus | +0.20 | ESI 1 correctly caught from objective markers (SBP<90, SpO2<90%, GCS<=10) |
| Resource correct | +0.10 per resource | |
| Resource false positive | -0.05 per resource | |
| Disposition correct | +0.20 | |
| Clinical reasoning (hard only) | 0.0 to +0.30 | Keyword-based: ESI-appropriate terms, sentinel mentions, objective findings |
| Idle penalty | -0.01 per step | |
| Episode bonus: zero critical misses | +0.50 | |

**Why this reward structure is interesting for RL:**
The vital concordance bonus creates a second signal path: the agent can score higher by grounding decisions in objective physiologic data rather than chief complaint pattern-matching. Combined with the sentinel escalation component (fully independent of ESI accuracy), the reward function tests three distinct clinical skills simultaneously.

---

## Reproducibility Guarantee

All episode variation is seed-locked. Vital sign noise is generated with `random.Random(seed)` before any LLM calls. PMH omissions use the same RNG. Same seed always produces identical patient presentations across machines and runs.

The baseline scores in this README were generated with seed=42, temperature=0.1, using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace inference router. Running `python inference.py` with the same credentials will reproduce them.

---

## Setup

### Run with Docker

```bash
docker build -t clinical-triage-navigator .
docker run -p 7860:7860 -e TRIAGE_TASK=easy clinical-triage-navigator
```

Available tasks: `easy`, `medium`, `hard`

### API endpoints

```
POST /reset     Start new episode
POST /step      Submit triage decision
GET  /state     Episode metadata
GET  /score     Normalised score [0.001, 0.999]
GET  /health    Health check
GET  /docs      Swagger UI
GET  /web       Browser UI (set ENABLE_WEB_INTERFACE=true)
```

### Run baseline inference

```bash
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

### Install for development

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

---

## Baseline Scores

Evaluated with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace inference router, temperature 0.1, seed 42.

| Task | Score | Notes |
|---|---|---|
| easy | 0.86 | Clear presentations, model handles well |
| medium | 0.75 | Atypical presentations and PMH omissions cause some misses |
| hard | 0.89 | Sentinel conditions (cardiac arrest, sepsis) correctly escalated |

---

## Project Structure

```
clinical-triage-navigator/
├── Dockerfile                             # Root-level, copies full project
├── inference.py                           # Baseline inference script
├── openenv.yaml                           # OpenEnv manifest
├── models.py                              # Pydantic models (Action, Observation, State)
├── server/
│   ├── app.py                             # FastAPI: /reset /step /state /score
│   ├── clinical_triage_environment.py     # Core simulation and reward engine
│   └── patient_cases.py                   # 30 hand-authored patient cases (10 per task)
└── tests/
    └── test_triage_env.py
```

---

## Future Work

**Dataset Integration**

The current patient cases are hand-authored. Two datasets would ground the environment in real clinical data:

*MIMIC-IV-ED* (PhysioNet): 400K+ real ED admissions from BIDMC 2011-2019. The plan uses outcome-based ESI correction — if a nurse assigned ESI-3 but the patient went to ICU, relabel as ESI-2 for cleaner ground truth. Requires PhysioNet credentialing and a signed DUA.

*TRIAGEAGENT* (EMNLP 2024): ~200 physician-curated vignettes with gold-standard ESI labels, designed for contrastive variant generation.

**Contrastive RL**

Each patient presented as 3 surface paraphrases (same vitals, same ESI, different phrasing). Interleaved randomly so the agent doesn't know it's seeing the same patient three times. An inconsistency penalty fires when the agent assigns different ESI levels to the same underlying case.

**Deterioration Mechanics**

For medium/hard tasks: hidden deterioration windows per patient. If the agent doesn't escalate before the window closes, a severity-weighted penalty fires. If the agent pre-escalated, the deterioration is classified as non-preventable and the agent gets a credit instead.
