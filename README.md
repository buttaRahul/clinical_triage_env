# Clinical Triage Navigator

An OpenEnv-compliant emergency department triage simulation for AI agent training and evaluation. Agents act as triage nurses — assessing synthetic patients, assigning Emergency Severity Index (ESI) levels, requesting resources, routing patients to the correct care area, and escalating sentinel conditions.

---

## Why This Environment

Emergency triage is one of the highest-stakes decision-making tasks in medicine. A triage nurse sees dozens of patients per shift, each with incomplete information, and must rapidly decide who is seen immediately and who can wait. Getting it wrong in the direction of under-triage kills people.

Existing agent benchmarks test language, math, and code. This environment tests something different: **clinical risk stratification under uncertainty** — a task humans do every day, where the cost of errors is asymmetric (under-triage is dangerous, over-triage is merely wasteful), and where surface-level pattern matching fails on atypical presentations.

This fills a real gap: there is no existing OpenEnv environment that evaluates an agent's ability to reason about medical acuity, sentinel condition recognition, or resource prioritisation under surge conditions.

---

## Environment Description

The environment presents a queue of synthetic patient cards, each containing:
- Chief complaint and history of present illness
- Past medical history (randomly omitted 40% per item per episode — simulates incomplete intake)
- Vital signs with seed-locked noise (±8 bpm HR, ±10 mmHg BP, ±1.5% SpO2, ±0.3°C temp)
- Medications, allergies, additional clinical context

The agent triages each patient in sequence. Some patients have hidden **sentinel conditions** (STEMI, stroke, sepsis, PE, etc.) that require explicit escalation — independent of ESI accuracy. The reward function is asymmetric: under-triage is penalised 5× harder than over-triage, reflecting real clinical stakes.

---

## Action Space

```json
{
  "esi_level": 1,
  "resources_requested": ["IV_ACCESS", "LABS", "ECG"],
  "disposition": "resus_bay",
  "escalation": "ACTIVATE_TEAM",
  "clinical_reasoning": "STEMI pattern with haemodynamic instability..."
}
```

| Field | Type | Values |
|---|---|---|
| `esi_level` | int | 1 (resuscitation) → 5 (non-urgent) |
| `resources_requested` | list[str] | `IV_ACCESS`, `LABS`, `ECG`, `IMAGING_XRAY`, `IMAGING_CT`, `IMAGING_MRI`, `MEDICATION_IV`, `MEDICATION_ORAL`, `SPECIALIST_CONSULT`, `BEDSIDE_MONITORING`, `PROCEDURE`, `NONE` |
| `disposition` | str | `resus_bay`, `trauma_bay`, `monitored_bed`, `fast_track`, `waiting_room` |
| `escalation` | str | `NONE`, `ACTIVATE_TEAM`, `ALERT_SENIOR` |
| `clinical_reasoning` | str | Free-text rationale (scored on hard task) |

**ESI levels:**
- **ESI 1** — Resuscitation: immediately life-threatening, needs intervention now (GCS ≤10, SBP <90, SpO2 <90%)
- **ESI 2** — Emergent: high-risk, time-critical, not yet in physiologic collapse
- **ESI 3** — Urgent: stable, needs ≥2 resources, can wait 30–60 min
- **ESI 4** — Less-urgent: needs exactly 1 resource
- **ESI 5** — Non-urgent: history and exam alone sufficient

**Escalation:**
- `ACTIVATE_TEAM` — immediate team activation for confirmed sentinel: STEMI, stroke, sepsis, cardiac arrest, ectopic pregnancy, overdose
- `ALERT_SENIOR` — high clinical suspicion not yet confirmed: PE, SAH, aortic dissection
- `NONE` — routine triage, no sentinel suspected

---

## Observation Space

Returned after every `reset()` and `step()`:

```json
{
  "current_patient": { "patient_id": "E001", "age": 68, "sex": "M", ... },
  "last_action_feedback": "UNDER-TRIAGE: You assigned ESI 2, correct was ESI 1...",
  "last_esi_correct": false,
  "last_esi_assigned": 2,
  "last_esi_ground_truth": 1,
  "patients_remaining": 9,
  "patients_triaged": 1,
  "step_reward": -0.21,
  "cumulative_reward": -0.21,
  "done": false,
  "task_name": "easy",
  "task_description": "..."
}
```

The `last_action_feedback` field provides rich clinical feedback after each step — ESI accuracy, missed/unnecessary resources, disposition, and sentinel escalation result. This gives the agent a learning signal within the episode.

---

## Tasks

### easy
10 patients with clear, unambiguous presentations covering all ESI levels (1–5). Vital signs are seed-perturbed on each reset. 3 sentinel conditions embedded: STEMI, SAH, STROKE.

**Grader:** `0.7 × ESI accuracy + 0.3 × sentinel accuracy`
**Pass threshold:** 0.60 | **Frontier model baseline:** 0.86

### medium
10 patients with higher acuity mix and competing priorities. Includes atypical presentations. PMH randomly omitted. 3 sentinel conditions: NSTEMI, ectopic pregnancy, PE.

**Grader:** `0.6 × ESI accuracy + 0.4 × sentinel accuracy`
**Pass threshold:** 0.55 | **Frontier model baseline:** 0.75

### hard
10-patient mass casualty scenario. Incomplete histories, ethical complexity (DNR patients, unknown overdose), degraded information quality. Clinical reasoning scored. 3 sentinel conditions: cardiac arrest, overdose, sepsis shock. Episode-end bonus (+0.5) for zero critical misses.

**Grader:** `0.55 × ESI accuracy + 0.45 × sentinel accuracy + episode_bonus`
**Pass threshold:** 0.45 | **Frontier model baseline:** 0.89

---

## Reward Function

The reward is asymmetric by design — under-triage (missing severity) is clinically dangerous and penalised 5× harder than over-triage.

| Component | Value |
|---|---|
| ESI exact match | +1.00 |
| Over-triage by 1 (cautious — safe) | +0.50 |
| Over-triage by 2 | −0.20 |
| Over-triage by 3+ | −0.50 |
| Under-triage by 1 | −1.00 |
| Under-triage by 2 | −2.50 |
| Under-triage by 3+ (critical miss) | −5.00 |
| Sentinel correctly escalated | +0.40 |
| Sentinel missed (no escalation) | −3.00 |
| False escalation (ACTIVATE_TEAM) | −0.20 |
| False escalation (ALERT_SENIOR) | −0.10 |
| Resource correct | +0.10 per resource |
| Resource false positive | −0.05 per resource |
| Disposition correct | +0.20 |
| Idle penalty | −0.01 per step |
| Episode bonus: zero critical misses | +0.50 |

Rewards are provided at every step (not just episode end), giving a dense training signal over the full trajectory.

---

## Setup

### Run with Docker

```bash
docker build -t clinical-triage-navigator .
docker run -p 8000:8000 -e TRIAGE_TASK=easy clinical-triage-navigator
```

Available tasks: `easy`, `medium`, `hard`

### API endpoints

```bash
POST /reset          # Start new episode
POST /step           # Submit triage decision
GET  /state          # Episode metadata
GET  /score          # Normalised score [0.0, 1.0]
GET  /health         # Health check
GET  /web            # Browser UI (set ENABLE_WEB_INTERFACE=true)
```

### Run baseline inference

```bash
# Set credentials
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://router.huggingface.co/v1   # default
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct            # default

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
| easy | 0.86 | Model handles clear presentations well |
| medium | 0.75 | Atypical presentations cause some confusion |
| hard | 0.89 | Critical cases (sepsis, cardiac arrest) are correctly escalated |

Scores are reproducible — same seed always produces the same episode (vital sign noise is seed-locked).

---

## Project Structure

```
clinical-triage-navigator/
├── Dockerfile                  # Root-level, copies full project
├── inference.py                # Baseline inference script
├── openenv.yaml                # OpenEnv manifest
├── models.py                   # Pydantic models (Action, Observation, State)
├── server/
│   ├── app.py                  # FastAPI: /reset /step /state /score
│   ├── clinical_triage_environment.py   # Core simulation + reward engine
│   └── patient_cases.py        # 30 hand-authored patient cases (10 per task)
└── tests/
    └── test_triage_env.py
```

---

## Future Work

### Dataset Integration

The current patient cases are hand-authored (30 cases across 3 tasks). The full architecture plan calls for grounding patient distributions in real clinical data:

**MIMIC-IV-ED** ([PhysioNet](https://physionet.org/content/mimic-iv-ed/))
400K+ real ED admissions from BIDMC 2011–2019. Tables: `triage` (vitals, chief complaint, nurse ESI), `edstays` (disposition), `diagnosis` (ICD codes), linkable to `icu.icustays` for outcome confirmation. The plan involves outcome-based ESI correction — if a nurse assigned ESI-3 but the patient went directly to ICU, relabel as ESI-2. This gives cleaner ground truth than raw nurse labels. Requires PhysioNet credentialing + CITI training + signed DUA (~1–2 days).

**TRIAGEAGENT** (EMNLP 2024)
~200 physician-curated clinical vignettes with gold-standard ESI labels (ESI-1: 12, ESI-2: 92, ESI-3: 65, ESI-4: 22, ESI-5: 25). Designed specifically for the contrastive variant generation pipeline. Labels are physician-assigned (not nurse-assigned), eliminating inter-rater noise.

### Contrastive RL Mechanics

The architecture plan includes a **contrastive consistency bonus** for tasks 1 and 2: each patient is presented as 3 surface paraphrases (same vitals, same ESI ground truth, different chief complaint phrasing and note style). The variants are interleaved randomly in the episode queue — the agent doesn't know it's seeing the same patient three times.

```python
# Consistency bonus
if all ESI assignments agree across 3 variants:  +0.1
if max - min == 1 (minor inconsistency):           0.0
if span >= 2 (same patient, wildly different):    -0.15
```

This penalises surface-pattern hacking — agents that change their ESI assignment based on complaint wording rather than clinical signs. Variants would be generated using GPT-4o with a strict prompt (same ESI, same vitals, different phrasing). Not applied to the hard/MCI task where decisions are order-dependent.

**Co-rewarding (partial):** A cross-variant reward comparison mechanism for tasks 1 and 2, rewarding internally consistent reasoning across paraphrased presentations. Full Co-rewarding is not applied to dynamic tasks where episode order changes the decision context.

### Deterioration Mechanics

For medium/hard tasks: 1–3 patients assigned a hidden `deterioration_window` (in steps). If the agent hasn't escalated the patient before the window closes, a deterioration event fires and a severity-weighted penalty is applied (`-2.0 × SEVERITY[condition]`). If the agent pre-escalated, the deterioration is classified as non-preventable and the agent receives a +0.5 credit. This requires a `DeptState` / department simulator model (concurrent patient tracking with a time axis) — a meaningful architectural extension beyond the current sequential queue model.

### Task-Specific Grader Formulas (Full)

Once deterioration and department state are implemented, the grader formulas become:
- **medium_atypical:** `0.4 × ESI + 0.3 × sentinel + 0.2 × deterioration + 0.1 × resources`
- **hard_mci:** `0.35 × allocation + 0.25 × no_preventable_deaths + 0.20 × sentinel + 0.20 × throughput`

Where `allocation` measures resource contention decisions (3 ESI-2 patients, 1 resus bay) and `throughput` measures % of patients seen within their ESI target wait time during surge.
