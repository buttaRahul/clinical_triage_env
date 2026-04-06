# Master context prompt — Clinical Triage Navigator (OpenEnv Hackathon)

Paste this entire message as your FIRST prompt to Claude in VS Code before asking it to do anything.

---

You are helping me build and submit a winning entry to the **OpenEnv Hackathon** (Meta PyTorch × Hugging Face × Scaler School of Technology, Bengaluru, April 2026). The prize is $10K in HF credits and a PyTorch blog feature. Judges are engineers from Meta, Hugging Face, and UC Berkeley.

## What OpenEnv is

OpenEnv (github.com/meta-pytorch/OpenEnv) is a framework by Meta and Hugging Face for building standardised, containerised RL training environments for LLM agents. It uses a Gymnasium-style API:
- `reset()` → returns initial `Observation`
- `step(action)` → returns `Observation`, reward, done, info
- `state()` → returns episode metadata

Environments run as FastAPI servers inside Docker containers, deployed to Hugging Face Spaces. Agents interact via HTTP/WebSocket. Every environment must pass `openenv validate`.

## What we are building

**Project name:** Clinical Triage Navigator

**One-line pitch:** An ER triage simulation where an AI agent learns to assign Emergency Severity Index (ESI) levels to synthetic patients — a real, high-stakes medical task performed millions of times daily in hospitals worldwide.

**Why it wins the hackathon:**
- Zero similar environments exist in the OpenEnv Hub or anywhere in the Gymnasium/RL ecosystem
- ESI triage levels are a published, standardised protocol — fully deterministic ground truth, no LLM-judge needed
- Dense partial-progress rewards at every step (not sparse end-of-episode reward)
- Under-triage is clinically dangerous and heavily penalised — makes the reward function medically authentic and interesting to judges
- The author has healthcare domain expertise — patient cases are clinically authentic, not fake
- Fills a critical gap explicitly identified in ICLR 2026 papers on scientific/medical workflow agents

## Hackathon judging criteria (weights)

| Criterion | Weight | What judges look for |
|---|---|---|
| Real-world utility | 30% | Does it model a genuine task? Would someone use this to train agents? |
| Task & grader quality | 25% | Well-defined objectives, deterministic graders 0.0–1.0, clear difficulty progression |
| Environment design | 20% | Clean state management, sensible action/observation spaces, dense reward shaping |
| Code quality & spec compliance | 15% | openenv validate passes, docker build works, HF Space deploys, baseline runs |
| Creativity & novelty | 10% | Novel domain, clever reward design, never seen in OpenEnv before |

**Minimum requirements to be accepted:**
- Full OpenEnv spec: typed Pydantic models, step/reset/state, openenv.yaml
- Minimum 3 tasks: easy → medium → hard, each with programmatic grader scoring 0.0–1.0
- Meaningful reward function with partial progress signals
- Baseline inference script using OpenAI API (reads OPENAI_API_KEY from env)
- Deployed to HF Spaces + working Dockerfile
- README with environment description, action/observation spaces, setup instructions

## Current project structure

The codebase already exists. Here is the complete file tree:

```
clinical_triage_env/
├── __init__.py                          # Public exports
├── models.py                            # All Pydantic models (TriageAction, TriageObservation, TriageState)
├── client.py                            # ClinicalTriageEnv HTTP client (async + sync)
├── baseline_inference.py                # Runs GPT-4o-mini against all 3 tasks, saves JSON results
├── openenv.yaml                         # OpenEnv Hub manifest
├── pyproject.toml                       # Package metadata + deps
├── outputs/
│   ├── logs/
│   └── evals/                           # baseline results saved here
├── tests/
│   └── test_triage_env.py               # Full pytest suite (reset/step/state/reward/all-tasks)
└── server/
    ├── __init__.py
    ├── app.py                           # FastAPI app with /reset /step /state /score /health /web
    ├── clinical_triage_environment.py   # Core simulation logic + reward computation
    ├── patient_cases.py                 # 30 synthetic patient cases (10 per task)
    ├── requirements.txt
    └── Dockerfile
```

## The typed models (models.py)

```python
# ESI levels 1–5 (1=Resuscitation, 5=Non-urgent)
class TriageAction(BaseModel):
    esi_level: ESILevel                      # int 1-5
    resources_requested: List[ResourceType]  # IV_ACCESS, LABS, ECG, IMAGING_CT, etc.
    clinical_reasoning: str                  # free-text rationale
    disposition: Literal["resus_bay", "trauma_bay", "monitored_bed", "fast_track", "waiting_room"]

class TriageObservation(BaseModel):
    current_patient: Optional[PatientCard]   # vitals, HPI, PMH, allergies, meds
    last_action_feedback: str                # clinical feedback on previous decision
    last_esi_correct: Optional[bool]
    last_esi_ground_truth: Optional[int]
    patients_remaining: int
    step_reward: float
    cumulative_reward: float
    done: bool
    task_name: str
    task_description: str

class TriageState(BaseModel):
    episode_id: str
    task_name: str
    step_count: int
    patients_triaged: int
    patients_total: int
    cumulative_reward: float
    done: bool
    esi_accuracy: float
    over_triage_count: int
    under_triage_count: int
```

## The 3 tasks

**Task: easy**
- 10 patients, one at a time, clear unambiguous presentations
- Examples: STEMI with diaphoresis (ESI 1), ankle sprain needing x-ray (ESI 4), viral rash (ESI 5)
- Agent just needs to assign correct ESI + resources

**Task: medium**
- 10 patients, arriving as a 5-patient queue
- Mix of time-critical emergencies (ectopic pregnancy, angioedema) and routine cases
- Correct prioritisation within the queue earns bonus reward

**Task: hard**
- 10 patients simultaneously, mass-casualty style
- Incomplete histories ("Unknown" PMH/medications)
- Ethical complexity (DNR patients, overdose with unknown substance)
- Clinical reasoning text is scored in addition to ESI accuracy
- Under-triage penalties are at maximum severity

## Reward function design

Per-patient reward components (summed, then normalised to 0.0–1.0 at episode end):

```
ESI exact match:           +1.00
ESI off by 1:              +0.40  (clinically acceptable)
ESI off by 2:              -0.20
ESI off by ≥3:             -0.60

Under-triage penalty (assigned ESI number too HIGH — patient more critical than assessed):
  Under by 1 level:        -0.30  (on top of accuracy penalty)
  Under by 2 levels:       -0.80
  Under by ≥3 levels:      -1.50  ← patient harm, max penalty

Over-triage (assigned ESI number too LOW — patient less critical than assessed):
  Over by 1 level:         -0.10  (mild — over-caution acceptable)
  Over by 2 levels:        -0.25
  Over by ≥3 levels:       -0.50

Resource correctness:      +0.10 per correct resource requested
False positive resource:   -0.05 per unnecessary resource
Correct disposition:       +0.20
Idle/loop penalty:         -0.01 per step (efficiency)
Hard task reasoning bonus: 0.0–0.30 based on clinical_reasoning quality
```

Final normalised score: `0.70 × (esi_exact_accuracy) + 0.30 × (reward_proxy)`

## ESI reference (the ground truth system)

```
ESI 1 — Resuscitation:  Immediately life-threatening. Needs intervention NOW.
                         (Cardiac arrest, unresponsive, severe shock, GCS ≤8)
ESI 2 — Emergent:       High-risk OR confused/lethargic OR severe pain/distress.
                         (STEMI, stroke, sepsis, ectopic pregnancy, severe allergic reaction)
ESI 3 — Urgent:         Stable but needs ≥2 resources. Not high-risk.
                         (Abdominal pain + fever, DVT workup, moderate asthma, low-risk chest pain)
ESI 4 — Less-urgent:    Needs exactly 1 resource.
                         (Simple laceration repair, strep throat, ankle sprain x-ray)
ESI 5 — Non-urgent:     Needs 0 resources.
                         (Viral URI, minor rash, prescription refill)
```

Resource types available: `IV_ACCESS, LABS, ECG, IMAGING_XRAY, IMAGING_CT, IMAGING_MRI, MEDICATION_IV, MEDICATION_ORAL, SPECIALIST_CONSULT, BEDSIDE_MONITORING, PROCEDURE, NONE`

## Server architecture

- **FastAPI** server with `/reset` (POST), `/step` (POST), `/action` (POST), `/state` (GET), `/score` (GET), `/health` (GET), `/web` (GET — HTML UI)
- Task selected via `TRIAGE_TASK` environment variable (easy/medium/hard)
- Web UI enabled via `ENABLE_WEB_INTERFACE=true`
- Dual-import pattern required: `try: from ..models import X / except ImportError: from models import X`
  (relative imports work in-repo; bare imports work inside Docker where PYTHONPATH=/app/env)
- `create_app(task)` factory function — never instantiate env at module level

## OpenEnv spec compliance requirements

1. `openenv.yaml` must be present with: name, version, description, action_type, observation_type, state_type, tasks list, reward spec, deployment config
2. All models must be Pydantic v2 BaseModel (not dataclasses)
3. `server/app.py` must use `create_app()` factory pattern
4. Dockerfile must: EXPOSE 8000, set PYTHONPATH, have HEALTHCHECK, use uvicorn
5. `openenv validate` must pass (checks /health, /reset, /step endpoints)

## HF Spaces deployment

The README.md must have this YAML frontmatter for HF Spaces to recognise it:

```yaml
---
title: Clinical Triage Navigator
emoji: 🏥
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
  - healthcare
  - reinforcement-learning
  - triage
  - ESI
pinned: true
---
```

## What still needs to be done / areas to improve

When I ask you to work on this project, here are the likely tasks in priority order:

1. **Fix any import errors** — the dual-import pattern (try/except ImportError) must work both when running locally (`PYTHONPATH=src:envs`) and inside Docker (`PYTHONPATH=/app/env`)
2. **Write the README.md** — must include: project description, action/observation space docs, ESI reference table, quick-start commands, Docker build instructions, HF Space deployment instructions, baseline scores table, reward function explanation
3. **Test the server locally** — `uvicorn server.app:app --port 8000`, hit `/health`, `/reset`, `/step`
4. **Run the test suite** — `pytest tests/ -v` — all tests must pass
5. **Build and test the Docker image** — `docker build -f server/Dockerfile -t clinical-triage-env .` then `docker run -p 8000:8000 -e TRIAGE_TASK=easy -e ENABLE_WEB_INTERFACE=true clinical-triage-env`
6. **Run openenv validate** — `pip install openenv-core && openenv validate --url http://localhost:8000`
7. **Run the baseline script** — `OPENAI_API_KEY=sk-... python baseline_inference.py --tasks easy medium hard`
8. **Deploy to HF Spaces** — `openenv push` or manual HF Space creation
9. **Enhance patient cases** — add more clinical nuance, edge cases, comorbidities
10. **Add .gitignore, LICENSE (Apache-2.0), CONTRIBUTING.md**

## How to run the server locally (quick reference)

```bash
# From inside the clinical_triage_env/ directory:
pip install fastapi uvicorn pydantic httpx

# Easy task with web UI
TRIAGE_TASK=easy ENABLE_WEB_INTERFACE=true PYTHONPATH=. uvicorn server.app:app --port 8000 --reload

# Then open: http://localhost:8000/web
# Health check: curl http://localhost:8000/health
# Reset: curl -X POST http://localhost:8000/reset
```

## Tone and approach for this codebase

- Always use **Pydantic v2** (`model_dump()` not `.dict()`, `model_validate()` not `.parse_obj()`)
- Always include the **dual-import try/except pattern** in every server file
- Keep the **reward logic inside the environment** class, never in app.py
- All patient case ground truth is in `server/patient_cases.py` — never hardcode ESI values elsewhere
- The `normalized_score()` method returns 0.0–1.0 — this is what the hackathon grader calls
- When writing new patient cases: be clinically authentic (realistic vitals, proper HPI format, real drug names, correct ESI per the v4 handbook)
- Never use `from openenv.core import ...` as the dependency may not be installed — implement the pattern from scratch

## My background

I work in healthcare AI. I have clinical domain knowledge and can validate that patient cases are realistic. When you generate new patient cases or modify existing ones, ask me to review them for clinical accuracy before finalising.

---

You now have complete context. Wait for my first task before doing anything. The first thing I will likely ask you is to read all the existing files, fix any issues, and get the server running locally.
