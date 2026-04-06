#!/usr/bin/env python3
"""
inference.py — Clinical Triage Navigator
=========================================
Mandatory stdout format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Environment variables (required):
  HF_TOKEN       Your HuggingFace / API key
  API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model ID       (default: Qwen/Qwen2.5-72B-Instruct)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from models import TriageAction, TriageObservation
from server.clinical_triage_environment import ClinicalTriageEnvironment

# ── Mandatory env vars ────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK             = "clinical-triage-navigator"
TASKS                 = ["easy", "medium", "hard"]
MAX_STEPS             = 20          # safety ceiling — all tasks have ≤10 patients
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE           = 0.1         # low temp for reproducibility
SEED                  = 42

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert emergency department triage nurse with 15 years of experience.
You will be presented with patient information and must assign an Emergency Severity Index (ESI) level.

ESI LEVELS:
  ESI 1 — Resuscitation: Immediately life-threatening. Needs life-saving intervention NOW.
           Examples: cardiac arrest, respiratory failure, unresponsive, severe shock.
  ESI 2 — Emergent: High-risk situation OR confused/lethargic OR severe pain/distress.
           Examples: STEMI, stroke, sepsis, severe allergic reaction, ectopic pregnancy.
  ESI 3 — Urgent: Stable but needs 2+ resources. Not high-risk but requires workup.
           Examples: abdominal pain with fever, DVT workup, moderate asthma, chest pain (low-risk).
  ESI 4 — Less-urgent: Needs exactly 1 resource.
           Examples: simple laceration needing repair, mild ankle sprain needing x-ray.
  ESI 5 — Non-urgent: Needs 0 resources. History/exam alone sufficient.
           Examples: simple URI, minor rash, prescription refill.

RESOURCE TYPES: IV_ACCESS, LABS, ECG, IMAGING_XRAY, IMAGING_CT, IMAGING_MRI,
                MEDICATION_IV, MEDICATION_ORAL, SPECIALIST_CONSULT, BEDSIDE_MONITORING, PROCEDURE

DISPOSITION: resus_bay (ESI 1), trauma_bay (major trauma), monitored_bed (ESI 2-3),
             fast_track (ESI 4), waiting_room (ESI 5)

CRITICAL: Under-triage (assigning too high an ESI number) is DANGEROUS. When in doubt,
assign the LOWER ESI number (higher acuity). Over-triage wastes resources but is safe.

Respond in valid JSON only:
{
  "esi_level": <integer 1-5>,
  "resources_requested": [<ResourceType>, ...],
  "disposition": "<disposition string>",
  "clinical_reasoning": "<brief rationale>"
}"""


# ── Mandatory logging ─────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Patient prompt builder ────────────────────────────────────────────────────

def patient_to_prompt(obs: TriageObservation) -> str:
    p = obs.current_patient
    v = p.vitals
    pmh      = ", ".join(p.past_medical_history)  or "None"
    meds     = ", ".join(p.current_medications)   or "None"
    allergies = ", ".join(p.allergies)             or "NKDA"

    lines = [
        f"PATIENT: {p.patient_id} | {p.age}y {p.sex}",
        f"CHIEF COMPLAINT: {p.chief_complaint}",
        f"HPI: {p.history_of_present_illness}",
        f"PMH: {pmh}  |  MEDS: {meds}  |  ALLERGIES: {allergies}",
        f"VITALS — HR:{v.heart_rate} BP:{v.blood_pressure_systolic}/{v.blood_pressure_diastolic} "
        f"RR:{v.respiratory_rate} SpO2:{v.spo2_percent}% Temp:{v.temperature_celsius}°C "
        f"GCS:{v.gcs_score} Pain:{v.pain_score}/10",
    ]
    if p.additional_context:
        lines.append(f"NOTE: {p.additional_context}")
    if obs.last_action_feedback:
        lines.insert(0, f"[Previous feedback: {obs.last_action_feedback}]")

    lines.append("\nProvide your triage decision as JSON:")
    return "\n".join(lines)


# ── LLM agent ─────────────────────────────────────────────────────────────────

def get_action(client: OpenAI, obs: TriageObservation) -> tuple[TriageAction, Optional[str]]:
    """Call LLM and parse triage action. Returns (action, error_string_or_None)."""
    error: Optional[str] = None
    try:
        kwargs = dict(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": patient_to_prompt(obs)},
            ],
            temperature=TEMPERATURE,
        )
        # seed is supported by OpenAI and some HF models — skip if it causes errors
        try:
            resp = client.chat.completions.create(**kwargs, seed=SEED)
        except Exception:
            resp = client.chat.completions.create(**kwargs)
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        error = f"json_parse_error:{e}"
        data = {}
    except Exception as e:
        error = f"llm_error:{e}"
        data = {}

    esi = max(1, min(5, int(data.get("esi_level", 3))))

    valid_resources = {
        "IV_ACCESS", "LABS", "ECG", "IMAGING_XRAY", "IMAGING_CT", "IMAGING_MRI",
        "MEDICATION_IV", "MEDICATION_ORAL", "SPECIALIST_CONSULT",
        "BEDSIDE_MONITORING", "PROCEDURE", "NONE",
    }
    raw_resources = data.get("resources_requested", ["LABS"])
    if not isinstance(raw_resources, list):
        raw_resources = ["LABS"]
    resources = [r for r in raw_resources if r in valid_resources] or ["LABS"]

    disposition = data.get("disposition", "monitored_bed")
    valid_dispositions = {"resus_bay", "trauma_bay", "monitored_bed", "fast_track", "waiting_room"}
    if disposition not in valid_dispositions:
        disposition = "monitored_bed"

    action = TriageAction(
        esi_level=esi,
        resources_requested=resources,
        disposition=disposition,
        clinical_reasoning=data.get("clinical_reasoning", ""),
    )
    return action, error


def action_str(action: TriageAction) -> str:
    """Compact single-line representation for [STEP] log."""
    resources = "+".join(action.resources_requested) or "NONE"
    return f"ESI{action.esi_level}|{action.disposition}|{resources}"


# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task: str) -> None:
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        env = ClinicalTriageEnvironment(task=task)
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if obs.done or obs.current_patient is None:
                break

            action, error = get_action(client, obs)
            obs = env.step(action)

            reward = obs.step_reward
            done   = obs.done

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str(action), reward=reward, done=done, error=error)

            if done:
                break

        score   = env.normalized_score()
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        # Always emit [END] even on exception
        error_msg = str(e)
        print(f"[DEBUG] Task {task} failed: {error_msg}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("[ERROR] Set HF_TOKEN environment variable.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        run_task(client, task)


if __name__ == "__main__":
    main()
