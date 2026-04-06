#!/usr/bin/env python3
"""
Baseline inference script — Clinical Triage Navigator.

Runs an LLM agent (via OpenAI-compatible API) against all three tasks
and reports reproducible scores.

Usage
-----
    # Against a running server
    OPENAI_API_KEY=sk-... python baseline_inference.py --base-url http://localhost:8000

    # Against HF Space
    OPENAI_API_KEY=sk-... python baseline_inference.py \
        --base-url https://your-space.hf.space \
        --model gpt-4o-mini \
        --tasks easy medium hard

Requirements
------------
    pip install openai httpx pydantic

Reproducibility
---------------
All random elements are seeded. Patient cases are deterministic.
Results are written to outputs/evals/baseline_TIMESTAMP.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# Import environment models
# ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from models import TriageAction, TriageObservation, TriageState


# ──────────────────────────────────────────────────────────────
# System prompt for the triage agent
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert emergency department triage nurse with 15 years of experience.
You will be presented with patient information and must assign an Emergency Severity Index (ESI) level.

ESI LEVELS:
  ESI 1 — Resuscitation: Immediately life-threatening. Patient needs life-saving intervention NOW.
           Examples: cardiac arrest, respiratory failure, unresponsive, severe shock.
  ESI 2 — Emergent: High-risk situation OR confused/lethargic OR severe pain/distress.
           Examples: STEMI, stroke, sepsis, severe allergic reaction, ectopic pregnancy.
  ESI 3 — Urgent: Stable but needs ≥2 resources. Not high-risk but requires workup.
           Examples: abdominal pain with fever, DVT workup, moderate asthma, chest pain (low-risk).
  ESI 4 — Less-urgent: Needs exactly 1 resource (1 lab, 1 x-ray, or 1 procedure).
           Examples: simple laceration repair, mild ankle sprain needing x-ray, strep throat.
  ESI 5 — Non-urgent: Needs 0 resources. Can be treated with history/exam alone.
           Examples: simple URI, minor rash, prescription refill.

RESOURCE TYPES: IV_ACCESS, LABS, ECG, IMAGING_XRAY, IMAGING_CT, IMAGING_MRI,
                MEDICATION_IV, MEDICATION_ORAL, SPECIALIST_CONSULT, BEDSIDE_MONITORING, PROCEDURE

DISPOSITION: resus_bay (ESI1), trauma_bay (major trauma), monitored_bed (ESI2-3),
             fast_track (ESI4), waiting_room (ESI5)

CRITICAL PRINCIPLE: Under-triage (assigning too high an ESI number = under-estimating severity)
is DANGEROUS and heavily penalized. When in doubt, assign LOWER ESI number (higher acuity).

You must respond in valid JSON matching this schema:
{
  "esi_level": <integer 1-5>,
  "resources_requested": [<list of ResourceType strings>],
  "disposition": "<one of the disposition options>",
  "clinical_reasoning": "<brief explanation of your decision>"
}
"""


def patient_to_prompt(patient_data: Dict[str, Any]) -> str:
    """Format a PatientCard dict as a readable prompt."""
    v = patient_data["vitals"]
    pmh = ", ".join(patient_data["past_medical_history"]) or "None"
    meds = ", ".join(patient_data["current_medications"]) or "None"
    allergies = ", ".join(patient_data["allergies"]) or "NKDA"
    ctx = patient_data.get("additional_context", "")

    prompt = f"""PATIENT: {patient_data['patient_id']} | {patient_data['age']}y {patient_data['sex']}

CHIEF COMPLAINT: {patient_data['chief_complaint']}

HISTORY OF PRESENT ILLNESS:
{patient_data['history_of_present_illness']}

PAST MEDICAL HISTORY: {pmh}
CURRENT MEDICATIONS: {meds}
ALLERGIES: {allergies}

VITAL SIGNS:
  Heart Rate: {v['heart_rate']} bpm
  Blood Pressure: {v['blood_pressure_systolic']}/{v['blood_pressure_diastolic']} mmHg
  Respiratory Rate: {v['respiratory_rate']}/min
  SpO2: {v['spo2_percent']}%
  Temperature: {v['temperature_celsius']}°C
  GCS: {v['gcs_score']}/15
  Pain Score: {v['pain_score']}/10
"""
    if ctx:
        prompt += f"\nADDITIONAL CONTEXT:\n{ctx}\n"
    prompt += "\nProvide your triage decision as JSON:"
    return prompt


# ──────────────────────────────────────────────────────────────
# LLM agent
# ──────────────────────────────────────────────────────────────

class TriageAgent:
    def __init__(self, client: OpenAI, model: str, temperature: float = 0.1):
        self.client = client
        self.model = model
        self.temperature = temperature

    def act(self, observation: TriageObservation) -> TriageAction:
        """Given an observation, return a TriageAction."""
        if observation.current_patient is None:
            raise ValueError("No current patient in observation.")

        patient_dict = observation.current_patient.model_dump()
        user_prompt = patient_to_prompt(patient_dict)

        # Add feedback from previous step if available
        if observation.last_action_feedback:
            user_prompt = (
                f"[Previous feedback: {observation.last_action_feedback}]\n\n"
                + user_prompt
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            seed=42,  # reproducibility
        )

        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [WARN] Failed to parse JSON: {raw[:200]}")
            data = {
                "esi_level": 3,
                "resources_requested": ["LABS"],
                "disposition": "monitored_bed",
                "clinical_reasoning": "JSON parse error — defaulting to ESI 3.",
            }

        # Validate and clamp
        esi = int(data.get("esi_level", 3))
        esi = max(1, min(5, esi))
        resources = data.get("resources_requested", ["LABS"])
        if not isinstance(resources, list):
            resources = ["LABS"]
        disposition = data.get("disposition", "monitored_bed")
        if disposition not in ["resus_bay", "trauma_bay", "monitored_bed", "fast_track", "waiting_room"]:
            disposition = "monitored_bed"

        return TriageAction(
            esi_level=esi,
            resources_requested=resources,
            disposition=disposition,
            clinical_reasoning=data.get("clinical_reasoning", ""),
        )


# ──────────────────────────────────────────────────────────────
# HTTP helpers (sync, no openenv-core dependency)
# ──────────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str, task: str):
        self.base = base_url.rstrip("/")
        self.task = task
        self._http = httpx.Client(timeout=30.0)

    def reset(self) -> TriageObservation:
        r = self._http.post(f"{self.base}/reset")
        r.raise_for_status()
        return TriageObservation(**r.json())

    def step(self, action: TriageAction) -> TriageObservation:
        r = self._http.post(
            f"{self.base}/step",
            json=action.model_dump(),
        )
        r.raise_for_status()
        return TriageObservation(**r.json())

    def state(self) -> TriageState:
        r = self._http.get(f"{self.base}/state")
        r.raise_for_status()
        return TriageState(**r.json())

    def score(self) -> float:
        r = self._http.get(f"{self.base}/score")
        r.raise_for_status()
        return r.json()["score"]

    def close(self):
        self._http.close()


# ──────────────────────────────────────────────────────────────
# Task runner
# ──────────────────────────────────────────────────────────────

def run_task(
    agent: TriageAgent,
    base_url: str,
    task: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one full episode for a task and return results."""
    env = EnvClient(base_url, task)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task.upper()}")
        print(f"{'='*60}")

    obs = env.reset()
    step_records = []

    while not obs.done:
        if obs.current_patient is None:
            break

        patient_id = obs.current_patient.patient_id
        if verbose:
            print(f"\n  Patient {patient_id} | {obs.current_patient.age}y "
                  f"{obs.current_patient.sex} | {obs.current_patient.chief_complaint}")

        t0 = time.time()
        action = agent.act(obs)
        latency = time.time() - t0

        if verbose:
            print(f"  → Agent: ESI {action.esi_level} | "
                  f"{action.disposition} | "
                  f"Resources: {action.resources_requested}")

        obs = env.step(action)

        if verbose and obs.last_esi_ground_truth is not None:
            correct = "CORRECT" if obs.last_esi_correct else "WRONG"
            print(f"  ← [{correct}] GT ESI={obs.last_esi_ground_truth} | "
                  f"Step reward: {obs.step_reward:+.3f}")

        step_records.append({
            "patient_id": patient_id,
            "esi_assigned": action.esi_level,
            "esi_ground_truth": obs.last_esi_ground_truth,
            "esi_correct": obs.last_esi_correct,
            "disposition": action.disposition,
            "resources_requested": action.resources_requested,
            "step_reward": obs.step_reward,
            "latency_s": round(latency, 3),
        })

    final_state = env.state()
    final_score = env.score()

    if verbose:
        print(f"\n  --- Episode Summary ---")
        print(f"  Patients triaged: {final_state.patients_triaged}")
        print(f"  ESI accuracy:     {final_state.esi_accuracy:.1%}")
        print(f"  Under-triage:     {final_state.under_triage_count}")
        print(f"  Over-triage:      {final_state.over_triage_count}")
        print(f"  Cumulative reward:{final_state.cumulative_reward:.3f}")
        print(f"  Normalized score: {final_score:.4f}  ({final_score*100:.1f}%)")

    env.close()

    return {
        "task": task,
        "score": final_score,
        "esi_accuracy": final_state.esi_accuracy,
        "under_triage_count": final_state.under_triage_count,
        "over_triage_count": final_state.over_triage_count,
        "cumulative_reward": final_state.cumulative_reward,
        "patients_triaged": final_state.patients_triaged,
        "steps": step_records,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference for Clinical Triage Navigator"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="URL of running environment server",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM sampling temperature (default 0.1 for reproducibility)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    # Health check
    try:
        r = httpx.get(f"{args.base_url}/health", timeout=10)
        r.raise_for_status()
        print(f"Server healthy: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.base_url}: {e}")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)
    agent = TriageAgent(openai_client, model=args.model, temperature=args.temperature)

    all_results = []
    for task in args.tasks:
        result = run_task(
            agent=agent,
            base_url=args.base_url,
            task=task,
            verbose=not args.quiet,
        )
        all_results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS — {args.model}")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Score':>8} {'ESI Acc':>9} {'Under':>7} {'Over':>6}")
    print(f"  {'-'*46}")
    for r in all_results:
        print(
            f"  {r['task']:<10} {r['score']:>8.4f} "
            f"{r['esi_accuracy']:>9.1%} "
            f"{r['under_triage_count']:>7} "
            f"{r['over_triage_count']:>6}"
        )
    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    print(f"  {'-'*46}")
    print(f"  {'AVERAGE':<10} {avg_score:>8.4f}")
    print(f"{'='*60}")

    # Save results
    output_dir = Path("outputs/evals")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"baseline_{args.model}_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "model": args.model,
        "temperature": args.temperature,
        "base_url": args.base_url,
        "average_score": avg_score,
        "tasks": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
