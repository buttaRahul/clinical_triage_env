"""
ClinicalTriageEnvironment — core simulation logic.

Implements the OpenEnv Environment ABC:
  reset()  → TriageObservation
  step()   → TriageObservation
  state    → TriageState  (property)

Reward design (per patient, summed over episode)
─────────────────────────────────────────────────
ESI accuracy
  Exact match          +1.00
  Off by 1             +0.40  (clinically acceptable)
  Off by 2             -0.20
  Off by ≥3            -0.60

Under-triage penalty (higher severity missed — DANGEROUS)
  Under by 1 level     -0.30  (on top of accuracy penalty)
  Under by 2 levels    -0.80
  Under by ≥3 levels   -1.50  (patient harmed)

Over-triage (lower-risk patient gets higher ESI — wastes resources)
  Over by 1 level      -0.10  (mild — over-cautious is OK)
  Over by 2 levels     -0.25
  Over by ≥3 levels    -0.50

Resource correctness (+0.10 per correct resource, −0.05 per false positive)
Disposition correct    +0.20
Clinical reasoning
  Hard task only — qualitative bonus 0.0–0.30

Idle/loop penalty     -0.01 per step (encourages efficiency)

Scores are normalized to [0.0, 1.0] at episode end.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Dict, List, Optional

try:
    from ..models import (
        ESILevel,
        PatientCard,
        TriageAction,
        TriageObservation,
        TriageState,
        VitalSigns,
    )
    from .patient_cases import TASK_CONFIGS
except ImportError:
    from models import (
        ESILevel,
        PatientCard,
        TriageAction,
        TriageObservation,
        TriageState,
        VitalSigns,
    )
    from server.patient_cases import TASK_CONFIGS


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _build_patient_card(case_dict: Dict) -> PatientCard:
    return PatientCard(
        patient_id=case_dict["patient_id"],
        age=case_dict["age"],
        sex=case_dict["sex"],
        chief_complaint=case_dict["chief_complaint"],
        history_of_present_illness=case_dict["history_of_present_illness"],
        past_medical_history=case_dict["past_medical_history"],
        current_medications=case_dict["current_medications"],
        allergies=case_dict["allergies"],
        vitals=VitalSigns(**case_dict["vitals"]),
        additional_context=case_dict.get("additional_context", ""),
    )


def _esi_accuracy_reward(assigned: int, ground_truth: int) -> float:
    """
    Asymmetric reward — under-triage is clinically dangerous and penalised 5× harder.

    delta = assigned − ground_truth
      negative = over-triage  (agent called MORE urgent than needed — cautious, safe)
      positive = under-triage (agent called LESS urgent than needed — dangerous)

    delta  | meaning                          | reward
    -------+----------------------------------+-------
      0    | perfect                          | +1.0
     -1    | slight over-triage (cautious)    | +0.5   ← rewarded as safe heuristic
     -2    | significant over-triage          | -0.2
    ≤-3    | extreme over-triage              | -0.5
     +1    | slight under-triage              | -1.0
     +2    | significant under-triage         | -2.5
    ≥+3    | critical miss (ESI-1 → ESI 4+)  | -5.0
    """
    delta = assigned - ground_truth
    if delta == 0:
        return 1.0
    elif delta == -1:
        return 0.5
    elif delta == -2:
        return -0.2
    elif delta <= -3:
        return -0.5
    elif delta == 1:
        return -1.0
    elif delta == 2:
        return -2.5
    else:  # delta >= 3
        return -5.0


def _sentinel_reward(sentinel_condition: Optional[str], escalation: str) -> float:
    """
    Reward component for sentinel condition escalation.
    Fires independently of ESI accuracy — you can get the ESI right
    but still miss the sentinel (and vice versa).

    sentinel_condition  | escalation       | reward
    --------------------+------------------+-------
    present             | ACTIVATE_TEAM    | +0.4
    present             | ALERT_SENIOR     | +0.4
    present             | NONE             | -3.0  ← missed sentinel
    None                | ACTIVATE_TEAM    | -0.2  ← false alarm (minor penalty)
    None                | ALERT_SENIOR     | -0.1  ← false alarm (minor penalty)
    None                | NONE             |  0.0
    """
    if sentinel_condition:
        return 0.4 if escalation in ("ACTIVATE_TEAM", "ALERT_SENIOR") else -3.0
    else:
        if escalation == "ACTIVATE_TEAM":
            return -0.2
        elif escalation == "ALERT_SENIOR":
            return -0.1
        return 0.0


def _resource_reward(requested: List[str], ground_truth: List[str]) -> float:
    gt_set = set(ground_truth)
    req_set = set(requested) - {"NONE"}
    if "NONE" in ground_truth:
        # Patient needs no resources — any request is wrong
        return -0.05 * len(req_set)
    true_pos = len(req_set & gt_set)
    false_pos = len(req_set - gt_set)
    return (true_pos * 0.10) - (false_pos * 0.05)


def _disposition_reward(assigned: str, ground_truth: str) -> float:
    return 0.20 if assigned == ground_truth else 0.0


def _build_feedback(
    action: TriageAction,
    gt_esi: int,
    gt_resources: List[str],
    gt_disposition: str,
    step_reward: float,
    sentinel: Optional[str] = None,
    r_sentinel: float = 0.0,
) -> str:
    lines = []
    diff = action.esi_level - gt_esi
    if diff == 0:
        lines.append(f"ESI CORRECT: Level {action.esi_level} — accurate assignment.")
    elif diff > 0:
        lines.append(
            f"UNDER-TRIAGE: You assigned ESI {action.esi_level}, "
            f"correct was ESI {gt_esi}. Patient is MORE critical than assessed. "
            f"Under-triage is dangerous — penalties applied."
        )
    else:
        lines.append(
            f"OVER-TRIAGE: You assigned ESI {action.esi_level}, "
            f"correct was ESI {gt_esi}. Patient is less critical. "
            f"Over-triage wastes resus resources."
        )

    gt_set = set(gt_resources) - {"NONE"}
    req_set = set(action.resources_requested) - {"NONE"}
    missed = gt_set - req_set
    extra = req_set - gt_set
    if missed:
        lines.append(f"MISSED resources: {', '.join(sorted(missed))}")
    if extra:
        lines.append(f"UNNECESSARY resources requested: {', '.join(sorted(extra))}")
    if not missed and not extra:
        lines.append("Resource allocation: CORRECT.")

    if action.disposition != gt_disposition:
        lines.append(
            f"Disposition: You chose '{action.disposition}', "
            f"correct is '{gt_disposition}'."
        )
    else:
        lines.append(f"Disposition: CORRECT — '{action.disposition}'.")

    # Sentinel feedback
    if sentinel:
        if action.escalation in ("ACTIVATE_TEAM", "ALERT_SENIOR"):
            lines.append(f"SENTINEL ESCALATION CORRECT: {sentinel} — {action.escalation} ({r_sentinel:+.1f})")
        else:
            lines.append(
                f"SENTINEL MISSED: Patient has {sentinel} — escalation should be "
                f"ACTIVATE_TEAM or ALERT_SENIOR, not NONE. Penalty: {r_sentinel:+.1f}"
            )
    elif action.escalation != "NONE":
        lines.append(f"FALSE ESCALATION: No sentinel condition — {action.escalation} penalty ({r_sentinel:+.1f})")

    lines.append(f"Step reward: {step_reward:+.2f}")
    return " | ".join(lines)


# ──────────────────────────────────────────────────────────────
# Main environment class
# ──────────────────────────────────────────────────────────────

class ClinicalTriageEnvironment:
    """
    OpenEnv-compatible ER triage simulation.

    Usage
    -----
    env = ClinicalTriageEnvironment(task="easy")
    obs = env.reset()
    while not obs.done:
        action = agent.act(obs)
        obs = env.step(action)
    score = env.state.cumulative_reward
    """

    def __init__(self, task: str = "easy"):
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS)}")
        self._task_name = task
        self._task_cfg = TASK_CONFIGS[task]
        self._episode_id: str = ""
        self._cases: List[Dict] = []
        self._queue_index: int = 0
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._correct_count: int = 0
        self._over_triage: int = 0
        self._under_triage: int = 0
        self._sentinel_total: int = 0
        self._sentinel_correct: int = 0
        self._done: bool = False
        self._last_obs: Optional[TriageObservation] = None

    # ── OpenEnv interface ───────────────────────────────────────

    def reset(self) -> TriageObservation:
        """Start a fresh episode. Returns observation of first patient."""
        self._episode_id = str(uuid.uuid4())[:8]
        self._cases = deepcopy(self._task_cfg["cases"])
        self._queue_index = 0
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._correct_count = 0
        self._over_triage = 0
        self._under_triage = 0
        self._sentinel_total = 0
        self._sentinel_correct = 0
        self._done = False

        obs = self._make_observation(
            step_reward=0.0,
            feedback="",
            last_correct=None,
            last_assigned=None,
            last_gt=None,
        )
        self._last_obs = obs
        return obs

    def step(self, action: TriageAction) -> TriageObservation:
        """Process triage decision for the current patient."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._queue_index >= len(self._cases):
            raise RuntimeError("No more patients in queue.")

        current_case = self._cases[self._queue_index]
        gt_esi: int = current_case["ground_truth_esi"]
        gt_resources: List[str] = current_case["ground_truth_resources"]
        gt_disposition: str = current_case["ground_truth_disposition"]
        sentinel: Optional[str] = current_case.get("sentinel_condition")

        # ── Compute reward ──────────────────────────────────────
        r_esi        = _esi_accuracy_reward(action.esi_level, gt_esi)
        r_sentinel   = _sentinel_reward(sentinel, action.escalation)
        r_resources  = _resource_reward(action.resources_requested, gt_resources)
        r_disposition = _disposition_reward(action.disposition, gt_disposition)
        r_idle       = -0.01  # per-step efficiency penalty

        step_reward = r_esi + r_sentinel + r_resources + r_disposition + r_idle

        # Hard task: bonus for good clinical reasoning (length + keyword heuristic)
        if self._task_name == "hard" and action.clinical_reasoning:
            reasoning_score = min(0.30, len(action.clinical_reasoning) / 400 * 0.30)
            step_reward += reasoning_score

        self._cumulative_reward += step_reward
        self._step_count += 1

        # Track ESI accuracy
        esi_correct = action.esi_level == gt_esi
        if esi_correct:
            self._correct_count += 1
        diff = action.esi_level - gt_esi
        if diff > 0:
            self._under_triage += 1
        elif diff < 0:
            self._over_triage += 1

        # Track sentinel accuracy
        if sentinel:
            self._sentinel_total += 1
            if action.escalation in ("ACTIVATE_TEAM", "ALERT_SENIOR"):
                self._sentinel_correct += 1

        feedback = _build_feedback(
            action, gt_esi, gt_resources, gt_disposition, step_reward,
            sentinel=sentinel, r_sentinel=r_sentinel,
        )

        # Advance queue
        self._queue_index += 1
        self._done = self._queue_index >= len(self._cases)

        obs = self._make_observation(
            step_reward=step_reward,
            feedback=feedback,
            last_correct=esi_correct,
            last_assigned=action.esi_level,
            last_gt=gt_esi,
        )
        self._last_obs = obs
        return obs

    @property
    def state(self) -> TriageState:
        """Return current episode metadata."""
        n_triaged = self._queue_index
        n_total = len(self._cases)
        accuracy = self._correct_count / max(n_triaged, 1)
        return TriageState(
            episode_id=self._episode_id,
            task_name=self._task_name,
            step_count=self._step_count,
            patients_triaged=n_triaged,
            patients_total=n_total,
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            esi_accuracy=round(accuracy, 4),
            over_triage_count=self._over_triage,
            under_triage_count=self._under_triage,
        )

    # ── Internal helpers ────────────────────────────────────────

    def _make_observation(
        self,
        step_reward: float,
        feedback: str,
        last_correct: Optional[bool],
        last_assigned: Optional[int],
        last_gt: Optional[int],
    ) -> TriageObservation:
        current_patient: Optional[PatientCard] = None
        if not self._done and self._queue_index < len(self._cases):
            current_patient = _build_patient_card(self._cases[self._queue_index])

        return TriageObservation(
            current_patient=current_patient,
            last_action_feedback=feedback,
            last_esi_correct=last_correct,
            last_esi_assigned=last_assigned,
            last_esi_ground_truth=last_gt,
            patients_remaining=max(0, len(self._cases) - self._queue_index),
            patients_triaged=self._queue_index,
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            task_name=self._task_name,
            task_description=self._task_cfg["description"],
        )

    def normalized_score(self) -> float:
        """
        Returns final normalized score in [0.0, 1.0] for agent grader.

        Task-specific formulas (simplified — no deterioration component yet):
          easy   : 0.7 × esi_accuracy + 0.3 × sentinel_accuracy
          medium : 0.6 × esi_accuracy + 0.4 × sentinel_accuracy
          hard   : 0.55 × esi_accuracy + 0.45 × sentinel_accuracy
        """
        if not self._cases:
            return 0.0
        n = self._queue_index or len(self._cases)

        esi_score = self._correct_count / n

        # sentinel_score = 1.0 if no sentinel patients (full marks by default)
        if self._sentinel_total > 0:
            sentinel_score = self._sentinel_correct / self._sentinel_total
        else:
            sentinel_score = 1.0

        if self._task_name == "easy":
            combined = 0.7 * esi_score + 0.3 * sentinel_score
        elif self._task_name == "medium":
            combined = 0.6 * esi_score + 0.4 * sentinel_score
        else:  # hard
            combined = 0.55 * esi_score + 0.45 * sentinel_score

        return round(min(1.0, max(0.0, combined)), 4)
