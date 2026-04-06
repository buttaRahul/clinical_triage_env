"""
Clinical Triage Navigator — typed models.

Action  : TriageAction   — agent's triage decision for one patient
Observation : TriageObservation — what the agent sees after each step
State   : TriageState    — episode metadata (task, step, done)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Shared enums / literals
# ─────────────────────────────────────────────

ESILevel = Literal[1, 2, 3, 4, 5]
"""
ESI 1 — Resuscitation        (immediately life-threatening)
ESI 2 — Emergent             (high-risk situation / confused / severe pain)
ESI 3 — Urgent               (stable but needs ≥2 resources)
ESI 4 — Less-urgent          (needs 1 resource)
ESI 5 — Non-urgent           (needs 0 resources / can wait)
"""

ResourceType = Literal[
    "IV_ACCESS",
    "LABS",
    "ECG",
    "IMAGING_XRAY",
    "IMAGING_CT",
    "IMAGING_MRI",
    "MEDICATION_IV",
    "MEDICATION_ORAL",
    "SPECIALIST_CONSULT",
    "BEDSIDE_MONITORING",
    "PROCEDURE",
    "NONE",
]


# ─────────────────────────────────────────────
# Patient card — shown to the agent
# ─────────────────────────────────────────────

class VitalSigns(BaseModel):
    heart_rate: int = Field(description="Beats per minute")
    blood_pressure_systolic: int = Field(description="mmHg")
    blood_pressure_diastolic: int = Field(description="mmHg")
    respiratory_rate: int = Field(description="Breaths per minute")
    spo2_percent: float = Field(description="Oxygen saturation %")
    temperature_celsius: float = Field(description="Body temperature")
    gcs_score: int = Field(description="Glasgow Coma Scale 3-15")
    pain_score: int = Field(description="0-10 numeric pain scale")


class PatientCard(BaseModel):
    patient_id: str
    age: int
    sex: Literal["M", "F", "Other"]
    chief_complaint: str = Field(description="Primary reason for ER visit")
    history_of_present_illness: str = Field(
        description="Brief HPI — onset, duration, associated symptoms"
    )
    past_medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vitals: VitalSigns
    additional_context: Optional[str] = None


# ─────────────────────────────────────────────
# ACTION — what the agent submits
# ─────────────────────────────────────────────

class TriageAction(BaseModel):
    """
    The agent's triage decision for the currently active patient.

    Fields
    ------
    esi_level : int (1-5)
        Assigned ESI triage level.
    resources_requested : list[ResourceType]
        Resources the agent believes this patient needs.
        Directly determines ESI 3/4/5 discrimination.
    clinical_reasoning : str
        Free-text rationale. Used for partial-credit scoring on hard tasks.
    disposition : str
        Where the patient should go: 'resus_bay', 'trauma_bay',
        'monitored_bed', 'fast_track', 'waiting_room'.
    """

    esi_level: ESILevel
    resources_requested: List[ResourceType] = Field(
        default_factory=list,
        description="Resources this patient needs (determines ESI 3/4/5)",
    )
    clinical_reasoning: str = Field(
        default="",
        description="Agent's reasoning for the triage decision",
    )
    disposition: Literal[
        "resus_bay", "trauma_bay", "monitored_bed", "fast_track", "waiting_room"
    ] = Field(description="Recommended patient placement")


# ─────────────────────────────────────────────
# OBSERVATION — what the agent receives
# ─────────────────────────────────────────────

class TriageObservation(BaseModel):
    """Returned after every step() and reset()."""

    # Current patient to triage
    current_patient: Optional[PatientCard] = Field(
        default=None,
        description="Patient card for the agent to triage. None if episode done.",
    )

    # Feedback on the last action (empty on reset)
    last_action_feedback: str = Field(
        default="",
        description="Clinical feedback on the previous triage decision",
    )
    last_esi_correct: Optional[bool] = Field(
        default=None,
        description="Whether the last ESI assignment was correct",
    )
    last_esi_assigned: Optional[int] = Field(default=None)
    last_esi_ground_truth: Optional[int] = Field(default=None)

    # Queue status
    patients_remaining: int = Field(
        default=0, description="Patients left in queue this episode"
    )
    patients_triaged: int = Field(default=0, description="Patients completed so far")

    # Reward signals
    step_reward: float = Field(default=0.0, description="Reward for the last action")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far")

    # Terminal flag
    done: bool = Field(default=False)

    # Task metadata
    task_name: str = Field(default="")
    task_description: str = Field(default="")


# ─────────────────────────────────────────────
# STATE — episode metadata
# ─────────────────────────────────────────────

class TriageState(BaseModel):
    """Returned by state(). Describes the current episode."""

    episode_id: str
    task_name: str
    step_count: int
    patients_triaged: int
    patients_total: int
    cumulative_reward: float
    done: bool
    esi_accuracy: float = Field(
        description="Fraction of correct ESI assignments so far"
    )
    over_triage_count: int = Field(
        description="Times agent triaged HIGHER severity than needed"
    )
    under_triage_count: int = Field(
        description="Times agent triaged LOWER severity than needed (dangerous)"
    )
