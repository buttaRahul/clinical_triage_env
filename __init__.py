"""
Clinical Triage Navigator — OpenEnv Environment

Quick start (sync):
    from clinical_triage_env import ClinicalTriageEnv, TriageAction

    with ClinicalTriageEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        while not obs.done:
            action = TriageAction(
                esi_level=2,
                resources_requested=["IV_ACCESS", "LABS", "ECG"],
                disposition="monitored_bed",
                clinical_reasoning="Chest pain, diaphoresis — STEMI risk.",
            )
            obs = env.step(action)
        print("Score:", env.score())
"""

from .client import ClinicalTriageEnv
from .models import (
    ESILevel,
    PatientCard,
    ResourceType,
    TriageAction,
    TriageObservation,
    TriageState,
    VitalSigns,
)

__all__ = [
    "ClinicalTriageEnv",
    "TriageAction",
    "TriageObservation",
    "TriageState",
    "PatientCard",
    "VitalSigns",
    "ESILevel",
    "ResourceType",
]
