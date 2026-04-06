"""
ClinicalTriageEnv — OpenEnv client.

Usage (async):
    from clinical_triage_env import ClinicalTriageEnv, TriageAction

    async with ClinicalTriageEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        while not obs.done:
            action = TriageAction(
                esi_level=2,
                resources_requested=["IV_ACCESS", "LABS", "ECG"],
                disposition="monitored_bed",
                clinical_reasoning="Chest pain with diaphoresis — high STEMI risk.",
            )
            obs = await env.step(action)

Usage (sync):
    with ClinicalTriageEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        obs = env.step(action)
"""

from __future__ import annotations

import httpx
from typing import Any, Dict, Optional

try:
    from .models import TriageAction, TriageObservation, TriageState
except ImportError:
    from models import TriageAction, TriageObservation, TriageState


class _SyncClinicalTriageEnv:
    """Synchronous wrapper for use in scripts and notebooks."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def reset(self) -> TriageObservation:
        r = self._client.post(f"{self._base}/reset")
        r.raise_for_status()
        return TriageObservation(**r.json())

    def step(self, action: TriageAction) -> TriageObservation:
        r = self._client.post(
            f"{self._base}/step",
            json=action.model_dump(),
        )
        r.raise_for_status()
        return TriageObservation(**r.json())

    def state(self) -> TriageState:
        r = self._client.get(f"{self._base}/state")
        r.raise_for_status()
        return TriageState(**r.json())

    def score(self) -> float:
        r = self._client.get(f"{self._base}/score")
        r.raise_for_status()
        return r.json()["score"]

    def health(self) -> Dict[str, Any]:
        r = self._client.get(f"{self._base}/health")
        r.raise_for_status()
        return r.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ClinicalTriageEnv:
    """
    Async OpenEnv client for the Clinical Triage Navigator.

    Parameters
    ----------
    base_url : str
        URL of a running Clinical Triage server.
    timeout : float
        HTTP timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def sync(self) -> _SyncClinicalTriageEnv:
        """Return a synchronous wrapper for script/notebook usage."""
        return _SyncClinicalTriageEnv(self._base, self._timeout)

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()

    async def reset(self) -> TriageObservation:
        r = await self._client.post(f"{self._base}/reset")
        r.raise_for_status()
        return TriageObservation(**r.json())

    async def step(self, action: TriageAction) -> TriageObservation:
        r = await self._client.post(
            f"{self._base}/step",
            json=action.model_dump(),
        )
        r.raise_for_status()
        return TriageObservation(**r.json())

    async def state(self) -> TriageState:
        r = await self._client.get(f"{self._base}/state")
        r.raise_for_status()
        return TriageState(**r.json())

    async def score(self) -> float:
        r = await self._client.get(f"{self._base}/score")
        r.raise_for_status()
        return r.json()["score"]

    async def health(self) -> Dict[str, Any]:
        r = await self._client.get(f"{self._base}/health")
        r.raise_for_status()
        return r.json()
