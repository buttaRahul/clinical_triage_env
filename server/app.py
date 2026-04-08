"""
FastAPI application for the Clinical Triage Navigator.

Exposes the standard OpenEnv endpoints:
  POST /reset          → TriageObservation
  POST /step           → TriageObservation
  GET  /state          → TriageState
  GET  /health         → {status: ok}
  GET  /web            → Interactive web UI (if ENABLE_WEB_INTERFACE=true)

Task is selected via the TRIAGE_TASK environment variable:
  TRIAGE_TASK=easy   (default)
  TRIAGE_TASK=medium
  TRIAGE_TASK=hard
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

try:
    from ..models import TriageAction, TriageObservation, TriageState
    from .clinical_triage_environment import ClinicalTriageEnvironment
except ImportError:
    from models import TriageAction, TriageObservation, TriageState
    from server.clinical_triage_environment import ClinicalTriageEnvironment


# ──────────────────────────────────────────────
# App factory (OpenEnv convention: create_app)
# ──────────────────────────────────────────────

def create_app(task: Optional[str] = None) -> FastAPI:
    task = task or os.environ.get("TRIAGE_TASK", "easy")

    app = FastAPI(
        title="Clinical Triage Navigator",
        description=(
            "ER triage simulation for AI agent training. "
            "Agent assigns ESI levels, requests resources, and disposes patients."
        ),
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Singleton environment per session (OpenEnv WebSocket handles isolation)
    env = ClinicalTriageEnvironment(task=task)

    @app.get("/health")
    async def health():
        return {"status": "ok", "task": task, "version": "1.0.0"}

    @app.post("/reset", response_model=TriageObservation)
    async def reset():
        """Reset the environment and return the first patient observation."""
        obs = env.reset()
        return obs

    @app.post("/step", response_model=TriageObservation)
    async def step(action: TriageAction):
        """Submit a triage decision and receive the next patient observation."""
        try:
            obs = env.step(action)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return obs

    @app.get("/state", response_model=TriageState)
    async def state():
        """Return current episode metadata."""
        return env.state

    @app.get("/score")
    async def score():
        """Return normalized episode score [0.0, 1.0]."""
        return {"score": env.normalized_score(), "task": task}

    # ── Web UI (for human testing) ──────────────────────────────
    if os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true":
        @app.get("/web", response_class=HTMLResponse)
        async def web_ui():
            return _web_ui_html(task)

    return app


# ──────────────────────────────────────────────
# Web interface HTML
# ──────────────────────────────────────────────

def _web_ui_html(task: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Clinical Triage Navigator — {task.upper()}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  .vitals {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
  .vital {{ background: #f5f5f5; padding: 6px 10px; border-radius: 4px; font-size: 0.85rem; }}
  select, textarea, input {{ width: 100%; margin: 4px 0 12px; padding: 6px; border: 1px solid #ccc; border-radius: 4px; }}
  button {{ background: #1a7a4a; color: white; border: none; padding: 10px 24px; border-radius: 4px; cursor: pointer; font-size: 1rem; }}
  button:hover {{ background: #145e38; }}
  .feedback {{ background: #f0f8ff; padding: 1rem; border-radius: 6px; border-left: 4px solid #378ADD; white-space: pre-wrap; font-size: 0.9rem; }}
  .reward {{ font-size: 1.1rem; font-weight: 600; }}
  .done {{ background: #e8f5e9; border-left: 4px solid #1D9E75; }}
  label {{ font-weight: 500; font-size: 0.9rem; }}
  .resources {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; }}
  .resources label {{ font-weight: 400; }}
</style>
</head>
<body>
<h1>Clinical Triage Navigator — Task: {task.upper()}</h1>
<div id="app">
  <button onclick="resetEnv()">Start / Reset Episode</button>
  <div id="patient-section" style="display:none">
    <div class="card" id="patient-card"></div>
    <div class="card">
      <h3>Your Triage Decision</h3>
      <label>ESI Level (1=Resuscitation → 5=Non-urgent)</label>
      <select id="esi">
        <option value="1">ESI 1 — Resuscitation</option>
        <option value="2">ESI 2 — Emergent</option>
        <option value="3" selected>ESI 3 — Urgent</option>
        <option value="4">ESI 4 — Less-urgent</option>
        <option value="5">ESI 5 — Non-urgent</option>
      </select>
      <label>Resources Requested</label>
      <div class="resources">
        {_resource_checkboxes()}
      </div>
      <label style="margin-top:12px;display:block">Disposition</label>
      <select id="disposition">
        <option value="resus_bay">Resus Bay</option>
        <option value="trauma_bay">Trauma Bay</option>
        <option value="monitored_bed" selected>Monitored Bed</option>
        <option value="fast_track">Fast Track</option>
        <option value="waiting_room">Waiting Room</option>
      </select>
      <label>Clinical Reasoning (required for hard task)</label>
      <textarea id="reasoning" rows="3" placeholder="Explain your triage rationale..."></textarea>
      <button onclick="submitAction()">Submit Triage Decision</button>
    </div>
    <div id="feedback-section" style="display:none">
      <div class="card feedback" id="feedback-text"></div>
    </div>
  </div>
  <div id="done-section" style="display:none" class="card done">
    <h2>Episode Complete</h2>
    <div id="final-score"></div>
    <button onclick="resetEnv()">Start New Episode</button>
  </div>
</div>
<script>
async function resetEnv() {{
  const r = await fetch('/reset', {{method:'POST'}});
  const obs = await r.json();
  document.getElementById('done-section').style.display='none';
  document.getElementById('feedback-section').style.display='none';
  document.getElementById('patient-section').style.display='block';
  renderPatient(obs);
}}
async function submitAction() {{
  const esi = parseInt(document.getElementById('esi').value);
  const resources = [...document.querySelectorAll('.res-check:checked')].map(c=>c.value);
  if(resources.length===0) resources.push('NONE');
  const disposition = document.getElementById('disposition').value;
  const reasoning = document.getElementById('reasoning').value;
  const action = {{esi_level:esi, resources_requested:resources, disposition, clinical_reasoning:reasoning}};
  const r = await fetch('/step', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(action)}});
  const obs = await r.json();
  document.getElementById('feedback-section').style.display='block';
  document.getElementById('feedback-text').textContent = obs.last_action_feedback;
  if(obs.done) {{
    const sc = await (await fetch('/score')).json();
    document.getElementById('patient-section').style.display='none';
    document.getElementById('done-section').style.display='block';
    document.getElementById('final-score').innerHTML =
      '<p>Patients triaged: '+obs.patients_triaged+'</p>'+
      '<p class="reward">Final score: '+(sc.score*100).toFixed(1)+'%</p>'+
      '<p>Cumulative reward: '+obs.cumulative_reward.toFixed(2)+'</p>';
  }} else {{
    renderPatient(obs);
  }}
}}
function renderPatient(obs) {{
  if(!obs.current_patient) return;
  const p = obs.current_patient;
  const v = p.vitals;
  document.getElementById('patient-card').innerHTML = `
    <h3>Patient ${{p.patient_id}} — ${{p.age}}y ${{p.sex}} &nbsp;<small style="color:#888">(${{obs.patients_triaged+1}}/${{obs.patients_triaged+obs.patients_remaining}})</small></h3>
    <p><strong>Chief Complaint:</strong> ${{p.chief_complaint}}</p>
    <p><strong>HPI:</strong> ${{p.history_of_present_illness}}</p>
    <p><strong>PMH:</strong> ${{p.past_medical_history.join(', ')||'None'}}</p>
    <p><strong>Meds:</strong> ${{p.current_medications.join(', ')||'None'}} &nbsp; <strong>Allergies:</strong> ${{p.allergies.join(', ')||'NKDA'}}</p>
    ${{p.additional_context?'<p><em>'+p.additional_context+'</em></p>':''}}
    <div class="vitals">
      <div class="vital">HR ${{v.heart_rate}} bpm</div>
      <div class="vital">BP ${{v.blood_pressure_systolic}}/${{v.blood_pressure_diastolic}}</div>
      <div class="vital">RR ${{v.respiratory_rate}}/min</div>
      <div class="vital">SpO2 ${{v.spo2_percent}}%</div>
      <div class="vital">Temp ${{v.temperature_celsius}}°C</div>
      <div class="vital">GCS ${{v.gcs_score}}/15</div>
      <div class="vital">Pain ${{v.pain_score}}/10</div>
    </div>
    <p><strong>Reward so far:</strong> ${{obs.cumulative_reward.toFixed(2)}} | Remaining: ${{obs.patients_remaining}}</p>
  `;
  document.getElementById('reasoning').value='';
  document.querySelectorAll('.res-check').forEach(c=>c.checked=false);
}}
</script>
</body>
</html>"""


def _resource_checkboxes() -> str:
    resources = [
        "IV_ACCESS", "LABS", "ECG", "IMAGING_XRAY", "IMAGING_CT",
        "IMAGING_MRI", "MEDICATION_IV", "MEDICATION_ORAL",
        "SPECIALIST_CONSULT", "BEDSIDE_MONITORING", "PROCEDURE",
    ]
    html = ""
    for r in resources:
        label = r.replace("_", " ").title()
        html += f'<label><input type="checkbox" class="res-check" value="{r}"> {label}</label>\n'
    return html


# ── Entry point ────────────────────────────────────────────────
app = create_app()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
