"""
Synthetic patient case bank — 30 cases total.
10 easy (single patient, obvious presentation)
10 medium (multi-patient queue, competing priorities)
10 hard (mass casualty / degrading / incomplete information)

All cases authored with ESI ground truth and resource requirements
per the Emergency Severity Index v4 implementation handbook.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ─────────────────────────────────────────────
# Case schema
# ─────────────────────────────────────────────

def _case(
    patient_id: str,
    age: int,
    sex: str,
    cc: str,
    hpi: str,
    pmh: list,
    meds: list,
    allergies: list,
    hr: int,
    sbp: int,
    dbp: int,
    rr: int,
    spo2: float,
    temp: float,
    gcs: int,
    pain: int,
    esi: int,
    resources: list,
    disposition: str,
    additional_context: str = "",
    sentinel_condition: str = None,
) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "chief_complaint": cc,
        "history_of_present_illness": hpi,
        "past_medical_history": pmh,
        "current_medications": meds,
        "allergies": allergies,
        "vitals": {
            "heart_rate": hr,
            "blood_pressure_systolic": sbp,
            "blood_pressure_diastolic": dbp,
            "respiratory_rate": rr,
            "spo2_percent": spo2,
            "temperature_celsius": temp,
            "gcs_score": gcs,
            "pain_score": pain,
        },
        "additional_context": additional_context,
        "sentinel_condition": sentinel_condition,
        "ground_truth_esi": esi,
        "ground_truth_resources": resources,
        "ground_truth_disposition": disposition,
    }


# ─────────────────────────────────────────────
# EASY CASES (ESI 1-5, single presentations)
# ─────────────────────────────────────────────

EASY_CASES: List[Dict[str, Any]] = [
    _case(
        "E001", 68, "M",
        "Chest pain with diaphoresis",
        "68-year-old male with 30 minutes of crushing substernal chest pain radiating to left arm, "
        "associated with diaphoresis and nausea. Onset at rest.",
        ["Hypertension", "Type 2 Diabetes", "Hyperlipidemia"],
        ["Metformin", "Lisinopril", "Atorvastatin"],
        ["Penicillin"],
        hr=104, sbp=88, dbp=60, rr=22, spo2=94.0, temp=36.8, gcs=15, pain=9,
        esi=1,
        resources=["IV_ACCESS", "LABS", "ECG", "BEDSIDE_MONITORING", "MEDICATION_IV"],
        disposition="resus_bay",
        additional_context="Patient is diaphoretic and pale. ST elevation noted on paramedic ECG.",
        sentinel_condition="STEMI",
    ),
    _case(
        "E002", 34, "F",
        "Sudden severe headache",
        "34-year-old female describes 'the worst headache of my life', sudden onset 45 minutes ago "
        "while exercising. Associated with neck stiffness and photophobia. No prior headache history.",
        ["Oral contraceptive use"],
        ["Norethindrone/Ethinyl estradiol"],
        ["NKDA"],
        hr=96, sbp=158, dbp=94, rr=18, spo2=99.0, temp=37.1, gcs=15, pain=10,
        esi=2,
        resources=["IV_ACCESS", "LABS", "IMAGING_CT", "MEDICATION_IV"],
        disposition="monitored_bed",
        additional_context="Kernig and Brudzinski signs present. Subarachnoid hemorrhage must be ruled out.",
        sentinel_condition="SAH",
    ),
    _case(
        "E003", 52, "F",
        "Shortness of breath",
        "52-year-old female with worsening dyspnea over 2 days, orthopnea (3-pillow), "
        "and bilateral ankle swelling. Known CHF. Missed two days of furosemide.",
        ["Congestive Heart Failure", "Atrial Fibrillation"],
        ["Furosemide", "Carvedilol", "Lisinopril", "Warfarin"],
        ["Sulfa drugs"],
        hr=112, sbp=162, dbp=96, rr=28, spo2=88.0, temp=37.2, gcs=15, pain=3,
        esi=2,
        resources=["IV_ACCESS", "LABS", "ECG", "IMAGING_XRAY", "BEDSIDE_MONITORING", "MEDICATION_IV"],
        disposition="monitored_bed",
        additional_context="Bilateral crackles to mid-lungs. JVD present. Bilateral pitting edema 2+."
    ),
    _case(
        "E004", 28, "M",
        "Right ankle injury after basketball",
        "28-year-old male twisted right ankle during basketball game 1 hour ago. "
        "Able to bear weight with discomfort. Localized swelling over lateral malleolus.",
        [],
        [],
        ["NKDA"],
        hr=78, sbp=124, dbp=76, rr=16, spo2=99.0, temp=36.6, gcs=15, pain=5,
        esi=4,
        resources=["IMAGING_XRAY"],
        disposition="fast_track",
        additional_context="Ottawa ankle rules positive for lateral malleolus tenderness."
    ),
    _case(
        "E005", 4, "F",
        "High fever and irritability",
        "4-year-old girl brought by mother for fever of 39.8°C for 2 days, irritable, "
        "decreased oral intake. No rash, no neck stiffness. Up to date on vaccines.",
        [],
        [],
        ["NKDA"],
        hr=138, sbp=94, dbp=58, rr=28, spo2=98.0, temp=39.8, gcs=15, pain=6,
        esi=2,
        resources=["LABS", "IV_ACCESS", "MEDICATION_IV"],
        disposition="monitored_bed",
        additional_context="Tachycardia and tachypnea for age. Febrile child requires workup to rule out sepsis."
    ),
    _case(
        "E006", 72, "M",
        "Altered mental status",
        "72-year-old male brought by wife for confusion since this morning. Not recognizing family, "
        "mumbling incoherently. Last seen normal at dinner last night. No trauma.",
        ["Hypertension", "Atrial Fibrillation", "Prior TIA"],
        ["Apixaban", "Amlodipine"],
        ["NKDA"],
        hr=84, sbp=196, dbp=108, rr=18, spo2=97.0, temp=36.9, gcs=10, pain=0,
        esi=1,
        resources=["IV_ACCESS", "LABS", "ECG", "IMAGING_CT", "BEDSIDE_MONITORING"],
        disposition="resus_bay",
        additional_context="GCS 10 (E3V3M4). Right facial droop noted. Arm drift on left. NIHSS assessment needed urgently.",
        sentinel_condition="STROKE",
    ),
    _case(
        "E007", 19, "F",
        "Sore throat and fever",
        "19-year-old female with 3 days of worsening sore throat, odynophagia, fever to 38.5°C. "
        "Able to swallow liquids. No drooling, no stridor.",
        [],
        [],
        ["NKDA"],
        hr=96, sbp=118, dbp=72, rr=16, spo2=99.0, temp=38.5, gcs=15, pain=6,
        esi=4,
        resources=["LABS"],
        disposition="fast_track",
        additional_context="Exudative tonsils. Rapid strep or monospot warranted. No airway compromise."
    ),
    _case(
        "E008", 45, "M",
        "Abdominal pain",
        "45-year-old male with 8 hours of periumbilical pain migrating to right lower quadrant. "
        "Nausea and vomiting x3. Low grade fever. No prior surgeries.",
        [],
        [],
        ["NKDA"],
        hr=102, sbp=128, dbp=80, rr=18, spo2=99.0, temp=38.2, gcs=15, pain=8,
        esi=2,
        resources=["IV_ACCESS", "LABS", "IMAGING_CT", "MEDICATION_IV"],
        disposition="monitored_bed",
        additional_context="McBurney's point tenderness. Rebound tenderness present. Suspected appendicitis."
    ),
    _case(
        "E009", 6, "M",
        "Mild rash and runny nose",
        "6-year-old boy with 2-day history of runny nose, mild cough, and fine maculopapular rash "
        "on trunk. Eating and drinking normally. Active and playful.",
        [],
        [],
        ["NKDA"],
        hr=96, sbp=98, dbp=62, rr=20, spo2=99.0, temp=37.8, gcs=15, pain=1,
        esi=5,
        resources=["NONE"],
        disposition="waiting_room",
        additional_context="Likely viral exanthem. Non-toxic appearing child. No resource needed."
    ),
    _case(
        "E010", 58, "F",
        "Hypoglycemia",
        "58-year-old female found by coworker unresponsive at desk. Paramedics report fingerstick "
        "glucose of 32 mg/dL. Responded partially to glucagon 1 mg IM given on scene.",
        ["Type 1 Diabetes Mellitus"],
        ["Insulin glargine", "Insulin lispro"],
        ["NKDA"],
        hr=108, sbp=102, dbp=68, rr=20, spo2=98.0, temp=36.7, gcs=12, pain=0,
        esi=1,
        resources=["IV_ACCESS", "LABS", "BEDSIDE_MONITORING", "MEDICATION_IV"],
        disposition="resus_bay",
        additional_context="GCS 12 on arrival. Glucose 38 on repeat. IV dextrose required. Monitor for recurrence."
    ),
]


# ─────────────────────────────────────────────
# MEDIUM CASES (5-patient queue, competing priorities)
# ─────────────────────────────────────────────

MEDIUM_CASES: List[Dict[str, Any]] = [
    _case(
        "M001", 55, "M",
        "Severe chest tightness and dyspnea",
        "55-year-old male with sudden-onset chest tightness and dyspnea for 20 minutes. "
        "History of COPD and prior CABG. SpO2 dropping despite EMS oxygen.",
        ["COPD", "CAD", "CABG 2019", "HTN"],
        ["Aspirin", "Metoprolol", "Atorvastatin", "Tiotropium inhaler"],
        ["Contrast dye — hives"],
        hr=118, sbp=86, dbp=52, rr=30, spo2=85.0, temp=37.0, gcs=14, pain=8,
        esi=1,
        resources=["IV_ACCESS", "LABS", "ECG", "BEDSIDE_MONITORING", "MEDICATION_IV"],
        disposition="resus_bay",
        additional_context="Accessory muscle use, tripod positioning. High suspicion NSTEMI vs flash pulmonary edema.",
        sentinel_condition="NSTEMI",
    ),
    _case(
        "M002", 38, "F",
        "Severe abdominal pain and vaginal bleeding — 8 weeks pregnant",
        "38-year-old female G3P2, 8 weeks by LMP, with sudden severe right lower quadrant pain "
        "and light vaginal bleeding for 1 hour. Feels faint. Positive home pregnancy test.",
        ["G3P2 — 2 prior C-sections"],
        ["Prenatal vitamins"],
        ["NKDA"],
        hr=126, sbp=92, dbp=56, rr=22, spo2=98.0, temp=36.5, gcs=15, pain=9,
        esi=1,
        resources=["IV_ACCESS", "LABS", "IMAGING_CT", "SPECIALIST_CONSULT", "MEDICATION_IV"],
        disposition="resus_bay",
        additional_context="High suspicion ectopic pregnancy. Hemodynamically unstable. Emergent OB consult.",
        sentinel_condition="ECTOPIC_PREGNANCY",
    ),
    _case(
        "M003", 44, "M",
        "Left leg pain and swelling x3 days",
        "44-year-old male with 3 days of progressive left calf pain and swelling. "
        "Just returned from a 14-hour flight from Singapore. No shortness of breath.",
        ["Obesity", "HTN"],
        ["Lisinopril"],
        ["NKDA"],
        hr=88, sbp=136, dbp=88, rr=16, spo2=98.0, temp=37.0, gcs=15, pain=5,
        esi=3,
        resources=["LABS", "IMAGING_CT"],
        disposition="monitored_bed",
        additional_context="Calf circumference 4 cm greater than right. Wells score 3 — high probability DVT/PE workup.",
        sentinel_condition="PE",
    ),
    _case(
        "M004", 29, "F",
        "Allergic reaction — face and lip swelling",
        "29-year-old female with sudden facial swelling and lip angioedema after eating shellfish "
        "at restaurant 30 minutes ago. No throat tightness, able to swallow, no stridor.",
        ["Shellfish allergy"],
        [],
        ["Shellfish"],
        hr=108, sbp=118, dbp=74, rr=20, spo2=99.0, temp=37.0, gcs=15, pain=3,
        esi=2,
        resources=["IV_ACCESS", "MEDICATION_IV", "BEDSIDE_MONITORING"],
        disposition="monitored_bed",
        additional_context="Angioedema without airway compromise currently. Risk of rapid progression — requires epinephrine and monitoring."
    ),
    _case(
        "M005", 67, "M",
        "Fall from standing — head laceration",
        "67-year-old male on warfarin who fell from standing height in bathroom. "
        "Small frontal laceration 3 cm. Denies LOC. Remembers the fall clearly.",
        ["Atrial Fibrillation", "HTN"],
        ["Warfarin", "Amlodipine"],
        ["NKDA"],
        hr=74, sbp=142, dbp=86, rr=16, spo2=99.0, temp=36.8, gcs=15, pain=3,
        esi=3,
        resources=["LABS", "IMAGING_CT", "PROCEDURE"],
        disposition="monitored_bed",
        additional_context="On anticoagulation — INR check mandatory. CT head needed despite no LOC given anticoagulation. Laceration repair needed."
    ),
    _case(
        "M006", 22, "M",
        "Testicular pain — sudden onset",
        "22-year-old male with sudden onset severe right testicular pain while at rest, "
        "1 hour ago. Associated nausea, no trauma, no fever.",
        [],
        [],
        ["NKDA"],
        hr=106, sbp=128, dbp=78, rr=18, spo2=99.0, temp=37.1, gcs=15, pain=9,
        esi=2,
        resources=["LABS", "IMAGING_CT", "SPECIALIST_CONSULT"],
        disposition="monitored_bed",
        additional_context="Testicular torsion must be excluded — 6-hour window. High-riding transverse lie on exam. Urology consult stat."
    ),
    _case(
        "M007", 80, "F",
        "Hip pain after fall",
        "80-year-old female who fell getting out of bed this morning. Right hip pain, "
        "unable to bear weight. No head strike. Lives alone, daughter with her.",
        ["Osteoporosis", "HTN", "Type 2 DM"],
        ["Alendronate", "Metformin", "Lisinopril"],
        ["NKDA"],
        hr=84, sbp=138, dbp=82, rr=16, spo2=98.0, temp=36.7, gcs=15, pain=7,
        esi=3,
        resources=["IMAGING_XRAY", "LABS", "MEDICATION_IV"],
        disposition="monitored_bed",
        additional_context="Right leg externally rotated and shortened. High suspicion femoral neck fracture. IV analgesia, orthopedics consult."
    ),
    _case(
        "M008", 35, "M",
        "Laceration right hand — power tool",
        "35-year-old male construction worker with 4 cm laceration to right palm from circular saw. "
        "Bleeding controlled with direct pressure. Neurovascular exam intact.",
        [],
        [],
        ["NKDA"],
        hr=90, sbp=130, dbp=82, rr=18, spo2=99.0, temp=36.8, gcs=15, pain=6,
        esi=4,
        resources=["PROCEDURE"],
        disposition="fast_track",
        additional_context="Wound clean, 4 cm laceration, no tendon involvement, intact sensation. Repair and tetanus."
    ),
    _case(
        "M009", 48, "F",
        "Migraine — not responding to home meds",
        "48-year-old female with known migraines presents with typical pattern headache for 10 hours, "
        "unresponsive to sumatriptan and ibuprofen at home. Nausea, photophobia. No new neuro sx.",
        ["Chronic migraines"],
        ["Sumatriptan", "Ibuprofen", "Propranolol"],
        ["Codeine — vomiting"],
        hr=82, sbp=128, dbp=80, rr=16, spo2=99.0, temp=37.0, gcs=15, pain=8,
        esi=3,
        resources=["IV_ACCESS", "MEDICATION_IV", "LABS"],
        disposition="monitored_bed",
        additional_context="Known migraine pattern but needs IV antiemetics and IV abortive therapy. First CT 2 years ago was normal."
    ),
    _case(
        "M010", 14, "M",
        "Asthma exacerbation — moderate",
        "14-year-old male with known asthma, using albuterol MDI 8x today without relief. "
        "Wheezing audible at rest, speaking in short sentences.",
        ["Asthma — diagnosed age 7"],
        ["Albuterol MDI", "Fluticasone inhaler"],
        ["NKDA"],
        hr=124, sbp=112, dbp=68, rr=28, spo2=93.0, temp=37.3, gcs=15, pain=3,
        esi=2,
        resources=["IV_ACCESS", "BEDSIDE_MONITORING", "MEDICATION_IV", "LABS"],
        disposition="monitored_bed",
        additional_context="Moderate exacerbation. Accessory muscle use. SpO2 93%. Requires continuous albuterol neb, ipratropium, steroids."
    ),
]


# ─────────────────────────────────────────────
# HARD CASES (mass casualty / degrading / incomplete info)
# ─────────────────────────────────────────────

HARD_CASES: List[Dict[str, Any]] = [
    _case(
        "H001", 32, "M",
        "MVC — unrestrained driver — polytrauma",
        "32-year-old male unrestrained driver in high-speed MVC. Ambulatory at scene, now confused. "
        "Abrasion across chest from steering wheel. Tachycardic and hypotensive.",
        [],
        [],
        ["NKDA"],
        hr=134, sbp=78, dbp=46, rr=26, spo2=92.0, temp=35.8, gcs=13, pain=7,
        esi=1,
        resources=["IV_ACCESS", "LABS", "ECG", "IMAGING_CT", "BEDSIDE_MONITORING", "MEDICATION_IV"],
        disposition="trauma_bay",
        additional_context="Mechanism: 60 mph collision. Decreased breath sounds right base. Possible pneumothorax. FAST exam needed. Hypothermic."
    ),
    _case(
        "H002", 61, "F",
        "Progressive weakness — unable to walk",
        "61-year-old female with 4-day ascending weakness now unable to stand. "
        "Had GI illness 2 weeks ago. Denies pain. Facial weakness beginning.",
        ["Hypothyroidism"],
        ["Levothyroxine"],
        ["NKDA"],
        hr=78, sbp=138, dbp=84, rr=20, spo2=96.0, temp=37.0, gcs=15, pain=1,
        esi=2,
        resources=["IV_ACCESS", "LABS", "IMAGING_MRI", "SPECIALIST_CONSULT", "BEDSIDE_MONITORING"],
        disposition="monitored_bed",
        additional_context="Ascending flaccid paralysis with areflexia. Guillain-Barré Syndrome high suspicion. Respiratory compromise risk — NIF monitoring critical."
    ),
    _case(
        "H003", 55, "M",
        "Unwitnessed collapse — found by neighbor",
        "55-year-old male found unresponsive by neighbor. Unknown downtime. "
        "No medical history available. No medications found. Unknown allergies.",
        ["Unknown"],
        ["Unknown"],
        ["Unknown"],
        hr=48, sbp=82, dbp=50, rr=8, spo2=82.0, temp=34.2, gcs=5, pain=0,
        esi=1,
        resources=["IV_ACCESS", "LABS", "ECG", "IMAGING_CT", "BEDSIDE_MONITORING", "MEDICATION_IV"],
        disposition="resus_bay",
        additional_context="GCS 5. Profound bradycardia. Hypothermia. Unknown etiology — consider overdose, cardiac, metabolic, neurological. Full resuscitation.",
        sentinel_condition="CARDIAC_ARREST",
    ),
    _case(
        "H004", 27, "F",
        "Psychiatric hold — suicidal ideation with plan",
        "27-year-old female brought by police after friend called 911. Reports taking 'a handful of pills' "
        "approximately 2 hours ago. Unable to specify which pills or how many.",
        ["Depression", "Prior suicide attempt 2022"],
        ["Sertraline", "Quetiapine"],
        ["NKDA"],
        hr=118, sbp=96, dbp=58, rr=22, spo2=97.0, temp=37.0, gcs=14, pain=2,
        esi=1,
        resources=["IV_ACCESS", "LABS", "ECG", "BEDSIDE_MONITORING", "SPECIALIST_CONSULT"],
        disposition="resus_bay",
        additional_context="Unknown overdose. QTc prolongation risk. Toxicology screen, acetaminophen level, ECG mandatory. Psychiatric evaluation after medical clearance.",
        sentinel_condition="OVERDOSE",
    ),
    _case(
        "H005", 78, "M",
        "Septic shock — nursing home transfer",
        "78-year-old male transferred from nursing home. Fever, altered mentation x 2 days, "
        "now obtunded. Urinary catheter in place. Last antibiotics 6 months ago.",
        ["Dementia", "CKD stage 3", "BPH", "HTN"],
        ["Donepezil", "Amlodipine", "Tamsulosin"],
        ["Penicillin — anaphylaxis"],
        hr=122, sbp=72, dbp=44, rr=28, spo2=91.0, temp=39.6, gcs=9, pain=0,
        esi=1,
        resources=["IV_ACCESS", "LABS", "ECG", "IMAGING_XRAY", "BEDSIDE_MONITORING", "MEDICATION_IV"],
        disposition="resus_bay",
        additional_context="qSOFA 3/3. SOFA score high. Source likely urosepsis. PCN allergy — use aztreonam + metronidazole or meropenem. Hour-1 sepsis bundle.",
        sentinel_condition="SEPSIS",
    ),
    _case(
        "H006", 40, "M",
        "Chest pain — atypical, low risk features",
        "40-year-old male with 3 hours of mild chest tightness. Non-radiating. "
        "No diaphoresis. Resolves partially with antacid. Smoker, no cardiac history.",
        ["Active smoker"],
        [],
        ["NKDA"],
        hr=82, sbp=132, dbp=84, rr=16, spo2=99.0, temp=37.0, gcs=15, pain=4,
        esi=3,
        resources=["LABS", "ECG", "BEDSIDE_MONITORING"],
        disposition="monitored_bed",
        additional_context="HEART score 3 — low risk but requires serial troponins and ECG. Atypical presentation in smoker warrants workup."
    ),
    _case(
        "H007", 88, "F",
        "Dyspnea — end stage COPD, DNR",
        "88-year-old female with end-stage COPD. Family brings her for dyspnea and distress. "
        "Has valid DNR/DNI on file. Family requesting comfort measures only.",
        ["COPD GOLD IV", "CHF", "CKD"],
        ["Ipratropium", "Budesonide", "Furosemide", "Morphine PRN"],
        ["NKDA"],
        hr=104, sbp=118, dbp=76, rr=32, spo2=82.0, temp=37.2, gcs=15, pain=6,
        esi=2,
        resources=["MEDICATION_IV", "BEDSIDE_MONITORING", "SPECIALIST_CONSULT"],
        disposition="monitored_bed",
        additional_context="Valid DNR/DNI. Goals of care conversation needed. Comfort-focused: morphine, anxiolytics, palliative O2. No intubation, no ICU. Chaplain/social work."
    ),
    _case(
        "H008", 16, "M",
        "Diabetic ketoacidosis",
        "16-year-old male with Type 1 DM presenting with polyuria, polydipsia, vomiting x2 days, "
        "and fruity breath. Parents report he ran out of insulin 4 days ago.",
        ["Type 1 DM — diagnosed age 8"],
        ["Insulin glargine — ran out"],
        ["NKDA"],
        hr=126, sbp=98, dbp=62, rr=28, spo2=98.0, temp=37.4, gcs=14, pain=5,
        esi=2,
        resources=["IV_ACCESS", "LABS", "BEDSIDE_MONITORING", "MEDICATION_IV"],
        disposition="monitored_bed",
        additional_context="Kussmaul respirations. Glucose 520 by paramedics. pH and bicarb critical. DKA protocol: fluids, insulin drip, potassium replacement. Endocrine consult."
    ),
    _case(
        "H009", 50, "F",
        "Acute vision loss — right eye",
        "50-year-old female with sudden painless vision loss in right eye upon waking, "
        "1.5 hours ago. Describes vision as 'curtain coming down'. No pain, no trauma.",
        ["HTN", "Hypercholesterolemia"],
        ["Amlodipine", "Atorvastatin"],
        ["Aspirin — GI intolerance"],
        hr=76, sbp=168, dbp=96, rr=16, spo2=99.0, temp=36.9, gcs=15, pain=0,
        esi=2,
        resources=["LABS", "SPECIALIST_CONSULT", "IMAGING_MRI"],
        disposition="monitored_bed",
        additional_context="Amaurosis fugax vs CRAO vs retinal detachment. Ophthalmology emergency consult. If CRAO — 4.5 hour thrombolysis window may apply."
    ),
    _case(
        "H010", 3, "M",
        "Febrile seizure — ongoing",
        "3-year-old male brought by parents — had a tonic-clonic seizure at home lasting 3 minutes, "
        "now post-ictal. Temperature 40.1°C. First seizure. No prior neuro history.",
        [],
        [],
        ["NKDA"],
        hr=156, sbp=96, dbp=58, rr=30, spo2=96.0, temp=40.1, gcs=11, pain=0,
        esi=2,
        resources=["IV_ACCESS", "LABS", "MEDICATION_IV", "BEDSIDE_MONITORING"],
        disposition="monitored_bed",
        additional_context="Post-ictal state. Simple febrile seizure most likely. Rule out meningitis, complex seizure. Antipyretics, seizure precautions, monitor for recurrence."
    ),
]


# ─────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "name": "easy",
        "description": (
            "Single-patient triage. One patient arrives at a time. "
            "Presentations are clear and unambiguous. "
            "Objective: assign correct ESI level and appropriate resources."
        ),
        "cases": EASY_CASES,
        "max_steps_per_patient": 1,
        "simultaneous_patients": 1,
    },
    "medium": {
        "name": "medium",
        "description": (
            "Multi-patient queue. 5 patients arrive together. "
            "Agent must triage each in sequence. "
            "Cases include time-sensitive emergencies mixed with lower-acuity visits. "
            "Correct prioritization of the queue order earns bonus reward."
        ),
        "cases": MEDIUM_CASES,
        "max_steps_per_patient": 1,
        "simultaneous_patients": 5,
    },
    "hard": {
        "name": "hard",
        "description": (
            "Mass casualty / complex scenario. All 10 patients arrive simultaneously. "
            "Vitals may degrade each step if under-triaged. "
            "Some patients have incomplete or unavailable medical histories. "
            "Agent must manage resource constraints and ethical complexity. "
            "Clinical reasoning is scored in addition to ESI accuracy."
        ),
        "cases": HARD_CASES,
        "max_steps_per_patient": 1,
        "simultaneous_patients": 10,
    },
}
