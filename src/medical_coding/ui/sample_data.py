"""Realistic clinical demonstration cases and synthetic PDF generation for testing."""

from typing import Any

import pymupdf

CLINICAL_CASES: dict[str, dict[str, Any]] = {
    "Case 1: Decompensated Heart Failure & Comorbidities": {
        "title": "Acute Systolic Heart Failure with Hypertension & T2DM",
        "encounter_id": "ENC-HF-2026-001",
        "description": "72yo male admitted with severe dyspnea, orthopnea, and lower extremity edema.",
        "text": """HOSPITAL DISCHARGE SUMMARY
PATIENT ENCOUNTER: ENC-HF-2026-001
ATTENDING PHYSICIAN: Dr. Robert Vance, MD
DEPARTMENT: Cardiology Services
ADMISSION DATE: 2026-09-24
DISCHARGE DATE: 2026-10-01

CHIEF COMPLAINT:
Severe shortness of breath on exertion and progressive 3-pillow orthopnea.

HISTORY OF PRESENT ILLNESS:
The patient is a 72-year-old male with a known history of chronic cardiovascular disease who presented to the Emergency Department with progressive exertional dyspnea and 15-pound weight gain over the last 2 weeks. Physical examination revealed bilateral pulmonary crackles, jugular venous distention of 8 cm, and 3+ pitting edema of both lower extremities. Echocardiogram demonstrated a left ventricular ejection fraction of 25% with marked ventricular systolic failure.

HOSPITAL COURSE & TREATMENT:
Intensive intravenous furosemide diuresis was initiated with significant clinical improvement. Potassium levels and renal function were closely monitored. The patient achieved a 7 kg fluid reduction and was transitioned to guideline-directed oral medical therapy.

DISCHARGE DIAGNOSES:
1. Primary: Acute systolic heart failure (decompensated congestive heart failure with systolic dysfunction).
2. Essential primary hypertension (longstanding, maintained on amlodipine).
3. Type 2 diabetes mellitus without complications (glycemic control stable).

PERTINENT NEGATIVES:
Patient explicitly denies chest pain, palpitations, or fever. Acute coronary syndrome was ruled out by serial troponin assays.

DISPOSITION:
Discharged to home in stable, euvolemic condition. Follow-up scheduled in Heart Failure Clinic in 7 days.""",
        "expected_primary": "I50.21",
        "expected_secondary": ["I10", "E11.9"],
    },
    "Case 2: Acute COPD Exacerbation with Pneumonia": {
        "title": "Acute Exacerbation of COPD with Community Pneumonia",
        "encounter_id": "ENC-COPD-2026-042",
        "description": "68yo female smoker presenting with purulent sputum and acute respiratory distress.",
        "text": """INPATIENT DISCHARGE SUMMARY
PATIENT ENCOUNTER: ENC-COPD-2026-042
SERVICE: Pulmonary Medicine
ADMISSION DATE: 2026-09-27
DISCHARGE DATE: 2026-10-02

CHIEF COMPLAINT:
Worsening cough, fever of 101.4F, and acute respiratory distress.

HISTORY OF PRESENT ILLNESS:
A 68-year-old female with a 45 pack-year smoking history and known chronic obstructive pulmonary disease presents with 4 days of increased dyspnea, wheezing, and yellow purulent sputum. Chest radiograph demonstrated patchy consolidation in the right lower lobe consistent with community-acquired pneumonia. Arterial blood gas showed acute hypercapnic respiratory decompensation.

CLINICAL MANAGEMENT:
Patient was admitted to Step-Down Unit and received supplemental oxygen, nebulized albuterol/ipratropium, systemic corticosteroids, and intravenous ceftriaxone plus azithromycin. Oxygen saturation improved to 93% on room air.

FINAL DIAGNOSES:
1. Chronic obstructive pulmonary disease with acute exacerbation (acute COPD exacerbation).
2. Pneumonia unspecified organism (community-acquired right lower lobe infiltrate).
3. Gastro-esophageal reflux disease without esophagitis.

NEGATIONS / AUDIT:
Patient denies hemoptysis or syncope. No evidence of pulmonary embolism.

DISCHARGE MEDICATIONS:
Prednisone taper, formoterol/budesonide inhaler, completion of 5-day antibiotic course.""",
        "expected_primary": "J44.1",
        "expected_secondary": ["J18.9", "K21.9"],
    },
    "Case 3: Acute Anterior STEMI Myocardial Infarction": {
        "title": "ST Elevation Myocardial Infarction (Anterior Wall)",
        "encounter_id": "ENC-STEMI-2026-103",
        "description": "59yo male presenting with sudden substernal chest pressure and anterior ST elevations.",
        "text": """EMERGENCY DISCHARGE SUMMARY
PATIENT ENCOUNTER: ENC-STEMI-2026-103
CARDIAC CATHETERIZATION SUITE / CCU
DATE OF SERVICE: 2026-10-02

CLINICAL SYNOPSIS:
59-year-old male transported via EMS with acute crushing substernal chest pain radiating to the left arm and diaphoresis. 12-lead ECG showed 4mm ST elevations across precordial leads V1-V4. Emergency coronary angiography was performed, demonstrating complete 100% thrombotic occlusion of the proximal Left Anterior Descending (LAD) coronary artery. Successful primary percutaneous coronary intervention with drug-eluting stent was deployed with TIMI 3 flow restoration.

FINAL DIAGNOSIS:
1. ST elevation myocardial infarction involving anterior wall (acute transmural anterior STEMI).
2. Essential primary hypertension.
3. Paroxysmal atrial fibrillation (brief peri-procedural runs, converted spontaneously to sinus rhythm).

PAST MEDICAL HISTORY:
Remote appendectomy 25 years ago (resolved). Denies prior stroke or intracranial hemorrhage.

DISPOSITION:
Patient discharged in stable hemodynamic state on dual antiplatelet therapy.""",
        "expected_primary": "I21.09",
        "expected_secondary": ["I10", "I48.0"],
    },
    "Case 4: Severe Sepsis & Acute Kidney Injury": {
        "title": "Systemic Sepsis with Acute Kidney Injury & CKD Stage 3",
        "encounter_id": "ENC-SEPSIS-2026-215",
        "description": "78yo female presenting with septic shock, oliguria, and elevated serum creatinine.",
        "text": """CRITICAL CARE DISCHARGE SUMMARY
PATIENT ENCOUNTER: ENC-SEPSIS-2026-215
INTENSIVE CARE UNIT (ICU)
ADMISSION DATE: 2026-09-20
DISCHARGE DATE: 2026-09-29

HOSPITAL COURSE:
78-year-old female admitted in septic shock secondary to severe urinary tract infection. She met criteria for systemic sepsis unspecified organism with hypotension requiring norepinephrine vasopressor support for 48 hours. Laboratory evaluation revealed acute kidney injury unspecified with serum creatinine rising from baseline 1.4 mg/dL to peak 3.8 mg/dL. Baseline chronic kidney disease stage 3 was noted.

OUTCOME:
With targeted intravenous fluids, antimicrobial therapy, and vasopressor weaning, hemodynamics stabilized and renal function recovered to near-baseline (creatinine 1.6 mg/dL).

FINAL CODING SUMMARY:
1. Sepsis unspecified organism (systemic septicemia, resolved).
2. Acute kidney injury unspecified (acute renal failure, resolving).
3. Chronic kidney disease stage 3 (moderate baseline renal impairment).
4. Essential primary hypertension.

DOCUMENTATION AUDIT:
Patient denies chest pain or shortness of breath. Neurologic status returned to baseline.""",
        "expected_primary": "A41.9",
        "expected_secondary": ["N17.9", "N18.3", "I10"],
    },
    "Case 5: Negation & Historical Rule-Outs (Abstention Demo)": {
        "title": "GERD Evaluation with Explicitly Negated Chest Pain & Historical Diabetes",
        "encounter_id": "ENC-ABSTAIN-2026-330",
        "description": "Case demonstrating deterministic abstentions for negated symptoms and unconfirmed conditions.",
        "text": """OUTPATIENT CLINIC NOTE
ENCOUNTER: ENC-ABSTAIN-2026-330
DEPARTMENT: Internal Medicine

SUBJECTIVE:
54-year-old male reports burning retrosternal sensation after large meals and acidic regurgitation when lying flat.
Patient explicitly denies chest pain, denies shortness of breath, and denies palpitations.
Family history of heart attack in father at age 62 (not patient condition).
Patient had childhood asthma which completely resolved 35 years ago with no current medications.

OBJECTIVE:
Abdominal examination soft, non-tender, no organomegaly. Heart rate regular, lungs clear to auscultation.
Upper endoscopy showed mild mucosal erythema consistent with gastro-esophageal reflux disease without esophagitis.

ASSESSMENT:
1. Gastro-esophageal reflux disease without esophagitis (active).
2. Chest pain: RULED OUT / NEGATED.
3. Childhood asthma: HISTORICAL / RESOLVED (no current treatment).

PLAN:
Initiated omeprazole 20mg daily with dietary lifestyle counseling.""",
        "expected_primary": "K21.9",
        "expected_secondary": [],
    },
}


def generate_sample_pdf(title: str, text: str) -> bytes:
    """Generate a clean, multi-page vector PDF in-memory using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page()

    # Draw header banner box
    page.draw_rect(pymupdf.Rect(40, 30, 572, 70), color=(0.1, 0.2, 0.4), fill=(0.92, 0.95, 0.98))
    page.insert_text(
        (50, 52),
        "ELECTRONIC HEALTH RECORD - CLINICAL DOCUMENTATION",
        fontsize=12,
        color=(0.1, 0.2, 0.5),
    )
    page.insert_text(
        (50, 64),
        title.upper(),
        fontsize=9,
        color=(0.3, 0.4, 0.5),
    )

    # Insert document body
    lines = text.strip().split("\n")
    y_pos = 90
    line_height = 14

    for line in lines:
        if y_pos > 780:
            page = doc.new_page()
            y_pos = 50

        is_heading = any(
            h in line
            for h in [
                "DISCHARGE SUMMARY",
                "CHIEF COMPLAINT:",
                "HISTORY OF PRESENT ILLNESS:",
                "HOSPITAL COURSE",
                "FINAL DIAGNOSIS:",
                "FINAL DIAGNOSES:",
                "DISCHARGE DIAGNOSES:",
                "PERTINENT NEGATIVES:",
                "ASSESSMENT:",
            ]
        )

        if is_heading:
            y_pos += 6
            page.insert_text((50, y_pos), line, fontsize=10, color=(0.05, 0.15, 0.35))
            y_pos += line_height + 2
        else:
            page.insert_text((50, y_pos), line, fontsize=9, color=(0.15, 0.15, 0.15))
            y_pos += line_height

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def generate_sample_image(title: str, text: str) -> bytes:
    """Generate a clean, high-contrast clinical document screenshot as PNG bytes using PIL."""
    import io

    from PIL import Image, ImageDraw

    lines = text.strip().split("\n")
    line_h = 20
    header_h = 75
    total_h = max(550, header_h + len(lines) * line_h + 40)
    width = 820

    img = Image.new("RGB", (width, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw header bar
    draw.rectangle([0, 0, width, header_h], fill=(15, 23, 42))
    draw.text((25, 18), "HOSPITAL CLINICAL DOCUMENTATION - DISCHARGE RECORD", fill=(56, 189, 248))
    draw.text((25, 42), title.upper(), fill=(241, 245, 249))

    # Draw body lines
    y = header_h + 18
    for line in lines:
        is_heading = any(
            h in line
            for h in (
                "DISCHARGE SUMMARY",
                "CHIEF COMPLAINT:",
                "HISTORY OF PRESENT ILLNESS:",
                "HOSPITAL COURSE",
                "FINAL DIAGNOSIS:",
                "FINAL DIAGNOSES:",
                "DISCHARGE DIAGNOSES:",
                "PERTINENT NEGATIVES:",
                "ASSESSMENT:",
            )
        )
        if is_heading:
            draw.text((25, y), line, fill=(14, 116, 144))
            y += line_h + 4
        else:
            draw.text((25, y), line, fill=(30, 41, 59))
            y += line_h

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_sample_txt(title: str, text: str) -> bytes:
    """Generate a clean UTF-8 encoded plain text clinical note as bytes."""
    header = f"=== CLINICAL ENCOUNTER: {title.upper()} ===\n\n"
    full = header + text.strip()
    return full.encode("utf-8")

