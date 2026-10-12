import asyncio
import pytest
from medical_coding.graph.pipeline import process_clinical_document

@pytest.mark.asyncio
async def test_ent_multidiagnosis_and_procedures():
    ent_text = """Discharge Summary - ENT
Age : 32Y 3M 25D	Bed No.: SS12B - 4230
Consultant : Dr. GANDHI SACHIN
Date of Admission: 02/07/2026	Discharge Type: FOLLOW UP DISCHARGE
Date of Discharge: 02/07/2026	 
Date of Procedure : 02/07/2026
DIAGNOSIS : Deviated nasal septum+ inferior turbinate hypertrophy+ sinusitis
DRUG ALLERGY: Not known
COMORBIDITIES: Not known
HISTORY OF PRESENT ILLNESS: Nasal obstruction
CLINICAL EXAMINATION ON ADMISSION:
NOSE: Deviated nasal septum to left with Bilateral inferior turbinate hypertrophy
COURSE IN THE HOSPITAL: Patient came with complaints of nasal obstruction. Examined and found to have deviated nasal septum+ Inferior turbinate hypertrophy+ sinusitis. Admitted and underwent septoplasty+ turbinoplasty+ fess. Patient stable and being discharged
PROCEDURE DETAILS:
Procedure Performed: Septoplasty+ turbinoplasty+ fess
Side : Bilateral
SURGEON : 76
SURGEON : 16884
Anaesthesiologist: 16316
Type of Anaesthesia: GA
INTRAOPERATIVE FINDING: Deviated nasal septum. Polypoidal tissue in anterior ethmoids
INVESTIGATION: CT PNS - DNS TO LEFT SIDE. ADENOID HYPERTROPHY CAUSING MILD AIRWAY NARROWING. MILD MUCOSAL THICKENING NOTED IN B/L MAXILLARY, ETHMOIDAL & SPHENOID SINUSES. OCCLUSION OF LEFT SPHENO-ETHMOIDAL RECESS. MASTOIDITIS.
STATUS AT DISCHARGE: Stable
DISCHARGE MEDICATIONS:
T. AUGMENTIN 625MG	1-0-1	7 DAYS
T. CIPROFLOXACIN 500MG	1-0-1	14 DAYS
PLAN ON DISCHARGE: Patient is stable hence being discharged"""

    res = await process_clinical_document(source=ent_text, document_id="test-ent-e2e")

    print("\n" + "=" * 50)
    print("ENT FORMATTED OUTPUT:")
    print(res.format_output())
    print("=" * 50)

    # 1. Primary diagnosis must be accurately identified as Deviated nasal septum
    assert res.primary_diagnosis is not None
    pri_term = (res.primary_diagnosis.raw_term or "").lower()
    assert "deviated nasal septum" in pri_term
    assert res.primary_diagnosis.code == "J34.2"

    # 2. Secondary diagnoses must not be empty and must capture inferior turbinate hypertrophy and sinusitis
    assert len(res.secondary_diagnoses) >= 2, f"Expected at least 2 secondary diagnoses, got {len(res.secondary_diagnoses)}"
    sec_terms = [(s.raw_term or s.normalized_diagnosis or "").lower() for s in res.secondary_diagnoses]
    sec_codes = [s.code for s in res.secondary_diagnoses]

    assert any("turbinate" in t for t in sec_terms), "Inferior turbinate hypertrophy must be a secondary diagnosis!"
    assert any("sinusitis" in t for t in sec_terms), "Sinusitis must be a secondary diagnosis!"

    # Turbinate hypertrophy must be correctly coded to J34.3 (not kidney N28.81)
    turbinate_diags = [s for s in res.secondary_diagnoses if "turbinate" in (s.raw_term or "").lower()]
    assert any(s.code == "J34.3" for s in turbinate_diags), f"Expected J34.3 for turbinate hypertrophy, got: {[s.code for s in turbinate_diags]}"

    # Sinusitis must be coded to sinusitis category (J01 / J32)
    sinusitis_diags = [s for s in res.secondary_diagnoses if "sinusitis" in (s.raw_term or "").lower()]
    assert any(s.code.startswith(("J01", "J32")) for s in sinusitis_diags), f"Expected J01 or J32 for sinusitis, got: {[s.code for s in sinusitis_diags]}"

    # 3. Verified CPT procedures
    assert len(res.procedures) >= 2, f"Expected verified procedures, got: {res.procedures}"
    assert any("30520" in p for p in res.procedures), "Septoplasty [CPT: 30520] must be verified!"
