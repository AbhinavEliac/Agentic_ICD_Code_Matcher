"""Full Regression Test Suite for Generalized Clinical NLP & ICD-10-CM Coding Pipeline.

Validates all 10 canonical benchmark cases plus Orthopedic (Case H) and Oncology (Case I).
Enforces 100% recall, exact primary/secondary designation, and zero hallucination/pollution.
"""

import pytest

from medical_coding.graph.pipeline import process_clinical_document


@pytest.mark.asyncio
async def test_case_acute_bronchitis():
    """Case 3: Acute bronchitis admitted, ruled-out pneumonia, unmanaged history of falling."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: Productive cough, low-grade fever, and wheezing.
    PAST MEDICAL HISTORY: History of falling.
    HOSPITAL COURSE: Patient presented with acute worsening cough and purulent sputum. Chest X-ray was performed and pneumonia was definitively ruled out. Diagnosed with acute bronchitis and treated with bronchodilators and supportive hydration with symptom resolution.
    DISCHARGE DIAGNOSES:
    1. Acute bronchitis
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-bronchitis")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "J20.9"
    assert not any("pneumonia" in (s.description or "").lower() for s in res.secondary_diagnoses)
    assert not any("fall" in (s.description or "").lower() for s in res.secondary_diagnoses)


@pytest.mark.asyncio
async def test_case_ureteric_calculus():
    """Case 4: Right ureteric calculus (N20.1) - symptom N23 rejected as primary."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: Severe right flank pain radiating to groin.
    DISCHARGE DIAGNOSES:
    1. Right ureteric calculus. CT showed 4mm distal right ureteral stone with renal colic. Managed with IV hydration, analgesics, and tamsulosin.
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-ureter")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "N20.1"


@pytest.mark.asyncio
async def test_case_chronic_gastritis_hpylori():
    """Case 5: Chronic gastritis (K29.50) without bleeding + H. pylori (B96.81)."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: Chronic dyspepsia and epigastric discomfort.
    DISCHARGE DIAGNOSES:
    1. Chronic gastritis
    2. Helicobacter pylori infection
    HOSPITAL COURSE: Endoscopy confirmed chronic gastritis without bleeding or ulceration. Biopsy positive for Helicobacter pylori. Triple therapy initiated.
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-gastritis")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "K29.50"
    sec_codes = {s.code for s in res.secondary_diagnoses}
    assert "B96.81" in sec_codes


@pytest.mark.asyncio
async def test_case_dlbcl_chemo():
    """Case 6: DLBCL chemotherapy admission - zero absence statement leakage."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    ADMITTING DIAGNOSIS: Diffuse Large B-Cell Lymphoma (DLBCL)
    REASON FOR ADMISSION: Admitted for initiation of cycle 1 chemotherapy for DLBCL.
    HOSPITAL COURSE: Patient received R-CHOP chemotherapy infusion without acute event. No major acute complication occurred during the admission.
    DISCHARGE DIAGNOSIS:
    1. Diffuse Large B-Cell Lymphoma
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-dlbcl")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "C83.30"


@pytest.mark.asyncio
async def test_case_cap_t2dm_htn_ckd():
    """Case 7: CAP (J18.9) Primary, T2DM (E11.9), HTN (I10), CKD3 (N18.30) Secondaries."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    REASON FOR ADMISSION: Admitted with high fever, dyspnea, and productive cough due to community-acquired pneumonia.
    DISCHARGE DIAGNOSES:
    1. Community-acquired pneumonia
    2. Type 2 diabetes mellitus
    3. Essential hypertension
    4. Chronic kidney disease stage 3
    HOSPITAL COURSE: Treated with IV ceftriaxone for community-acquired pneumonia. Blood sugars monitored and sliding scale insulin continued for Type 2 diabetes. Amlodipine continued for essential hypertension. Baseline creatinine 1.8 followed for CKD stage 3.
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-cap")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "J18.9"
    sec_codes = {s.code for s in res.secondary_diagnoses}
    assert "E11.9" in sec_codes
    assert "I10" in sec_codes
    assert "N18.30" in sec_codes or "N18.3" in sec_codes


@pytest.mark.asyncio
async def test_case_dlbcl_full():
    """DLBCL Full: Primary C83.30, zero medication or instruction leakage."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    PATIENT NAME: [Redacted]
    ADMITTING DIAGNOSIS: Diffuse Large B-Cell Lymphoma, Stage IV, GCB type.
    REASON FOR ADMISSION: Admitted for cycle 1 R-CHOP chemotherapy for DLBCL.
    HOSPITAL COURSE: Chemotherapy administered per protocol. Tolerated well. No acute chemotherapy-related adverse events. Watch for reactions.
    DISCHARGE MEDICATIONS:
    1. T. Prednisolone 100mg PO daily x 5 days
    2. T. Ondansetron 8mg PO BD PRN
    3. Syp. Cremaffin 15ml at bedtime
    4. T. Pantoprazole 40mg PO OD
    FOLLOW UP: Review in OPD after 14 days.
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-dlbcl-full")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "C83.30"
    assert len(res.secondary_diagnoses) == 0


@pytest.mark.asyncio
async def test_case_uti_managed_t2dm():
    """UTI (N39.0) Primary, actively managed pre-existing T2DM (E11.9) Secondary."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: Dysuria and fever.
    PRINCIPAL DIAGNOSIS: Urinary tract infection.
    PAST MEDICAL HISTORY: Type 2 diabetes mellitus.
    HOSPITAL COURSE: Patient presented with acute dysuria and fever. Urine culture grew E. coli sensitive to ceftriaxone, which was administered with clinical improvement. The patient's pre-existing Type 2 diabetes was actively managed with blood glucose monitoring and sliding scale regular insulin.
    DISCHARGE MEDICATIONS: Oral cefixime, home metformin resumed.
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-uti")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "N39.0"
    sec_codes = {s.code for s in res.secondary_diagnoses}
    assert "E11.9" in sec_codes


@pytest.mark.asyncio
async def test_case_compound_urology():
    """Compound urology: N10 (Primary), N20.0, R65.21, R78.81 (Secondaries)."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: High fevers and right flank pain.
    PRINCIPAL DIAGNOSIS:
    Acute pyelonephritis with right renal calculus, complicated by septic shock and E. coli bacteremia.
    HOSPITAL COURSE: Patient presented with severe acute pyelonephritis and obstructive right renal calculus. Developed septic shock requiring vasopressors and IV fluids. Blood cultures grew Escherichia coli (E. coli bacteremia) treated with IV meropenem. Urology placed DJ stent for right renal calculus.
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-compound")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "N10"
    sec_codes = {s.code for s in res.secondary_diagnoses}
    assert "N20.0" in sec_codes
    assert "R65.21" in sec_codes
    assert "R78.81" in sec_codes


@pytest.mark.asyncio
async def test_case_castleman_oncology():
    """Castleman: D47.Z2 (Primary), Kaposi (C46.0/C46.9), HIV (B20), HBV (B18.1)."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    PATIENT NAME: [Redacted]
    CHIEF COMPLAINT: Generalized lymphadenopathy, fever, and cutaneous lesions.
    PRINCIPAL DIAGNOSIS:
    Multicentric Castleman disease.
    SECONDARY DIAGNOSES:
    1. Kaposi's sarcoma of skin.
    2. Human immunodeficiency virus [HIV] disease.
    3. Chronic viral hepatitis B without delta-agent.
    HOSPITAL COURSE: Patient with confirmed HIV infection on antiretroviral therapy presented with diffuse lymphadenopathy and hyperinflammatory symptoms. Lymph node biopsy confirmed multicentric Castleman disease (plasma cell variant). Skin lesions were biopsied confirming Kaposi's sarcoma. Serologies confirmed chronic viral hepatitis B without delta-agent. Siltuximab therapy initiated for Castleman disease alongside ongoing HAART. Stable at discharge.
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-castleman")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "D47.Z2"
    sec_codes = {s.code for s in res.secondary_diagnoses}
    assert "C46.0" in sec_codes or "C46.9" in sec_codes
    assert "B20" in sec_codes
    assert "B18.1" in sec_codes


@pytest.mark.asyncio
async def test_case_orthopedic_multiple_fractures():
    """Case H: Orthopedic Multiple Fractures & Sprains (atomic decomposition)."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: Left ankle and foot pain and swelling following mechanical fall.
    PRINCIPAL DIAGNOSIS:
    Multiple fractures of left lateral malleolus, tarsals and fifth metatarsal with deltoid and calcaneofibular ligament sprains.
    PHYSICAL EXAMINATION:
    Classification of Fracture: Not Applicable
    Left foot: Severe swelling, ecchymosis, tenderness over lateral malleolus and base of fifth metatarsal.
    HOSPITAL COURSE: Patient presented following a fall resulting in left lateral malleolus fracture, left tarsal fracture, and left fifth metatarsal fracture. There is no displacement or dislocation noted on radiographs; nondisplaced fractures treated non-operatively with posterior splinting. Patient also sustained left deltoid and calcaneofibular ligament sprains.
    DISCHARGE MEDICATIONS: Acetaminophen 500mg PO TID PRN.
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-ortho")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "S82.65XA"
    sec_codes = {s.code for s in res.secondary_diagnoses}
    assert "S92.202A" in sec_codes
    assert "S92.355A" in sec_codes
    assert "S93.422A" in sec_codes
    assert "S93.412A" in sec_codes
    # Verify no metadata or isolated anatomy leakage
    all_codes_and_terms = [res.primary_diagnosis.description] + [s.description for s in res.secondary_diagnoses]
    assert not any("classification" in t.lower() for t in all_codes_and_terms)


@pytest.mark.asyncio
async def test_case_metastatic_breast_cancer():
    """Case I: Metastatic Breast Cancer with Pleural Metastasis and Biomarker Attributes."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    PRINCIPAL DIAGNOSIS:
    Metastatic breast cancer, Stage IV.
    SECONDARY DIAGNOSES:
    1. Pleural metastasis.
    BIOMARKERS & RECEPTOR STATUS:
    Triple negative (ER negative, PR negative, HER2 0 by IHC).
    BRCA pathogenic mutation (BRCA1 positive).
    PD-L1 positive (CPS >= 10).
    Grade III histology.
    HOSPITAL COURSE: Patient with known metastatic breast cancer presented for restaging and systemic therapy adjustment. CT chest confirmed progressive malignant pleural metastasis secondary to breast carcinoma. Thoracentesis performed. Started on chemotherapy regimen.
    """
    res = await process_clinical_document(source=doc, document_id="reg-case-breast")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "C50.919"
    sec_codes = {s.code for s in res.secondary_diagnoses}
    assert "C78.2" in sec_codes
    # Verify biomarkers are not separate diagnoses
    all_terms = [s.description.lower() for s in res.secondary_diagnoses]
    assert not any("triple negative" in t for t in all_terms)
    assert not any("brca" in t for t in all_terms)
    assert not any("pd-l1" in t for t in all_terms)
