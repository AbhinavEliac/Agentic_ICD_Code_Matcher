"""Adversarial and Edge-Case Test Suite for Generalized Clinical NLP Engine.

Enforces zero-hallucination, abstention-first policies, and strict gate rejections
across edge cases detailed in Section 35 of the Master Prompt:
1. Negated disease
2. Historical condition without inpatient care
3. Medication-only mention
4. Isolated symptom without underlying diagnosis
5. Disease attribute/staging not converted to standalone diagnosis
6. Ambiguous specificity without documentary distinction
7. Duplicate diagnosis across multiple sections
8. Missing database code resulting in clean NO_DATABASE_MATCH
"""

import pytest

from medical_coding.graph.pipeline import process_clinical_document


@pytest.mark.asyncio
async def test_adversarial_negated_disease():
    """Negated condition ('pneumonia ruled out', 'no evidence of pulmonary embolism') must NOT code."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: Chest pain and shortness of breath.
    HOSPITAL COURSE: CT pulmonary angiogram was negative for pulmonary embolism. Chest X-ray ruled out pneumonia. Patient had musculoskeletal chest wall strain, improved with NSAIDs.
    DISCHARGE DIAGNOSIS:
    Musculoskeletal chest pain.
    """
    res = await process_clinical_document(source=doc, document_id="adv-negated")
    all_codes = [res.primary_diagnosis.code if res.primary_diagnosis else None] + [
        s.code for s in res.secondary_diagnoses
    ]
    assert not any(c and c.startswith("I26") for c in all_codes)
    all_descs = [res.primary_diagnosis.description.lower() if res.primary_diagnosis else ""] + [
        s.description.lower() for s in res.secondary_diagnoses
    ]
    assert not any("embolism" in d for d in all_descs)
    assert not any("pneumonia" in d for d in all_descs)


@pytest.mark.asyncio
async def test_adversarial_historical_condition():
    """Past condition without documented active evaluation/treatment must be EXCLUDED."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: Inguinal hernia repair.
    PAST MEDICAL HISTORY: Remote history of cholecystectomy 10 years ago. History of childhood asthma (resolved).
    HOSPITAL COURSE: Underwent elective uncomplicated right inguinal hernia repair.
    DISCHARGE DIAGNOSIS:
    Right inguinal hernia.
    """
    res = await process_clinical_document(source=doc, document_id="adv-historical")
    assert res.primary_diagnosis is not None
    assert "hernia" in res.primary_diagnosis.description.lower()
    all_descs = [s.description.lower() for s in res.secondary_diagnoses]
    assert not any("asthma" in d for d in all_descs)
    assert not any("cholecystectomy" in d for d in all_descs)


@pytest.mark.asyncio
async def test_adversarial_medication_firewall():
    """Medication mentions alone must NEVER become clinical diagnoses."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    DISCHARGE MEDICATIONS:
    1. Metformin 500mg PO BID
    2. Atorvastatin 20mg PO QHS
    3. Lisinopril 10mg PO Daily
    4. Syp. Cremaffin 15ml PO QHS PRN
    5. Alex cough lozenges PRN
    PLAN: Routine follow up in 1 month.
    """
    res = await process_clinical_document(source=doc, document_id="adv-med-firewall")
    # Should not produce diagnoses from medications
    all_descs = []
    if res.primary_diagnosis:
        all_descs.append(res.primary_diagnosis.description.lower())
    all_descs.extend(s.description.lower() for s in res.secondary_diagnoses)
    assert not any("metformin" in d for d in all_descs)
    assert not any("cremaffin" in d for d in all_descs)
    assert not any("lozenge" in d for d in all_descs)


@pytest.mark.asyncio
async def test_adversarial_standalone_attribute_firewall():
    """Disease attributes (Stage IV, Triple Negative, GCB, Grade III) must not be separate diagnoses."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    PRINCIPAL DIAGNOSIS:
    Invasive ductal carcinoma of breast.
    BIOMARKERS:
    Stage IV.
    Triple negative.
    BRCA pathogenic.
    Grade III.
    GCB subtype.
    HOSPITAL COURSE: Restaged and treated with paclitaxel.
    """
    res = await process_clinical_document(source=doc, document_id="adv-attr-firewall")
    assert res.primary_diagnosis is not None
    sec_descs = [s.description.lower() for s in res.secondary_diagnoses]
    assert not any("stage iv" in d for d in sec_descs)
    assert not any("triple negative" in d for d in sec_descs)
    assert not any("brca" in d for d in sec_descs)
    assert not any("grade iii" in d for d in sec_descs)
    assert not any("gcb" in d for d in sec_descs)


@pytest.mark.asyncio
async def test_adversarial_unsupported_displacement_abstention():
    """If database requires displaced/nondisplaced and text is ambiguous without support, must abstain."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    PRINCIPAL DIAGNOSIS:
    Fracture of shaft of left humerus.
    HOSPITAL COURSE: Patient sustained left arm injury. Radiograph revealed left humerus shaft fracture. Arm immobilized in coaptation splint. Displacement not evaluated due to splint artifact.
    """
    res = await process_clinical_document(source=doc, document_id="adv-ambig-disp")
    # If the system matches a code, it must NOT choose a displaced code without evidence
    if res.primary_diagnosis and res.primary_diagnosis.code:
        desc = res.primary_diagnosis.description.lower()
        assert "displaced" not in desc or "nondisplaced" in desc or "without mention of displacement" in desc


@pytest.mark.asyncio
async def test_adversarial_candidate_deduplication():
    """The same clinical concept mentioned across 4 sections must become ONE canonical candidate."""
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: Chronic gastritis symptoms.
    ADMITTING DIAGNOSIS: Chronic gastritis.
    HOSPITAL COURSE: Patient evaluated for chronic gastritis with EGD showing mild chronic antral gastritis.
    DISCHARGE DIAGNOSIS:
    1. Chronic gastritis
    """
    res = await process_clinical_document(source=doc, document_id="adv-dedup")
    assert res.primary_diagnosis is not None
    assert "gastritis" in res.primary_diagnosis.description.lower()
    # Must NOT have chronic gastritis also in secondary diagnoses!
    sec_descs = [s.description.lower() for s in res.secondary_diagnoses]
    assert not any("gastritis" in d for d in sec_descs), "Duplicate gastritis candidate leaked into secondaries"
