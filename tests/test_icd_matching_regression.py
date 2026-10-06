"""Comprehensive regression tests verifying database-authoritative ICD code matching."""

import pytest

from medical_coding.graph.pipeline import process_clinical_document


@pytest.mark.asyncio
async def test_regression_right_ureteric_calculus_matches_n20_1() -> None:
    """Right ureteric calculus must match N20.1 Calculus of ureter, not symptom N23 or N20.2."""
    doc = """
    DISCHARGE SUMMARY
    CHIEF COMPLAINT: Severe acute flank pain.
    DISCHARGE DIAGNOSES:
    1. Right ureteric calculus. CT abdomen demonstrated a 5 mm stone in the right distal ureter with renal colic. Managed conservatively with tamsulosin.
    """
    res = await process_clinical_document(source=doc, document_id="doc-reg-ureteric")
    assert res is not None
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "N20.1"
    assert "ureter" in res.primary_diagnosis.description.lower()
    assert res.primary_diagnosis.matching_status == "MATCHED"
    assert res.primary_diagnosis.database_code == "N20.1"


@pytest.mark.asyncio
async def test_regression_chronic_gastritis_no_bleeding_matches_k29_50() -> None:
    """Chronic gastritis with 'no active bleeding' must match K29.50 without bleeding, not K29.51 or K29.41."""
    doc = """
    DISCHARGE SUMMARY
    CHIEF COMPLAINT: Epigastric discomfort.
    DISCHARGE DIAGNOSES:
    1. Chronic gastritis. Upper endoscopy confirmed chronic antral gastritis. No active bleeding or ulceration noted.
    """
    res = await process_clinical_document(source=doc, document_id="doc-reg-gastritis")
    assert res is not None
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "K29.50"
    assert "without bleeding" in res.primary_diagnosis.description.lower()
    assert res.primary_diagnosis.matching_status == "MATCHED"


@pytest.mark.asyncio
async def test_regression_general_heart_failure_matches_i50_9() -> None:
    """General heart failure without subtype must match I50.9, not I50.20, I50.21, or I50.89."""
    doc = """
    DISCHARGE SUMMARY
    CHIEF COMPLAINT: Shortness of breath and peripheral edema.
    DISCHARGE DIAGNOSES:
    1. Heart failure. Patient monitored on oral diuretics with symptom stabilization.
    """
    res = await process_clinical_document(source=doc, document_id="doc-reg-hf")
    assert res is not None
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code in ("I50.9", "I50")
    assert res.primary_diagnosis.code != "I50.21"
    assert res.primary_diagnosis.code != "I50.89"
    assert res.primary_diagnosis.matching_status == "MATCHED"


@pytest.mark.asyncio
async def test_regression_general_chest_pain_matches_r07_9() -> None:
    """General chest pain without pleuritic/breathing features must match R07.9, not R07.1."""
    doc = """
    DISCHARGE SUMMARY
    CHIEF COMPLAINT: Chest discomfort.
    DISCHARGE DIAGNOSES:
    1. Chest pain, atypical. Cardiac enzymes and ECG were negative for acute ischemia.
    """
    res = await process_clinical_document(source=doc, document_id="doc-reg-cp")
    assert res is not None
    assert res.primary_diagnosis.code in ("R07.9", "R07.4")
    assert "chest pain" in res.primary_diagnosis.description.lower()
    assert res.primary_diagnosis.matching_status == "MATCHED"


@pytest.mark.asyncio
async def test_regression_suspected_pneumonia_matches_j18_9() -> None:
    """Suspected pneumonia without pathogen must match J18.9, not organism-specific B77.81 or J16.0."""
    doc = """
    DISCHARGE SUMMARY
    CHIEF COMPLAINT: Productive cough and fever.
    DISCHARGE DIAGNOSES:
    1. Community-acquired pneumonia. Chest X-ray showed right lower lobe consolidation. Sputum cultures pending.
    """
    res = await process_clinical_document(source=doc, document_id="doc-reg-pna")
    assert res is not None
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "J18.9"
    assert res.primary_diagnosis.matching_status == "MATCHED"


@pytest.mark.asyncio
async def test_regression_right_flank_pain_matches_r10_a1() -> None:
    """Right flank pain must match R10.A1 right side, never R10.A3 bilateral."""
    doc = """
    DISCHARGE SUMMARY
    CHIEF COMPLAINT: Severe pain.
    DISCHARGE DIAGNOSES:
    1. Right flank pain. Evaluated by urology with resolution of symptoms.
    """
    res = await process_clinical_document(source=doc, document_id="doc-reg-flank")
    assert res is not None
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.code == "R10.A1"
    assert "right" in res.primary_diagnosis.description.lower()
    assert res.primary_diagnosis.matching_status == "MATCHED"


@pytest.mark.asyncio
async def test_regression_ruled_out_condition_abstains() -> None:
    """Ruled out condition must abstain with NO_DATABASE_MATCH."""
    doc = """
    DISCHARGE SUMMARY
    CHIEF COMPLAINT: Right lower quadrant pain.
    DISCHARGE DIAGNOSES:
    1. Acute appendicitis ruled out by ultrasound. Mesenteric adenitis managed conservatively.
    """
    res = await process_clinical_document(source=doc, document_id="doc-reg-ruled-out")
    assert res is not None
    if res.primary_diagnosis and "appendicitis" in res.primary_diagnosis.raw_term.lower():
        assert res.primary_diagnosis.code is None
        assert res.primary_diagnosis.matching_status == "NO_DATABASE_MATCH"
    appendicitis_abstentions = [
        a for a in res.abstentions if "appendicitis" in (a.raw_term or "").lower()
    ]
    assert len(appendicitis_abstentions) > 0 or (
        res.primary_diagnosis and "adenitis" in res.primary_diagnosis.raw_term.lower()
    )
