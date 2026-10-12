import pytest
from medical_coding.orchestration.pipeline import MedicalCodingPipeline
from medical_coding.schemas.enums import DiagnosisRole
from medical_coding.schemas.response import CodingResult


@pytest.mark.asyncio
async def test_secondary_diagnoses_not_dropped_when_in_pmh():
    """Verify that conditions active in discharge diagnoses are not dropped into historical

    even if they appear in Past Medical History without medication details.
    """
    pipeline = MedicalCodingPipeline()
    text = """
    HOSPITAL DISCHARGE SUMMARY

    DISCHARGE DIAGNOSES:
    1. Acute Appendicitis
    2. Essential Hypertension
    3. Type 2 Diabetes Mellitus
    4. Chronic Kidney Disease

    PAST MEDICAL HISTORY:
    Hypertension, diabetes mellitus, chronic kidney disease.

    HOSPITAL COURSE:
    Patient underwent laparoscopic appendectomy. Blood pressure and glucose levels were monitored.
    Renal function was followed closely.
    """
    result = await pipeline.run_document(document_id="test-doc-sec-pmh", text=text)

    assert result.primary_diagnosis is not None
    assert "appendicitis" in (result.primary_diagnosis.normalized_diagnosis or "").lower()

    # Verify secondary diagnoses are not dropped
    sec_diagnoses = [s.normalized_diagnosis.lower() for s in result.secondary_diagnoses]
    assert any("hypertension" in s for s in sec_diagnoses), f"Hypertension missing from {sec_diagnoses}"
    assert any("diabetes" in s for s in sec_diagnoses), f"Diabetes missing from {sec_diagnoses}"
    assert any("kidney" in s for s in sec_diagnoses), f"CKD missing from {sec_diagnoses}"


@pytest.mark.asyncio
async def test_cpt_and_icdo_verification():
    """Verify that neoplasm cases get ICD-O morphology codes and procedures get verified CPT codes."""
    pipeline = MedicalCodingPipeline()
    text = """
    HOSPITAL DISCHARGE SUMMARY

    PRINCIPAL DIAGNOSIS:
    Infiltrating duct carcinoma of right breast

    SECONDARY DIAGNOSES:
    1. Essential Hypertension

    PROCEDURES:
    Laparoscopic appendectomy
    DJ stenting

    HOSPITAL COURSE:
    Patient admitted with invasive ductal carcinoma of right breast.
    Underwent surgical intervention and monitoring.
    """
    result = await pipeline.run_document(document_id="test-doc-cpt-icdo", text=text)

    # Primary diagnosis ICD-10-CM and ICD-O
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.icd10cm is not None
    assert result.primary_diagnosis.icdo is not None
    assert result.primary_diagnosis.icdo.startswith("M85")

    # Procedures with verified CPT
    assert len(result.procedures) >= 1
    assert any("CPT:" in p for p in result.procedures), f"CPT missing from procedures: {result.procedures}"

    formatted = result.format_output()
    assert "primary diagnosis:" in formatted
    assert "ICD-O:" in formatted
    assert "secondary diagnosises:" in formatted


@pytest.mark.asyncio
async def test_physical_exam_findings_captured():
    """Verify physical exam findings and signs are captured when clinically relevant."""
    pipeline = MedicalCodingPipeline()
    text = """
    HOSPITAL DISCHARGE SUMMARY

    CHIEF COMPLAINT:
    Shortness of breath and fever.

    CLINICAL EXAMINATION:
    Jaundice noted. Auscultation revealed diffuse wheezing and pulmonary rales.
    Bilateral ankle edema present.

    ASSESSMENT AND PLAN:
    Acute bronchitis. Patient also noted to have jaundice and essential hypertension.
    Treated with bronchodilators, hydration, and antihypertensive therapy.
    """
    result = await pipeline.run_document(document_id="test-doc-exam", text=text)

    assert result.primary_diagnosis is not None
    sec_terms = [s.normalized_diagnosis.lower() for s in result.secondary_diagnoses]
    # Check that hypertension and jaundice or wheezing are evaluated
    assert any("hypertension" in s for s in sec_terms) or any("jaundice" in s for s in sec_terms)
