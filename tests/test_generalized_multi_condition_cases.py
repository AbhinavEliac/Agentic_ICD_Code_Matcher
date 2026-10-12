import asyncio
import pytest
from medical_coding.graph.pipeline import process_clinical_document

@pytest.mark.asyncio
async def test_generalized_adjacent_conditions_separation():
    """Test that multiple adjacent conditions without commas or 'and' are cleanly split into individual diagnoses."""
    sample_text = """DIAGNOSIS: Cholelithiasis Pancreatitis
HISTORY OF PRESENT ILLNESS: Patient presented with severe right upper quadrant abdominal pain and elevated lipase.
COURSE IN HOSPITAL: Ultrasound confirmed gallstones. Gastroenterology evaluated and confirmed acute pancreatitis."""
    
    res = await process_clinical_document(source=sample_text, document_id="test-adj-1")
    assert res.primary_diagnosis is not None
    pri_name = (res.primary_diagnosis.normalized_diagnosis or res.primary_diagnosis.raw_term or "").lower()
    assert "cholelithiasis pancreatitis" not in pri_name
    
    all_names = [pri_name] + [
        (s.normalized_diagnosis or s.raw_term or s.description or "").lower()
        for s in res.secondary_diagnoses
    ]
    assert any("cholelithiasis" in n for n in all_names)
    assert any("pancreatitis" in n for n in all_names)

@pytest.mark.asyncio
async def test_generalized_consult_findings_extraction():
    """Test that specialist consult findings in hospital course are extracted as secondary diagnoses."""
    sample_text = """DIAGNOSIS: Acute myocardial infarction
HISTORY OF PRESENT ILLNESS: Patient admitted under Cardiology for acute chest pain and troponin elevation.
COURSE IN HOSPITAL: Patient underwent coronary angiography. Endocrinology consultation concluded that patient had uncontrolled diabetes mellitus type 2, and advised insulin glargine."""
    
    res = await process_clinical_document(source=sample_text, document_id="test-consult-1")
    assert res.primary_diagnosis is not None
    
    all_names = [
        (res.primary_diagnosis.normalized_diagnosis or res.primary_diagnosis.raw_term or "").lower()
    ] + [
        (s.normalized_diagnosis or s.raw_term or s.description or "").lower()
        for s in res.secondary_diagnoses
    ]
    assert any("infarction" in n for n in all_names)
    assert any("diabetes" in n for n in all_names)

@pytest.mark.asyncio
async def test_generalized_evidence_purity():
    """Verify that clinical criteria and internal audit strings never leak into evidence quotes."""
    sample_text = """FINAL DIAGNOSIS: Essential hypertension, Type 2 diabetes mellitus
HISTORY: Patient with long standing history of elevated blood pressure and hyperglycemia."""
    
    res = await process_clinical_document(source=sample_text, document_id="test-purity-1")
    for d in [res.primary_diagnosis] + res.secondary_diagnoses:
        if d and d.evidence_quote:
            eq = d.evidence_quote.lower()
            assert "designated as primary diagnosis" not in eq
            assert "clinical criteria:" not in eq
            assert "explicitly documented" not in eq
