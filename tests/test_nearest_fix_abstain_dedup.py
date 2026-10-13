"""Unit and integration tests for branch nearest_fix:
1. Nearest match codes of diagnosis are placed in the abstain section (not billable primary/secondary).
2. Duplicate evidence points are dropped from diagnoses and output.
"""

import pytest
from medical_coding.graph.pipeline import process_clinical_document
from medical_coding.schemas.enums import AbstentionReason, Acuity, Certainty, DiagnosisRole
from medical_coding.schemas.response import CodedDiagnosisResponse, CodingResult
from medical_coding.schemas.validation import AbstentionRecord


def test_coded_diagnosis_response_drops_duplicate_evidence_points() -> None:
    """Test that CodedDiagnosisResponse automatically deduplicates evidence quotes and list."""
    resp = CodedDiagnosisResponse(
        raw_term="Essential hypertension",
        normalized_diagnosis="Essential hypertension",
        description="Essential (primary) hypertension",
        code="I10",
        role=DiagnosisRole.PRIMARY,
        acuity=Acuity.CHRONIC,
        certainty=Certainty.CONFIRMED,
        evidence_quote="Blood pressure elevated 150/90 mmHg. | Blood pressure elevated 150/90 mmHg. | Patient taking amlodipine daily.",
        evidence=[
            {"quote": "Blood pressure elevated 150/90 mmHg.", "section": "VITALS"},
            {"quote": "Blood pressure elevated 150/90 mmHg.", "section": "VITALS"},
            {"quote": "Patient taking amlodipine daily.", "section": "MEDICATIONS"},
            {"quote": "patient taking amlodipine daily", "section": "MEDICATIONS"},
        ],
        confidence_score=0.95,
    )

    # Verify evidence_quote deduplication
    assert resp.evidence_quote == "Blood pressure elevated 150/90 mmHg. | Patient taking amlodipine daily."

    # Verify evidence list deduplication
    assert len(resp.evidence) == 2
    assert resp.evidence[0]["quote"] == "Blood pressure elevated 150/90 mmHg."
    assert resp.evidence[1]["quote"] == "Patient taking amlodipine daily."


def test_coding_result_format_output_drops_duplicate_evidence() -> None:
    """Test that format_output drops duplicate evidence snippets."""
    resp = CodedDiagnosisResponse(
        raw_term="Type 2 diabetes mellitus",
        normalized_diagnosis="Type 2 diabetes mellitus",
        description="Type 2 diabetes mellitus without complications",
        code="E11.9",
        role=DiagnosisRole.PRIMARY,
        acuity=Acuity.CHRONIC,
        certainty=Certainty.CONFIRMED,
        evidence_quote="HbA1c 8.2% on metformin. | HbA1c 8.2% on metformin. | Managed on insulin.",
        evidence=[
            {"quote": "HbA1c 8.2% on metformin."},
            {"quote": "HbA1c 8.2% on metformin."},
            {"quote": "Managed on insulin."},
        ],
        confidence_score=0.90,
    )
    result = CodingResult(
        document_id="doc-test-dedup",
        status="SUCCESS",
        primary_diagnosis=resp,
        processing_time_ms=10.0,
    )
    output = result.format_output()
    assert "HbA1c 8.2% on metformin. | Managed on insulin." in output
    # Count occurrences of the phrase in output
    assert output.count("HbA1c 8.2% on metformin.") == 1


@pytest.mark.asyncio
async def test_nearest_match_placed_in_abstain_section() -> None:
    """Test that conditions lacking verified local database code have their nearest match in the abstain section."""
    note = """
    DISCHARGE SUMMARY
    ADMITTING DIAGNOSIS: Acute cholecystitis
    SECONDARY DIAGNOSES: Choledocholithiasis, Essential hypertension
    HOSPITAL COURSE:
    Patient admitted with acute cholecystitis, underwent laparoscopic cholecystectomy.
    MRCP demonstrated choledocholithiasis without cholangitis.
    Hypertension managed with home medications.
    """
    final_result = await process_clinical_document(
        source=note,
        document_id="doc-nearest-test",
    )
    assert final_result is not None
    assert final_result.primary_diagnosis is not None
    assert "cholecystitis" in final_result.primary_diagnosis.description.lower()

    # Secondaries should only have verified matched codes (not choledocholithiasis)
    sec_terms = [s.normalized_diagnosis.lower() for s in final_result.secondary_diagnoses]
    assert "choledocholithiasis" not in sec_terms
    assert any("hypertension" in t for t in sec_terms)

    # Choledocholithiasis must be in abstentions with its nearest match code recorded
    choledo_abst = [
        a for a in final_result.abstentions
        if a.raw_term and "choledocholithiasis" in a.raw_term.lower()
    ]
    assert len(choledo_abst) >= 1
    abst_rec = choledo_abst[0]
    assert abst_rec.reason == AbstentionReason.NO_MATCHING_ICD_CANDIDATE
    # Nearest match code must be present in record detail or nearest_code
    assert abst_rec.nearest_code is not None or "Nearest database match:" in abst_rec.detail
    if abst_rec.nearest_code:
        assert len(abst_rec.nearest_code) >= 3
