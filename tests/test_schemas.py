"""Unit tests for Pydantic domain schemas and invariants."""

import pytest
from pydantic import ValidationError

from medical_coding.schemas.clinical import (
    ClinicalDocument,
    EvidenceSnippet,
    ExtractedDiagnosis,
)
from medical_coding.schemas.enums import (
    Acuity,
    Certainty,
    DiagnosisRole,
    ExecutionStatus,
)
from medical_coding.schemas.response import (
    CodedDiagnosisResponse,
    CodingResult,
)


def test_clinical_document_instantiation() -> None:
    doc = ClinicalDocument(text="Patient diagnosed with acute appendicitis.")
    assert len(doc.document_id) > 0
    assert doc.source_type == "text"
    assert "appendicitis" in doc.text


def test_extracted_diagnosis_requires_evidence() -> None:
    evidence = EvidenceSnippet(quote="Admitted for acute asthma exacerbation.")
    extracted = ExtractedDiagnosis(
        raw_term="acute asthma exacerbation",
        evidence=evidence,
        extraction_confidence=0.92,
    )
    assert extracted.raw_term == "acute asthma exacerbation"
    assert extracted.evidence.quote == "Admitted for acute asthma exacerbation."

    # Must fail without evidence
    with pytest.raises(ValidationError):
        ExtractedDiagnosis(  # type: ignore[call-arg]
            raw_term="acute asthma exacerbation",
            extraction_confidence=0.92,
        )


def test_coding_result_single_primary_invariant() -> None:
    """Verify invariant: secondary_diagnoses cannot contain role PRIMARY."""
    primary = CodedDiagnosisResponse(
        code="I50.21",
        description="Acute systolic heart failure",
        role=DiagnosisRole.PRIMARY,
        acuity=Acuity.ACUTE,
        certainty=Certainty.CONFIRMED,
        evidence_quote="Patient presented with acute systolic heart failure.",
        confidence_score=0.95,
    )

    invalid_secondary = CodedDiagnosisResponse(
        code="E11.9",
        description="Type 2 diabetes",
        role=DiagnosisRole.PRIMARY,  # INVARIANT VIOLATION: Role is PRIMARY in secondary list
        acuity=Acuity.CHRONIC,
        certainty=Certainty.CONFIRMED,
        evidence_quote="History of diabetes.",
        confidence_score=0.90,
    )

    valid_secondary = CodedDiagnosisResponse(
        code="E11.9",
        description="Type 2 diabetes",
        role=DiagnosisRole.SECONDARY,
        acuity=Acuity.CHRONIC,
        certainty=Certainty.CONFIRMED,
        evidence_quote="History of diabetes.",
        confidence_score=0.90,
    )

    # Valid scenario
    result = CodingResult(
        document_id="doc-123",
        status=ExecutionStatus.SUCCESS,
        primary_diagnosis=primary,
        secondary_diagnoses=[valid_secondary],
        abstentions=[],
        processing_time_ms=120.5,
    )
    assert result.primary_diagnosis.code == "I50.21"
    assert len(result.secondary_diagnoses) == 1

    # Invalid scenario: secondary list contains PRIMARY role
    with pytest.raises(ValueError, match="Secondary diagnoses cannot have role PRIMARY"):
        CodingResult(
            document_id="doc-123",
            status=ExecutionStatus.SUCCESS,
            primary_diagnosis=primary,
            secondary_diagnoses=[invalid_secondary],
            abstentions=[],
            processing_time_ms=120.5,
        )
