import tempfile
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from medical_coding.graph.pipeline import (
    process_clinical_document,
)
from medical_coding.graph.workflow import (
    export_graph_ascii,
    export_graph_mermaid,
)
from medical_coding.orchestration.pipeline import MedicalCodingPipeline
from medical_coding.schemas.enums import (
    AbstentionReason,
    DiagnosisRole,
    ExecutionStatus,
)
from medical_coding.schemas.response import CodingResult


@pytest.fixture
def sample_discharge_summary() -> str:
    """Realistic discharge summary with primary, secondary, and historical conditions."""
    return """
    DISCHARGE SUMMARY
    Patient Name: John Doe
    MRN: 987654321
    Admission Date: 2026-09-15
    Discharge Date: 2026-09-20

    CHIEF COMPLAINT:
    Acute shortness of breath and fluid overload.

    DISCHARGE DIAGNOSES:
    1. Acute systolic heart failure - Patient admitted with acute exacerbation of systolic congestive heart failure. Transthoracic echocardiogram revealed left ventricular ejection fraction of 20% with severe diffuse hypokinesis. Treated with intravenous furosemide diuresis with 4 kg fluid loss and symptomatic resolution. Transitioned to oral lisinopril and carvedilol.
    2. Type 2 diabetes mellitus - Monitored daily fasting blood glucose; blood sugars remained controlled on home metformin 1000 mg twice daily.
    3. Essential primary hypertension - Blood pressure monitored on telemetry throughout admission; stable.

    PAST MEDICAL HISTORY:
    1. Remote appendectomy in 2005.
    2. Resolved seasonal allergic rhinitis.

    HOSPITAL COURSE:
    Patient presented to the emergency department in acute decompensated heart failure with bilateral lower extremity edema and pulmonary congestion. IV loop diuretics were administered with good response. Renal function was tracked and remained baseline. Patient was discharged in stable condition.
    """


@pytest.fixture
def sample_unsupported_specificity_summary() -> str:
    """Discharge summary documenting only general condition without subtype specificity."""
    return """
    DISCHARGE SUMMARY
    CHIEF COMPLAINT: Fatigue and peripheral edema.
    DISCHARGE DIAGNOSES:
    1. Heart failure. Patient was monitored on oral diuretic.
    """


@pytest.fixture
def sample_ruled_out_summary() -> str:
    """Discharge summary containing explicitly ruled-out conditions."""
    return """
    DISCHARGE SUMMARY
    CHIEF COMPLAINT: Chest pressure.
    ASSESSMENT:
    1. Acute myocardial infarction definitively ruled out by serial negative troponins and normal ECG.
    2. Gastroesophageal reflux disease - managed with oral omeprazole with relief.
    """


# ==============================================================================
# 1. END-TO-END PIPELINE EXECUTION TEST
# ==============================================================================


@pytest.mark.asyncio
async def test_end_to_end_pipeline_success(sample_discharge_summary: str) -> None:
    """Verify that a full clinical document executes across all 10 nodes to produce a verified CodingResult."""
    result: CodingResult = await process_clinical_document(
        source=sample_discharge_summary,
        document_id="doc-e2e-001",
    )

    assert result is not None
    assert result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS)

    # 1. Guardrail 9: Maximum ONE Primary Diagnosis
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.role == DiagnosisRole.PRIMARY
    assert result.primary_diagnosis.code == "I50.21"
    assert "Acute systolic heart failure" in result.primary_diagnosis.description
    assert result.primary_diagnosis.is_terminal_billable is True
    assert result.primary_diagnosis.evidence_quote != ""

    # 2. Guardrails 3, 5: Secondary Diagnoses Active & Managed
    sec_codes = [s.code for s in result.secondary_diagnoses]
    for s in result.secondary_diagnoses:
        assert s.role == DiagnosisRole.SECONDARY
        assert s.is_terminal_billable is True
        assert s.evidence_quote != ""

    # Either Diabetes E11.9 or Hypertension I10 are captured as valid active secondaries
    assert "E11.9" in sec_codes or "I10" in sec_codes

    # 3. Guardrail 5: Historical appendectomy NOT coded as secondary
    assert not any("append" in s.description.lower() for s in result.secondary_diagnoses)

    # 4. Internal Audit Trail Requirement
    audit_trail = result.metadata.get("audit_trail", [])
    assert len(audit_trail) >= 1
    sample_audit = audit_trail[0]
    assert "document_id" in sample_audit
    assert "evidence" in sample_audit
    assert "diagnosis_id" in sample_audit
    assert "context" in sample_audit
    assert "classification" in sample_audit
    assert "retrieved_candidates" in sample_audit
    assert "selected_candidate" in sample_audit
    assert "validation" in sample_audit
    assert "final_result" in sample_audit


# ==============================================================================
# 2. PARALLELISM: ASYNCHRONOUS BATCH OF >= 10 DOCUMENTS
# ==============================================================================


@pytest.mark.asyncio
async def test_asynchronous_batch_processing_at_least_10_documents(
    sample_discharge_summary: str,
) -> None:
    """Verify that at least 10 documents are processed concurrently with bounded concurrency."""
    batch_size = 10
    documents = [
        {
            "document_id": f"batch-doc-{i:02d}",
            "text": f"{sample_discharge_summary}\nENCOUNTER ID: {i}",
            "metadata": {"doc_index": i},
        }
        for i in range(batch_size)
    ]

    pipeline = MedicalCodingPipeline()
    batch_status = await pipeline.run_batch_documents(documents, max_concurrency=10)

    assert len(batch_status) == batch_size
    for i, res in enumerate(batch_status):
        assert res.document_id == f"batch-doc-{i:02d}"
        assert res.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS)
        assert res.primary_diagnosis is not None
        assert res.primary_diagnosis.code == "I50.21"


# ==============================================================================
# 3. GUARDRAILS: RULED OUT & HISTORICAL CONDITIONS
# ==============================================================================


@pytest.mark.asyncio
async def test_guardrail_ruled_out_cannot_be_coded_as_confirmed(
    sample_ruled_out_summary: str,
) -> None:
    """Guardrail 4: Ruled-out diagnoses must NEVER be coded as confirmed."""
    result = await process_clinical_document(
        source=sample_ruled_out_summary,
        document_id="doc-ruled-out",
    )

    assert result is not None
    # Myocardial infarction was explicitly ruled out; must NOT appear as primary or secondary
    coded_codes = [s.code for s in result.secondary_diagnoses]
    if result.primary_diagnosis:
        coded_codes.append(result.primary_diagnosis.code)

    # I21 is myocardial infarction category
    assert not any(c.startswith("I21") for c in coded_codes)


# ==============================================================================
# 4. GUARDRAILS: UNSUPPORTED SPECIFICITY
# ==============================================================================


@pytest.mark.asyncio
async def test_guardrail_unsupported_specificity_realigned_or_abstained(
    sample_unsupported_specificity_summary: str,
) -> None:
    """Guardrail 11: Documenting general condition cannot be forced into specific subtype."""
    result = await process_clinical_document(
        source=sample_unsupported_specificity_summary,
        document_id="doc-unsupported-spec",
    )

    assert result is not None
    if result.primary_diagnosis:
        # If coded, must be general/unspecified code I50.9, NOT acute systolic I50.21!
        assert result.primary_diagnosis.code != "I50.21"
        assert result.primary_diagnosis.code in ("I50.9", "I50")


# ==============================================================================
# 5. GUARDRAILS: MISSING / EMPTY EVIDENCE
# ==============================================================================


@pytest.mark.asyncio
async def test_guardrail_empty_input_clean_abstention() -> None:
    """Guardrail 12: Empty or whitespace input cleanly abstains without unhandled exception."""
    result = await process_clinical_document(
        source="",
        document_id="doc-empty",
    )

    assert result.status == ExecutionStatus.ABSTAINED
    assert result.primary_diagnosis is None
    assert len(result.secondary_diagnoses) == 0
    assert len(result.abstentions) >= 1
    assert result.abstentions[0].reason == AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE


# ==============================================================================
# 6. GRAPH VISUALIZATION & TOPOLOGY EXPORTS
# ==============================================================================


def test_graph_visualization_exports() -> None:
    """Verify that graph topology exports to Mermaid and ASCII format without errors."""
    mermaid_def = export_graph_mermaid()
    assert "__start__" in mermaid_def or "START" in mermaid_def
    assert "validate_document" in mermaid_def
    assert "extract_text" in mermaid_def
    assert "extract_diagnoses" in mermaid_def
    assert "analyze_context" in mermaid_def
    assert "classify_diagnoses" in mermaid_def
    assert "retrieve_candidates" in mermaid_def
    assert "rank_candidates" in mermaid_def
    assert "validate_codes" in mermaid_def
    assert "evaluate_confidence" in mermaid_def
    assert "finalize_output" in mermaid_def
    assert "__end__" in mermaid_def or "END" in mermaid_def

    ascii_def = export_graph_ascii()
    assert "START" in ascii_def
    assert "validate_document" in ascii_def
    assert "END" in ascii_def


# ==============================================================================
# 7. DIAGNOSTIC FAILURE REPORTING FIXTURE
# ==============================================================================


@pytest.fixture
def diagnostic_pipeline_harness():
    """Harness running a variety of valid and deliberately broken inputs, producing a failure report."""

    class DiagnosticHarness:
        async def evaluate_suite(self) -> dict[str, Any]:
            cases = [
                (
                    "Valid Acute Systolic HF",
                    "Patient admitted with acute systolic heart failure and fluid overload.",
                ),
                (
                    "Unspecified Heart Failure",
                    "Patient has heart failure on maintenance oral diuretic.",
                ),
                (
                    "Ruled Out MI",
                    "Patient had chest discomfort. Acute myocardial infarction was ruled out by negative troponins.",
                ),
                ("Empty Document", "   "),
                (
                    "Gibberish / No Clinical Content",
                    "Hello world this is random text without medical findings.",
                ),
            ]
            report = []
            for name, text in cases:
                res = await process_clinical_document(text, document_id=f"diag-{name}")
                report.append(
                    {
                        "case_name": name,
                        "status": str(res.status),
                        "primary_code": res.primary_diagnosis.code
                        if res.primary_diagnosis
                        else None,
                        "secondary_count": len(res.secondary_diagnoses),
                        "abstention_count": len(res.abstentions),
                        "abstention_reasons": [str(a.reason) for a in res.abstentions],
                    }
                )
            return {"total_cases": len(cases), "cases": report}

    return DiagnosticHarness()


@pytest.mark.asyncio
async def test_diagnostic_failure_report_execution(
    diagnostic_pipeline_harness: Any,
) -> None:
    """Execute diagnostic evaluation and assert that failure cases are properly captured."""
    report = await diagnostic_pipeline_harness.evaluate_suite()

    assert report["total_cases"] == 5
    case_map = {c["case_name"]: c for c in report["cases"]}

    # Valid HF should succeed with primary I50.21
    assert case_map["Valid Acute Systolic HF"]["status"] in ("SUCCESS", "PARTIAL_SUCCESS")
    assert case_map["Valid Acute Systolic HF"]["primary_code"] == "I50.21"

    # Ruled Out MI must not code MI
    assert case_map["Ruled Out MI"]["primary_code"] != "I21.9"

    # Empty document must abstain
    assert case_map["Empty Document"]["status"] == "ABSTAINED"
    assert case_map["Empty Document"]["abstention_count"] >= 1


@pytest.mark.asyncio
async def test_end_to_end_pdf_extraction_and_coding(sample_discharge_summary: str) -> None:
    """Verify that a raw PDF input is extracted and processed through all 10 LangGraph nodes."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), sample_discharge_summary, fontsize=10)
    pdf_bytes = doc.write()
    doc.close()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(pdf_bytes)

    try:
        result = await process_clinical_document(
            source="",
            pdf_path=tmp_path,
            document_id="doc-pdf-e2e-001",
        )
        assert result is not None
        assert result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS)
        assert result.primary_diagnosis is not None
        assert result.primary_diagnosis.code == "I50.21"
        assert len(result.secondary_diagnoses) >= 1
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
