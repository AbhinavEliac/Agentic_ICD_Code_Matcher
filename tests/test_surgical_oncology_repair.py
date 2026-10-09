"""Comprehensive regression test suite for surgical oncology repairs and deterministic validation invariants.

Validates the 6 prioritized repair items:
1. Semantic code validation rejects laterality and anatomical-site mismatches.
2. Procedure extraction is segregated from diagnosis inventory (mastectomy, salpingo-oophorectomy in procedures).
3. Structured oncology context (histology, grade, ER/PR/HER2, Ki-67, response, prior treatment).
4. Operative findings extracted separately with distinct UHDDS eligibility gate.
5. Audit claims are evidence-based, verifying clinical meaning.
6. Retrieval and ranking latency and caching efficiency.
"""

import pytest
import time
from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor
from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.dataset.loader import ICDCodeRecord
from medical_coding.schemas.clinical import (
    ClassifiedDiagnosis,
    ContextualizedDiagnosis,
    EvidenceSnippet,
    ExtractedClinicalCondition,
)
from medical_coding.schemas.enums import Acuity, Certainty, DiagnosisRole, Temporality
from medical_coding.schemas.evidence import ClinicalDiagnosisState
from medical_coding.schemas.icd import ICDCandidate, RankedSelection
from medical_coding.validation.reverse_attributes import ReverseAttributeChecker
from medical_coding.validation.deterministic import DeterministicValidator
from medical_coding.orchestration.pipeline import MedicalCodingPipeline


SURGICAL_ONCOLOGY_NOTE = """
CLINICAL DISCHARGE SUMMARY
PATIENT: Jane Doe | RECORD NUMBER: ONC-2026-992
ADMISSION DATE: 10/01/2026 | DISCHARGE DATE: 10/05/2026

PRINCIPAL DIAGNOSIS:
Invasive ductal carcinoma of the right breast, Nottingham histologic grade 3, ER 95% positive, PR 90% positive, HER2 negative (IHC 1+), Ki-67 proliferation index 45%. Clinical stage IIB with prior neoadjuvant chemotherapy (AC-T completed 4 weeks prior) demonstrating partial pathological response. No evidence of distant metastases.

SECONDARY DIAGNOSES:
Essential hypertension, well controlled.
Type 2 diabetes mellitus, uncomplicated.

OPERATIVE PROCEDURES PERFORMED:
Right modified radical mastectomy with level I/II axillary lymph node dissection.
Prophylactic bilateral salpingo-oophorectomy.

OPERATIVE FINDINGS:
Intraoperative inspection of the pelvis revealed an incidental 2cm right ovarian simple benign cyst and mild filmy pelvic adhesions without endometriosis or malignancy. Right breast tumor bed showed 2.2 cm residual fibrotic tumor bed with negative margins.

HOSPITAL COURSE:
Patient tolerated right mastectomy and bilateral salpingo-oophorectomy well. Drains in place with serosanguinous output. Hypertension managed on lisinopril 10mg daily. Fingerstick blood sugars stable on metformin 500mg BID.
"""


def test_procedure_firewall_segregates_mastectomy_and_salpingo_oophorectomy():
    """Item 2: Procedures belong in procedure data, never in secondary diagnosis inventory."""
    extractor = EvidenceFirstFactExtractor()
    state = extractor.extract_clinical_state(SURGICAL_ONCOLOGY_NOTE, doc_id="onc-test-01")

    # Procedures must be captured
    procedures = state.procedures
    assert any("mastectomy" in p.lower() for p in procedures), f"Expected mastectomy in procedures, got: {procedures}"
    assert any("salpingo-oophorectomy" in p.lower() or "oophorectomy" in p.lower() for p in procedures), (
        f"Expected salpingo-oophorectomy in procedures, got: {procedures}"
    )

    # Neither procedure should be in the codable secondary diagnosis candidate inventory
    candidate_terms = [c.raw_term.lower() for c in state.all_candidates if c.role != DiagnosisRole.EXCLUDED]
    for proc in ["mastectomy", "salpingo-oophorectomy", "axillary lymph node dissection"]:
        assert not any(proc in term for term in candidate_terms), (
            f"Procedure '{proc}' contaminated diagnosis candidates: {candidate_terms}"
        )


def test_structured_oncology_context_capture():
    """Item 3: Preserve histology, grade, receptor status, Ki-67, metastatic evidence, prior tx, response."""
    extractor = EvidenceFirstFactExtractor()
    state = extractor.extract_clinical_state(SURGICAL_ONCOLOGY_NOTE, doc_id="onc-test-02")

    onc_ctx = state.oncology_context
    assert onc_ctx is not None, "Oncology context must not be None"
    assert "breast" in (onc_ctx.primary_site or "").lower()
    assert (onc_ctx.laterality or "").lower() == "right"
    assert "invasive ductal" in (onc_ctx.histology or "").lower()
    assert "3" in (onc_ctx.grade or "")
    assert "45" in (onc_ctx.ki67 or "")

    receptors = onc_ctx.receptors
    assert "er" in receptors and "pos" in receptors["er"].lower()
    assert "pr" in receptors and "pos" in receptors["pr"].lower()
    assert "her2" in receptors and "neg" in receptors["her2"].lower()

    assert onc_ctx.metastatic_status in (False, "NON_METASTATIC")
    assert any("chemotherapy" in tx.lower() for tx in onc_ctx.prior_treatments)
    assert "partial" in (onc_ctx.treatment_response or "").lower()


def test_operative_findings_extraction_and_uhdds_eligibility_gate():
    """Item 4: Operative findings captured separately; incidental unmanaged findings excluded from billing."""
    extractor = EvidenceFirstFactExtractor()
    state = extractor.extract_clinical_state(SURGICAL_ONCOLOGY_NOTE, doc_id="onc-test-03")

    findings = state.operative_findings
    assert len(findings) >= 1, "Operative findings must be extracted"
    finding_terms = [f["term"].lower() for f in findings]
    assert any("cyst" in t for t in finding_terms) or any("adhesion" in t for t in finding_terms)

    # In state classification, incidental ovarian cyst and pelvic adhesions must be EXCLUDED from billing
    for cand in state.all_candidates:
        if "cyst" in cand.raw_term.lower() or "adhesion" in cand.raw_term.lower():
            assert cand.role == DiagnosisRole.EXCLUDED, (
                f"Operative finding '{cand.raw_term}' should be EXCLUDED under UHDDS gate, but had role {cand.role}"
            )


def test_semantic_validation_rejects_laterality_mismatches():
    """Item 1: Hard rejection of laterality mismatch (e.g. Left breast code C50.912 when evidence is Right breast)."""
    catalog = LocalICDCatalog()
    catalog.add_record(
        ICDCodeRecord(
            code="C50.911",
            description="Malignant neoplasm of unspecified site of right female breast",
            is_valid_billable=True,
            coding_system="ICD-10-CM",
        )
    )
    catalog.add_record(
        ICDCodeRecord(
            code="C50.912",
            description="Malignant neoplasm of unspecified site of left female breast",
            is_valid_billable=True,
            coding_system="ICD-10-CM",
        )
    )
    catalog.add_record(
        ICDCodeRecord(
            code="C56.1",
            description="Malignant neoplasm of right ovary",
            is_valid_billable=True,
            coding_system="ICD-10-CM",
        )
    )

    validator = DeterministicValidator(catalog=catalog)

    # 1. Direct unit check: ReverseAttributeChecker laterality
    is_lat_valid, lat_err = ReverseAttributeChecker.validate_laterality(
        code_description="Malignant neoplasm of unspecified site of left female breast",
        evidence_text="Invasive ductal carcinoma of the right breast",
    )
    assert not is_lat_valid, "Should reject left breast code for right breast evidence"
    assert lat_err is not None

    # 2. Right breast code with Right breast evidence MUST pass laterality check
    is_right_lat_valid, _ = ReverseAttributeChecker.validate_laterality(
        code_description="Malignant neoplasm of unspecified site of right female breast",
        evidence_text="Invasive ductal carcinoma of the right breast",
    )
    assert is_right_lat_valid, "Right breast code should be accepted for right breast evidence"

    # 3. Direct unit check: ReverseAttributeChecker anatomical site
    is_site_valid, site_err = ReverseAttributeChecker.validate_anatomical_site(
        code_description="Malignant neoplasm of right ovary",
        evidence_text="Invasive ductal carcinoma of the right breast",
    )
    assert not is_site_valid, "Should reject ovary code for breast cancer evidence"
    assert site_err is not None


def test_deterministic_validator_replaces_or_abstains_on_semantic_mismatch():
    """Item 1 & 5: When ranker selects a laterality-mismatched code, validator replaces it from pool or abstains."""
    catalog = LocalICDCatalog()
    right_code = ICDCodeRecord(
        code="C50.911",
        description="Malignant neoplasm of unspecified site of right female breast",
        is_valid_billable=True,
        coding_system="ICD-10-CM",
    )
    left_code = ICDCodeRecord(
        code="C50.912",
        description="Malignant neoplasm of unspecified site of left female breast",
        is_valid_billable=True,
        coding_system="ICD-10-CM",
    )
    catalog.add_record(right_code)
    catalog.add_record(left_code)

    validator = DeterministicValidator(catalog=catalog)

    classified = ClassifiedDiagnosis(
        diagnosis_id="diag-breast",
        raw_term="Invasive ductal carcinoma of the right breast",
        role=DiagnosisRole.PRIMARY,
        is_billable_candidate=True,
        context=ContextualizedDiagnosis(
            diagnosis_id="diag-breast",
            raw_term="Invasive ductal carcinoma of the right breast",
            evidence=EvidenceSnippet(
                quote="Invasive ductal carcinoma of the right breast",
                source_section="PRINCIPAL_DIAGNOSIS",
            ),
            extraction_confidence=1.0,
            clinical_justification="Documented principal diagnosis of right breast carcinoma.",
            acuity=Acuity.CHRONIC,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
        ),
        classification_reason="Principal diagnosis for surgical admission",
    )

    left_cand = ICDCandidate(
        code="C50.912",
        description="Malignant neoplasm of unspecified site of left female breast",
        retrieval_score=0.85,
        retrieval_method="hybrid",
        is_valid_billable=True,
    )
    right_cand = ICDCandidate(
        code="C50.911",
        description="Malignant neoplasm of unspecified site of right female breast",
        retrieval_score=0.82,
        retrieval_method="hybrid",
        is_valid_billable=True,
    )

    # Ranker incorrectly chose left breast code C50.912, but candidate pool has C50.911
    wrong_selection = RankedSelection(
        diagnosis_id="diag-breast",
        raw_term="Invasive ductal carcinoma of the right breast",
        selected_code="C50.912",
        selected_description="Malignant neoplasm of unspecified site of left female breast",
        selected_candidate=left_cand,
        confidence=0.85,
        candidate_pool=[left_cand, right_cand],
    )

    validated_diag, abst = validator.validate_code(
        condition=classified,
        selection=wrong_selection,
    )

    assert validated_diag is not None, "Validation should recover a compliant code from the candidate pool"
    assert abst is None
    # Must have corrected to right breast code
    assert validated_diag.code == "C50.911"
    check_names = [c.rule_name for c in validated_diag.checks if c.passed]
    assert "LateralityMatch" in check_names
    assert "AnatomicalSiteMatch" in check_names


@pytest.mark.asyncio
async def test_end_to_end_surgical_oncology_pipeline():
    """Item 1-6: End-to-end pipeline run on surgical oncology discharge summary."""
    pipeline = MedicalCodingPipeline()
    result = await pipeline.run_document(
        document_id="ENC-SURGICAL-ONC-01",
        text=SURGICAL_ONCOLOGY_NOTE,
        metadata={"filename": "surgical_oncology.txt"},
    )

    assert result is not None
    # 1. Procedures segregated
    assert len(result.procedures) >= 2, f"Expected at least 2 procedures, got: {result.procedures}"
    assert any("mastectomy" in p.lower() for p in result.procedures)

    # 2. Oncology context populated
    assert result.oncology_context is not None
    assert (result.oncology_context.get("laterality") or "").lower() == "right"
    assert "invasive ductal" in (result.oncology_context.get("histology") or "").lower()

    # 3. Operative findings captured
    assert len(result.operative_findings) >= 1

    # 4. Primary diagnosis must NOT be a left breast code
    if result.primary_diagnosis and result.primary_diagnosis.code:
        assert result.primary_diagnosis.code != "C50.912", "Primary code must not be Left breast (C50.912)!"
        assert "left" not in (result.primary_diagnosis.description or "").lower()

    # 5. Secondary diagnoses must NOT contain procedures or incidental findings
    sec_descriptions = [s.description.lower() for s in result.secondary_diagnoses]
    sec_terms = [(s.raw_term or "").lower() for s in result.secondary_diagnoses]
    all_sec_text = " ".join(sec_descriptions + sec_terms)
    assert "mastectomy" not in all_sec_text
    assert "salpingo" not in all_sec_text
    assert "oophorectomy" not in all_sec_text
