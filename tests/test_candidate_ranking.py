"""Comprehensive unit and integration tests for the ICD Candidate Ranking Component."""

import json
from typing import Any

import pytest

from medical_coding.agents.ranker import CandidateRankingAgent
from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.models.base import BaseLocalLLM
from medical_coding.schemas.clinical import (
    ClassifiedDiagnosis,
    ConditionClassification,
    ContextualizedDiagnosis,
    EvidenceSnippet,
)
from medical_coding.schemas.enums import (
    Acuity,
    Certainty,
    DiagnosisRole,
    Temporality,
)
from medical_coding.schemas.icd import ICDCandidate, ICDCodeRecord
from medical_coding.validation.deterministic import (
    CandidateRankingDeterministicValidator,
)


class MockLLM(BaseLocalLLM):
    """Configurable mock LLM for testing candidate ranking agent behaviors."""

    def __init__(self, response_text: str = "") -> None:
        self.response_text = response_text
        self.call_count = 0
        self.last_prompt = ""

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.response_text

    def get_model_info(self) -> dict[str, Any]:
        return {"model_name": "mock-ranking-llm", "offline_only": True}


@pytest.fixture
def ranking_test_catalog() -> LocalICDCatalog:
    """Authoritative local catalog containing verified test codes."""
    catalog = LocalICDCatalog()
    records = [
        ICDCodeRecord(
            code="I50.21",
            unformatted_code="I5021",
            description="Acute systolic heart failure",
            is_valid_billable=True,
            category="I50",
            synonyms=["acute systolic HF", "HFrEF acute"],
        ),
        ICDCodeRecord(
            code="I50.22",
            unformatted_code="I5022",
            description="Chronic systolic heart failure",
            is_valid_billable=True,
            category="I50",
            synonyms=["chronic systolic HF", "compensated systolic heart failure"],
        ),
        ICDCodeRecord(
            code="I50.31",
            unformatted_code="I5031",
            description="Acute diastolic heart failure",
            is_valid_billable=True,
            category="I50",
        ),
        ICDCodeRecord(
            code="I50.9",
            unformatted_code="I509",
            description="Heart failure, unspecified",
            is_valid_billable=True,
            category="I50",
            synonyms=["HF", "CHF", "heart failure"],
        ),
        ICDCodeRecord(
            code="I10",
            unformatted_code="I10",
            description="Essential primary hypertension",
            is_valid_billable=True,
            category="I10",
            synonyms=["HTN", "high blood pressure"],
        ),
        ICDCodeRecord(
            code="E11.9",
            unformatted_code="E119",
            description="Type 2 diabetes mellitus without complications",
            is_valid_billable=True,
            category="E11",
            synonyms=["T2DM", "type 2 diabetes"],
        ),
        ICDCodeRecord(
            code="E11.21",
            unformatted_code="E1121",
            description="Type 2 diabetes mellitus with diabetic nephropathy",
            is_valid_billable=True,
            category="E11",
            synonyms=["T2DM nephropathy", "diabetic kidney disease"],
        ),
    ]
    for r in records:
        catalog.add_record(r)
    catalog._initialized = True
    return catalog


@pytest.fixture
def heart_failure_candidates() -> list[ICDCandidate]:
    """Sample candidate pool retrieved from local dataset for heart failure."""
    return [
        ICDCandidate(
            code="I50.21",
            description="Acute systolic heart failure",
            retrieval_score=0.91,
            retrieval_method="hybrid",
            is_valid_billable=True,
            category="I50",
        ),
        ICDCandidate(
            code="I50.22",
            description="Chronic systolic heart failure",
            retrieval_score=0.78,
            retrieval_method="hybrid",
            is_valid_billable=True,
            category="I50",
        ),
        ICDCandidate(
            code="I50.9",
            description="Heart failure, unspecified",
            retrieval_score=0.82,
            retrieval_method="hybrid",
            is_valid_billable=True,
            category="I50",
        ),
    ]


# ==============================================================================
# 1. CLEAR CANDIDATE TEST
# ==============================================================================


def test_clear_candidate_selection(
    ranking_test_catalog: LocalICDCatalog,
    heart_failure_candidates: list[ICDCandidate],
) -> None:
    """Clear evidence with specific diagnostic terms must select the matching candidate."""
    evidence = "Patient admitted for acute exacerbation of systolic congestive heart failure. Echo showed EF 20%."
    condition = ClassifiedDiagnosis(
        diagnosis_id="diag-hf-clear",
        raw_term="acute systolic heart failure",
        context=ContextualizedDiagnosis(
            diagnosis_id="diag-hf-clear",
            raw_term="acute systolic heart failure",
            evidence=EvidenceSnippet(quote=evidence, source_section="Discharge Diagnosis"),
            extraction_confidence=1.0,
            acuity=Acuity.ACUTE,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            clinical_justification="Acute decompensated systolic heart failure actively treated.",
        ),
        role=DiagnosisRole.PRIMARY,
        is_billable_candidate=True,
        classification_reason="Reason for admission requiring IV diuretics.",
    )

    validator = CandidateRankingDeterministicValidator(catalog=ranking_test_catalog)
    # LLM confirms the clear candidate
    llm_response = json.dumps(
        {
            "selected_code": "I50.21",
            "selected_description": "Acute systolic heart failure",
            "ranking_reason": "Documentation explicitly states acute systolic heart failure with EF 20%.",
            "supporting_evidence": evidence,
            "confidence": 0.95,
            "decision": "ACCEPTED",
        }
    )
    agent = CandidateRankingAgent(llm=MockLLM(llm_response), validator=validator)

    selection = agent.rank_candidates(condition, heart_failure_candidates)

    assert selection.selected_code == "I50.21"
    assert selection.selected_description == "Acute systolic heart failure"
    assert selection.decision == "ACCEPTED"
    assert selection.confidence >= 0.85
    assert selection.abstention_reason is None
    assert selection.selected_candidate is not None
    assert selection.selected_candidate.code == "I50.21"


# ==============================================================================
# 2. AMBIGUOUS CANDIDATES TEST
# ==============================================================================


def test_ambiguous_candidates_evaluates_best_supported_level(
    ranking_test_catalog: LocalICDCatalog,
    heart_failure_candidates: list[ICDCandidate],
) -> None:
    """When documentation mentions general condition, agent must differentiate and pick best supported level."""
    evidence = "Past medical history significant for congestive heart failure and hypertension, clinically compensated."
    condition = ClassifiedDiagnosis(
        diagnosis_id="diag-hf-ambig",
        raw_term="congestive heart failure",
        context=ContextualizedDiagnosis(
            diagnosis_id="diag-hf-ambig",
            raw_term="congestive heart failure",
            evidence=EvidenceSnippet(quote=evidence, source_section="Past Medical History"),
            extraction_confidence=0.90,
            acuity=Acuity.CHRONIC,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            clinical_justification="Monitored on home regimen.",
        ),
        role=DiagnosisRole.SECONDARY,
        is_billable_candidate=True,
        classification_reason="Monitored secondary condition.",
    )

    validator = CandidateRankingDeterministicValidator(catalog=ranking_test_catalog)
    # Agent evaluates candidates deterministically or via LLM selecting unspecified
    agent = CandidateRankingAgent(llm=None, validator=validator)

    selection = agent.rank_candidates(condition, heart_failure_candidates)

    # Neither systolic nor acute was documented in the evidence; must NOT select I50.21 (acute systolic)
    assert selection.selected_code != "I50.21"
    # Should select either unspecified (I50.9) or chronic systolic if supported, or unspecified I50.9
    assert selection.selected_code in ["I50.9", "I50.22"]
    assert selection.decision == "ACCEPTED"


# ==============================================================================
# 3. UNSUPPORTED SPECIFICITY TEST
# ==============================================================================


def test_unsupported_specificity_realigned_to_unspecified(
    ranking_test_catalog: LocalICDCatalog,
    heart_failure_candidates: list[ICDCandidate],
) -> None:
    """If documentation only supports general condition, LLM cannot force acute systolic code."""
    evidence = "Discharge Diagnosis: Heart failure. Patient stabilized on furosemide."
    condition = ConditionClassification(
        diagnosis_id="diag-hf-general",
        diagnosis="Heart failure",
        role=DiagnosisRole.PRIMARY,
        is_billable_candidate=True,
        classification_reason="Primary reason for admission.",
        evidence_quote=evidence,
    )

    validator = CandidateRankingDeterministicValidator(catalog=ranking_test_catalog)
    # Model erroneously tries to pick highly specific code I50.21 despite lack of systolic proof
    hallucinated_specificity = json.dumps(
        {
            "selected_code": "I50.21",
            "selected_description": "Acute systolic heart failure",
            "ranking_reason": "Heart failure mention sounds severe so picking acute systolic.",
            "supporting_evidence": evidence,
            "confidence": 0.88,
            "decision": "ACCEPTED",
        }
    )
    agent = CandidateRankingAgent(llm=MockLLM(hallucinated_specificity), validator=validator)

    selection = agent.rank_candidates(condition, heart_failure_candidates)

    # Deterministic validation catches unsupported systolic specificity and realigns to I50.9 (unspecified)!
    assert selection.selected_code == "I50.9"
    assert "systolic specificity" in selection.ranking_reason.lower()
    assert selection.decision == "ACCEPTED"


def test_unsupported_specificity_abstains_when_no_general_code_present(
    ranking_test_catalog: LocalICDCatalog,
) -> None:
    """If only specific candidates exist and documentation is general, agent must abstain."""
    evidence = "Diagnosis: Heart failure NOS."
    condition = ConditionClassification(
        diagnosis_id="diag-hf-only-specific",
        diagnosis="Heart failure",
        role=DiagnosisRole.SECONDARY,
        is_billable_candidate=True,
        classification_reason="Monitored condition.",
        evidence_quote=evidence,
    )
    # Candidate pool contains ONLY specific codes (no I50.9)
    specific_only_candidates = [
        ICDCandidate(
            code="I50.21",
            description="Acute systolic heart failure",
            retrieval_score=0.75,
            retrieval_method="faiss",
            is_valid_billable=True,
        ),
        ICDCandidate(
            code="I50.31",
            description="Acute diastolic heart failure",
            retrieval_score=0.70,
            retrieval_method="faiss",
            is_valid_billable=True,
        ),
    ]

    validator = CandidateRankingDeterministicValidator(catalog=ranking_test_catalog)
    llm_resp = json.dumps(
        {
            "selected_code": "I50.21",
            "selected_description": "Acute systolic heart failure",
            "ranking_reason": "Forcing systolic.",
            "supporting_evidence": evidence,
            "confidence": 0.75,
            "decision": "ACCEPTED",
        }
    )
    agent = CandidateRankingAgent(llm=MockLLM(llm_resp), validator=validator)

    selection = agent.rank_candidates(condition, specific_only_candidates)

    # Must abstain because evidence does NOT support systolic or diastolic, and no general code exists in pool
    assert selection.selected_code is None
    assert selection.decision == "ABSTAINED"
    assert selection.abstention_reason == "UNSUPPORTED_SPECIFICITY"


# ==============================================================================
# 4. CANDIDATE NOT SELECTED TEST
# ==============================================================================


def test_candidate_not_selected_unrelated_condition(
    ranking_test_catalog: LocalICDCatalog,
    heart_failure_candidates: list[ICDCandidate],
) -> None:
    """When candidates have zero clinical support in evidence, no candidate should be selected."""
    evidence = "Patient underwent skin biopsy for suspected dermatological rash on forearm."
    condition = ConditionClassification(
        diagnosis_id="diag-rash",
        diagnosis="Eczematous dermatitis",
        role=DiagnosisRole.SECONDARY,
        is_billable_candidate=True,
        classification_reason="Skin examination finding.",
        evidence_quote=evidence,
    )

    validator = CandidateRankingDeterministicValidator(catalog=ranking_test_catalog)
    # LLM correctly abstains
    llm_resp = json.dumps(
        {
            "selected_code": None,
            "selected_description": None,
            "ranking_reason": "None of the heart failure candidates match the dermatological evidence.",
            "supporting_evidence": evidence,
            "confidence": 0.0,
            "decision": "ABSTAINED",
            "abstention_reason": "NO_MATCHING_CANDIDATE",
        }
    )
    agent = CandidateRankingAgent(llm=MockLLM(llm_resp), validator=validator)

    selection = agent.rank_candidates(condition, heart_failure_candidates)

    assert selection.selected_code is None
    assert selection.decision == "ABSTAINED"
    assert selection.abstention_reason in ["NO_MATCHING_CANDIDATE", "CONFIDENCE_BELOW_THRESHOLD"]


# ==============================================================================
# 5. HARD CONSTRAINT: INVALID LLM CODE REJECTION
# ==============================================================================


def test_invalid_llm_code_deterministic_rejection(
    ranking_test_catalog: LocalICDCatalog,
    heart_failure_candidates: list[ICDCandidate],
) -> None:
    """CRITICAL HARD CONSTRAINT: If LLM hallucinates a code not in candidate list, it MUST be rejected."""
    evidence = "Patient has severe acute systolic heart failure."
    condition = ConditionClassification(
        diagnosis_id="diag-hf",
        diagnosis="acute systolic heart failure",
        role=DiagnosisRole.PRIMARY,
        is_billable_candidate=True,
        classification_reason="Primary admission reason.",
        evidence_quote=evidence,
    )

    validator = CandidateRankingDeterministicValidator(catalog=ranking_test_catalog)
    # LLM hallucinates a code 'I42.0' (Dilated cardiomyopathy) which is NOT in candidates list [I50.21, I50.22, I50.9]!
    hallucinated_code_response = json.dumps(
        {
            "selected_code": "I42.0",  # NOT in candidates!
            "selected_description": "Dilated cardiomyopathy",
            "ranking_reason": "I think patient also has dilated cardiomyopathy.",
            "supporting_evidence": evidence,
            "confidence": 0.99,
            "decision": "ACCEPTED",
        }
    )
    agent = CandidateRankingAgent(
        llm=MockLLM(hallucinated_code_response),
        validator=validator,
        allow_rerun=False,
    )

    selection = agent.rank_candidates(condition, heart_failure_candidates)

    # MUST REJECT! The code does NOT exist in candidate list
    assert selection.selected_code is None
    assert selection.decision == "REJECTED_MISMATCH"
    assert selection.abstention_reason == "INVALID_LLM_CODE_NOT_IN_CANDIDATE_POOL"
    assert selection.confidence == 0.0
    assert "Deterministic rejection" in selection.ranking_reason


# ==============================================================================
# 6. INSUFFICIENT EVIDENCE TEST
# ==============================================================================


def test_insufficient_evidence_triggers_clean_abstention(
    ranking_test_catalog: LocalICDCatalog,
    heart_failure_candidates: list[ICDCandidate],
) -> None:
    """Missing or empty evidence quote must immediately abstain without assigning a code."""
    condition = ConditionClassification(
        diagnosis_id="diag-no-ev",
        diagnosis="Essential hypertension",
        role=DiagnosisRole.SECONDARY,
        is_billable_candidate=True,
        classification_reason="Listed without note.",
        evidence_quote="",  # Empty evidence!
    )

    validator = CandidateRankingDeterministicValidator(catalog=ranking_test_catalog)
    mock_llm = MockLLM()
    agent = CandidateRankingAgent(llm=mock_llm, validator=validator)

    selection = agent.rank_candidates(condition, heart_failure_candidates)

    assert selection.selected_code is None
    assert selection.decision == "ABSTAINED"
    assert selection.abstention_reason == "INSUFFICIENT_EVIDENCE"
    # Guaranteed: No LLM calls made when evidence is missing!
    assert mock_llm.call_count == 0


# ==============================================================================
# 7. NO CANDIDATES TEST
# ==============================================================================


def test_no_candidates_pool_triggers_immediate_abstention(
    ranking_test_catalog: LocalICDCatalog,
) -> None:
    """When candidate pool is empty, agent must cleanly abstain with zero LLM inference."""
    condition = ConditionClassification(
        diagnosis_id="diag-empty-pool",
        diagnosis="Rare metabolic disease",
        role=DiagnosisRole.SECONDARY,
        is_billable_candidate=True,
        classification_reason="Evaluated.",
        evidence_quote="Metabolic workup documented.",
    )

    validator = CandidateRankingDeterministicValidator(catalog=ranking_test_catalog)
    mock_llm = MockLLM()
    agent = CandidateRankingAgent(llm=mock_llm, validator=validator)

    selection = agent.rank_candidates(condition, candidates=[])  # Empty candidates list!

    assert selection.selected_code is None
    assert selection.decision == "ABSTAINED"
    assert selection.abstention_reason == "NO_CANDIDATES"
    assert selection.candidate_pool == []
    # Zero LLM inference calls
    assert mock_llm.call_count == 0


# ==============================================================================
# 8. CANDIDATE-RANKING EVALUATION FIXTURE
# ==============================================================================


@pytest.fixture
def candidate_ranking_evaluation_fixture(ranking_test_catalog: LocalICDCatalog):
    """Reusable evaluation fixture executing a benchmark suite across clinical scenarios."""

    class CandidateRankingEvaluationHarness:
        def __init__(self) -> None:
            self.catalog = ranking_test_catalog
            self.validator = CandidateRankingDeterministicValidator(catalog=self.catalog)
            self.agent = CandidateRankingAgent(llm=None, validator=self.validator)

        def run_benchmark(self) -> dict[str, Any]:
            test_cases = [
                {
                    "name": "Exact Acute Systolic HF",
                    "term": "Acute systolic heart failure",
                    "evidence": "Admitted with acute systolic congestive heart failure exacerbation.",
                    "candidates": [
                        ICDCandidate(
                            code="I50.21",
                            description="Acute systolic heart failure",
                            retrieval_score=0.95,
                            retrieval_method="hybrid",
                            is_valid_billable=True,
                        ),
                        ICDCandidate(
                            code="I50.9",
                            description="Heart failure, unspecified",
                            retrieval_score=0.80,
                            retrieval_method="hybrid",
                            is_valid_billable=True,
                        ),
                    ],
                    "expected_code": "I50.21",
                    "expected_decision": "ACCEPTED",
                },
                {
                    "name": "General HF without Subtype",
                    "term": "Congestive heart failure",
                    "evidence": "Patient with congestive heart failure on maintenance oral diuretic.",
                    "candidates": [
                        ICDCandidate(
                            code="I50.21",
                            description="Acute systolic heart failure",
                            retrieval_score=0.88,
                            retrieval_method="hybrid",
                            is_valid_billable=True,
                        ),
                        ICDCandidate(
                            code="I50.9",
                            description="Heart failure, unspecified",
                            retrieval_score=0.84,
                            retrieval_method="hybrid",
                            is_valid_billable=True,
                        ),
                    ],
                    "expected_code": "I50.9",
                    "expected_decision": "ACCEPTED",
                },
                {
                    "name": "Uncomplicated Type 2 Diabetes",
                    "term": "Type 2 diabetes mellitus",
                    "evidence": "Blood glucose was monitored; patient has type 2 diabetes on metformin.",
                    "candidates": [
                        ICDCandidate(
                            code="E11.9",
                            description="Type 2 diabetes mellitus without complications",
                            retrieval_score=0.90,
                            retrieval_method="hybrid",
                            is_valid_billable=True,
                        ),
                        ICDCandidate(
                            code="E11.21",
                            description="Type 2 diabetes mellitus with diabetic nephropathy",
                            retrieval_score=0.85,
                            retrieval_method="hybrid",
                            is_valid_billable=True,
                        ),
                    ],
                    "expected_code": "E11.9",
                    "expected_decision": "ACCEPTED",
                },
                {
                    "name": "Alien Disease No Match",
                    "term": "Extraterrestrial pulmonary syndrome",
                    "evidence": "Patient reports alien infection.",
                    "candidates": [
                        ICDCandidate(
                            code="I50.9",
                            description="Heart failure, unspecified",
                            retrieval_score=0.20,
                            retrieval_method="hybrid",
                            is_valid_billable=True,
                        ),
                    ],
                    "expected_code": None,
                    "expected_decision": "ABSTAINED",
                },
            ]

            results = []
            for case in test_cases:
                cond = ConditionClassification(
                    diagnosis_id=f"bench-{case['name']}",
                    diagnosis=case["term"],
                    role=DiagnosisRole.PRIMARY,
                    is_billable_candidate=True,
                    classification_reason="Benchmark evaluation",
                    evidence_quote=case["evidence"],
                )
                sel = self.agent.rank_candidates(cond, case["candidates"])
                passed = (sel.selected_code == case["expected_code"]) and (
                    sel.decision == case["expected_decision"]
                )
                results.append(
                    {
                        "case_name": case["name"],
                        "passed": passed,
                        "selected_code": sel.selected_code,
                        "decision": sel.decision,
                        "confidence": sel.confidence,
                        "reason": sel.ranking_reason,
                    }
                )

            total = len(results)
            passed_count = sum(1 for r in results if r["passed"])
            return {
                "total_cases": total,
                "passed_count": passed_count,
                "accuracy": passed_count / total if total > 0 else 0.0,
                "results": results,
            }

    return CandidateRankingEvaluationHarness()


def test_candidate_ranking_evaluation_fixture_execution(
    candidate_ranking_evaluation_fixture: Any,
) -> None:
    """Execute the candidate-ranking evaluation fixture and verify benchmark metrics."""
    metrics = candidate_ranking_evaluation_fixture.run_benchmark()

    assert metrics["total_cases"] == 4
    assert metrics["passed_count"] == 4
    assert metrics["accuracy"] == 1.0
    for r in metrics["results"]:
        assert r["passed"] is True
