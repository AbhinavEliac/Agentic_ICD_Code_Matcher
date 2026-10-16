"""Targeted tests for main-fix-1 branch requirements:
1. Noise & normal findings rejection (exam, lab, routine care).
2. Grounded vocabulary normalization without hallucinated non-source words.
3. Strict UHDDS primary vs secondary reasoning (no false promotion of symptoms or baseline conditions).
4. Strengthened concept and attribute matching (core entity mismatch zeroing, laterality/site disqualification).
"""

import pytest
from medical_coding.validation.clinical_gate import HardClinicalCandidateGate
from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor
from medical_coding.agents.classifier import PrimarySecondaryClassifier
from medical_coding.reasoning.concept_matcher import compute_concept_match_score
from medical_coding.schemas.clinical import (
    ConditionStatus,
    ContextAssessment,
    ExtractedClinicalCondition,
)
from medical_coding.schemas.enums import (
    Certainty,
    DiagnosisRole,
    Temporality,
)
from medical_coding.models.base import BaseLocalLLM


class MockLLM(BaseLocalLLM):
    def generate(self, prompt: str) -> str:
        return ""

    def get_model_info(self) -> dict:
        return {"model_name": "mock-llm"}


def test_normal_findings_rejected_by_gate():
    """Verify normal physical exam, normal lab results, and routine nursing notes are rejected."""
    normal_phrases = [
        "per abdomen soft, non-tender",
        "P/A soft, non tender",
        "chest clear",
        "bilateral air entry equal",
        "GCS 15/15",
        "conscious, alert and oriented",
        "no pedal edema",
        "no organomegaly",
        "no icterus",
        "pupils equal and reactive",
        "hemodynamically stable",
        "CBC normal",
        "LFT within normal limits",
        "blood culture sterile",
        "tolerating oral diet",
        "adequate urine output",
        "wound dressing done",
        "foley catheter removed",
    ]
    for phrase in normal_phrases:
        is_valid, reason = HardClinicalCandidateGate.evaluate_candidate(phrase)
        assert not is_valid, f"Expected '{phrase}' to be rejected as a normal finding/routine care, but passed!"
        assert HardClinicalCandidateGate.is_normal_finding(phrase) is True


def test_grounded_normalization_preserves_source_vocabulary():
    """Verify normalization does not inject words that were never written in discharge summary."""
    extractor = EvidenceFirstFactExtractor()

    # 1. Hypertension should remain 'Hypertension', not hallucinate 'Essential'
    assert extractor._canonicalize_term("Hypertension") == "Hypertension"
    assert extractor._canonicalize_term("Essential hypertension") == "Essential hypertension"

    # 2. Colitis should remain 'Colitis', not inject 'Noninfective gastroenteritis and colitis, unspecified'
    assert extractor._canonicalize_term("Colitis") == "Colitis"
    assert extractor._canonicalize_term("noninfective colitis") == "Colitis"

    # 3. Thrombocytopenia should remain 'Thrombocytopenia' unless secondary is documented
    assert extractor._canonicalize_term("Thrombocytopenia") == "Thrombocytopenia"
    assert extractor._canonicalize_term("Secondary thrombocytopenia") == "Secondary thrombocytopenia"

    # 4. PUJ calculus should remain 'PUJ calculus', not inject full ICD description
    assert extractor._canonicalize_term("Left PUJ calculus") == "PUJ calculus"
    assert extractor._canonicalize_term("Renal calculus") == "Renal calculus"

    # 5. Flu-like illness should remain 'Flu-like illness', not inject 'Unspecified acute lower respiratory infection'
    assert extractor._canonicalize_term("Viral flu like illness") == "Flu-like illness"

    # 6. Type 2 diabetes should remain 'Type 2 diabetes' when mellitus is omitted
    assert extractor._canonicalize_term("Type 2 diabetes") == "Type 2 diabetes"
    assert extractor._canonicalize_term("Type 2 diabetes mellitus") == "Type 2 diabetes mellitus"


def test_pure_symptoms_cannot_be_falsely_promoted_to_primary():
    """Verify UHDDS rule: pure symptoms without explicit provider designation cannot become primary."""
    classifier = PrimarySecondaryClassifier(llm=MockLLM())

    # Patient with routine non-specific fever and baseline chronic hypertension
    c1 = ContextAssessment(
        condition_id="c1",
        diagnosis="Essential hypertension",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.CHRONIC,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Known hypertension on amlodipine.",
        reason="Baseline chronic disease.",
        treated_or_managed=False,
    )
    c2 = ContextAssessment(
        condition_id="c2",
        diagnosis="Headache",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Patient complained of mild headache.",
        reason="Symptom complaint.",
        treated_or_managed=False,
    )

    classifications = classifier.classify_diagnoses([c1, c2], clinical_text="")
    # Neither headache (symptom) nor hypertension (baseline chronic) should be falsely promoted to primary
    primaries = [c for c in classifications if c.role == DiagnosisRole.PRIMARY]
    assert len(primaries) == 0, f"Expected 0 primary diagnoses, but got: {[p.diagnosis for p in primaries]}"

    # All should be secondary
    secondaries = [c for c in classifications if c.role == DiagnosisRole.SECONDARY]
    assert len(secondaries) == 2


def test_concept_matcher_core_entity_mismatch_and_laterality():
    """Verify strengthened concept matcher zeroes out incompatible core entities and contradictions."""
    # Core entity mismatch: Kidney stone vs Skin disorder code (L98.9)
    score, rationale, is_exact = compute_concept_match_score(
        candidate_code="L98.9",
        candidate_description="Disorder of the skin and subcutaneous tissue, unspecified",
        concept_name="Renal calculus",
        evidence_text="Patient has a 6mm renal calculus in left kidney.",
    )
    assert score == 0.0
    assert "mismatch" in rationale.lower()

    # Laterality contradiction: Left PUJ calculus matching a right-sided code
    score_lat, rationale_lat, is_exact_lat = compute_concept_match_score(
        candidate_code="N20.1",
        candidate_description="Calculus of ureter, right",
        concept_name="Left PUJ calculus",
        evidence_text="CT revealed left PUJ calculus.",
    )
    assert score_lat == 0.0
    assert "laterality" in rationale_lat.lower()
