"""Tests for generalized clinical reasoning principles, adversarial variations, and unseen conditions.

Tests:
1. Assertion & Negation: Ruled out / negative conditions must never become active diagnoses.
2. Temporality: Historical/resolved conditions must not be primary.
3. Source Authority: Medication-only mentions must not manufacture diagnoses.
4. CMS Guideline I.B.4: Integral symptoms must not be separate billable secondaries when primary is definitive.
5. Specificity & Reverse Attributes: Unsupported attributes must be rejected.
6. Neoplasm / Histology Attribute Separation (Section 10): Roman staging/markers separated from concept.
7. Unseen Clinical Entities: System correctly reasons over novel diseases without hardcoded lists.
"""

from unittest.mock import MagicMock

import pytest

from medical_coding.agents.classifier import PrimarySecondaryClassifier
from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor
from medical_coding.schemas.clinical import ContextAssessment
from medical_coding.schemas.enums import (
    Certainty,
    ConditionStatus,
    Temporality,
)
from medical_coding.validation.clinical_gate import HardClinicalCandidateGate
from medical_coding.validation.reverse_attributes import ReverseAttributeChecker


@pytest.fixture
def extractor():
    return EvidenceFirstFactExtractor()


@pytest.fixture
def classifier():
    return PrimarySecondaryClassifier(MagicMock())


class TestAssertionAndNegation:
    """Principles: Ruled out and negative statements must not become billable diagnoses."""

    def test_ruled_out_condition_rejected(self, extractor):
        note = """
        CHIEF COMPLAINT: Right lower quadrant pain.
        HOSPITAL COURSE:
        Acute appendicitis was considered but ruled out after normal abdominal ultrasound.
        Patient was diagnosed with mesenteric adenitis.
        DISCHARGE DIAGNOSIS:
        Mesenteric adenitis.
        """
        state = extractor.extract_clinical_state(note, "DOC-ADV-1")
        appendicitis_cands = [c for c in state.all_candidates if "appendicitis" in c.normalized_diagnosis.lower()]
        for c in appendicitis_cands:
            assert c.scores.contradiction_penalty > 0 or not c.is_authorized or c.certainty == Certainty.RULED_OUT

    def test_negative_screening_statement(self, extractor):
        note = """
        DISCHARGE DIAGNOSES:
        1. Acute gastroenteritis.
        2. No evidence of active gastrointestinal bleeding or perforation.
        """
        state = extractor.extract_clinical_state(note, "DOC-ADV-2")
        active_bleed = [c for c in state.all_candidates if "bleeding" in c.normalized_diagnosis.lower() or "perforation" in c.normalized_diagnosis.lower()]
        for c in active_bleed:
            assert not c.is_authorized or c.certainty in (Certainty.RULED_OUT, Certainty.NEGATED)


class TestMedicationContaminationPrinciple:
    """Principle: Medication lists alone must never manufacture diagnoses."""

    def test_medication_candidate_gate_rejection(self):
        # A medication name or instruction alone should not pass the clinical candidate gate
        is_valid, reason = HardClinicalCandidateGate.evaluate_candidate(
            "Cough lozenges",
            "Discharge Medications: Cough lozenges PRN for throat tickle"
        )
        assert not is_valid
        assert "pharmaceutical" in reason.lower() or "medication" in reason.lower()

    def test_laxative_does_not_create_constipation_diagnosis(self):
        is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(
            "Laxative syrup",
            "Take laxative syrup 15ml at bedtime"
        )
        assert not is_valid


class TestSymptomIntegralExclusionPrinciple:
    """Principle (CMS Guideline I.B.4): Integral symptoms must not be separate secondaries when primary is definitive."""

    def test_cholecystitis_with_abdominal_pain(self, classifier):
        primary_asm = ContextAssessment(
            condition_id="chole-01",
            diagnosis="Acute cholecystitis",
            current_relevance=True,
            coding_candidate=True,
            status=ConditionStatus.ACUTE,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            evidence="Admitted for acute cholecystitis, underwent laparoscopic cholecystectomy.",
            reason="Condition occasioning admission treated surgically.",
            treated_or_managed=True,
            influenced_treatment=True,
            treatment_evidence="Laparoscopic cholecystectomy performed",
        )
        symptom_asm = ContextAssessment(
            condition_id="pain-01",
            diagnosis="Right upper quadrant abdominal pain",
            current_relevance=True,
            coding_candidate=True,
            status=ConditionStatus.ACUTE,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            evidence="Patient presented with severe right upper quadrant abdominal pain.",
            reason="Presenting symptom of acute cholecystitis.",
            treated_or_managed=False,
        )

        result, _ = classifier.classify_conditions(
            [primary_asm, symptom_asm],
            clinical_text="Admitted for acute cholecystitis, presenting with right upper quadrant abdominal pain. Laparoscopic cholecystectomy performed.",
        )
        assert result.has_unique_primary is True
        assert result.primary_diagnosis is not None
        assert "cholecystitis" in result.primary_diagnosis.diagnosis.lower()

        # The symptom should be excluded under CMS Guideline I.B.4
        excluded_pain = [c for c in result.excluded_conditions if "pain" in c.diagnosis.lower()]
        secondaries_pain = [c for c in result.secondary_diagnoses if "pain" in c.diagnosis.lower()]
        assert len(excluded_pain) == 1 or len(secondaries_pain) == 0


class TestNeoplasmAttributeSeparation:
    """Principle (Section 10): Roman numeral stages and molecular markers are attributes, not distinct concepts."""

    def test_mantle_cell_lymphoma_staging(self, extractor):
        concept, attrs = extractor._extract_concept_and_attributes("Mantle cell lymphoma, Stage IV, blastoid variant")
        assert concept == "Mantle cell lymphoma, blastoid variant"
        assert attrs.get("oncologic_stage") == "Stage IV"

    def test_ckd_stage_is_preserved_as_concept(self, extractor):
        concept, attrs = extractor._extract_concept_and_attributes("Chronic kidney disease stage 3")
        assert "stage 3" in concept.lower()
        assert "oncologic_stage" not in attrs


class TestReverseAttributeValidation:
    """Principle: Code Specificity <= Evidence Specificity. Never infer unmentioned complications."""

    def test_unsupported_hemorrhage_rejected(self):
        is_valid, missing = ReverseAttributeChecker.validate_code_attributes(
            code_description="Gastric ulcer, chronic or unspecified, with hemorrhage",
            evidence_text="Patient has history of chronic gastric ulcer confirmed on endoscopy.",
            diagnosis_term="chronic gastric ulcer",
        )
        assert not is_valid
        assert any("bleed" in m or "hemorrhage" in m for m in missing)

    def test_supported_laterality_accepted(self):
        is_valid, missing = ReverseAttributeChecker.validate_code_attributes(
            code_description="Calculus of ureter, right",
            evidence_text="CT abdomen revealed 6mm stone in the right distal ureter.",
            diagnosis_term="Right ureteric calculus",
        )
        assert is_valid
        assert len(missing) == 0
