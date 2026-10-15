"""Unit and regression tests for clinical normalization, parenthetical preservation,
admission header matching, and secondary comorbidity non-promotion guardrails.
"""

import pytest
from medical_coding.schemas.clinical import (
    ConditionStatus,
    ContextAssessment,
    ExtractedClinicalCondition,
)
from medical_coding.schemas.enums import (
    AbstentionReason,
    Acuity,
    Certainty,
    DiagnosisRole,
    Laterality,
    NegationStatus,
    Temporality,
)
from medical_coding.schemas.response import CodedDiagnosisResponse, CodingResult
from medical_coding.utils.text import (
    _is_administrative_parenthetical,
    normalize_diagnosis_name,
    extract_search_query_variations,
)
from medical_coding.utils.clinical_lexicon import CLINICAL_ABBREVIATIONS, CLINICAL_MORPHOLOGY
from medical_coding.agents.classifier import PrimarySecondaryClassifier
from medical_coding.models.base import BaseLocalLLM
from medical_coding.graph.nodes import finalize_output_node


class DummyLLM(BaseLocalLLM):
    def generate(self, prompt: str) -> str:
        return ""

    def get_model_info(self) -> dict:
        return {"model_name": "dummy"}


# ==============================================================================
# 1. Normalization & Parenthetical Preservation Tests
# ==============================================================================


def test_administrative_parenthetical_detection():
    """Verify administrative artifacts (dates, codes, MRN) are detected for stripping,
    while critical clinical qualifiers are preserved."""
    # Administrative artifacts -> True (strip)
    assert _is_administrative_parenthetical("12/05/2023") is True
    assert _is_administrative_parenthetical("2022-04-10") is True
    assert _is_administrative_parenthetical("15 jan 2024") is True
    assert _is_administrative_parenthetical("ICD-10: I10") is True
    assert _is_administrative_parenthetical("icd: e11.9") is True
    assert _is_administrative_parenthetical("code C61") is True
    assert _is_administrative_parenthetical("I10") is True
    assert _is_administrative_parenthetical("E11.9") is True
    assert _is_administrative_parenthetical("K80.20") is True
    assert _is_administrative_parenthetical("MRN: 12345") is True
    assert _is_administrative_parenthetical("1") is True
    assert _is_administrative_parenthetical("iv") is True

    # Clinical qualifiers -> False (preserve!)
    assert _is_administrative_parenthetical("HFrEF") is False
    assert _is_administrative_parenthetical("stage 3b") is False
    assert _is_administrative_parenthetical("calculous") is False
    assert _is_administrative_parenthetical("paroxysmal") is False
    assert _is_administrative_parenthetical("lumbar spine, pelvis") is False
    assert _is_administrative_parenthetical("Gleason 4+4=8") is False
    assert _is_administrative_parenthetical("mild intermittent") is False
    assert _is_administrative_parenthetical("gangrenous") is False


def test_normalize_diagnosis_name_preserves_clinical_parentheticals():
    """Verify that clinical parentheticals are unwrapped cleanly into the diagnosis name
    rather than destroyed, and administrative parentheticals are stripped."""
    # Clinical parentheticals preserved
    assert "calculous" in normalize_diagnosis_name("Acute cholecystitis (calculous)").lower()
    assert "stage 3b" in normalize_diagnosis_name("Chronic kidney disease (stage 3b)").lower()
    assert "paroxysmal" in normalize_diagnosis_name("Atrial fibrillation (paroxysmal)").lower()
    assert "lumbar spine" in normalize_diagnosis_name("Metastatic carcinoma (lumbar spine, pelvis)").lower()
    assert "gleason" in normalize_diagnosis_name("Adenocarcinoma of prostate (Gleason 4+4=8)").lower()

    # Administrative parentheticals stripped
    assert "i10" not in normalize_diagnosis_name("Essential hypertension (ICD-10: I10)").lower()
    assert "2023" not in normalize_diagnosis_name("Type 2 diabetes (diagnosed 12/04/2023)").lower()
    assert "mrn" not in normalize_diagnosis_name("Hypertension (MRN: 98765)").lower()


def test_clean_abbreviation_expansions():
    """Verify abbreviations expand to standard clinical phrases without synonym soups."""
    # CVA should be cerebrovascular accident, not concatenated with stroke
    cva_exp = CLINICAL_ABBREVIATIONS.get("cva")
    assert cva_exp == "cerebrovascular accident"

    # SOB should be shortness of breath, not concatenated with dyspnea
    sob_exp = CLINICAL_ABBREVIATIONS.get("sob")
    assert sob_exp == "shortness of breath"

    # choledocholithiasis is not an abbreviation; must not be mapped to synonym soup in CLINICAL_ABBREVIATIONS
    assert "choledocholithiasis" not in CLINICAL_ABBREVIATIONS

    # choledocholithiasis has appropriate morphological synonyms
    assert "choledocholithiasis" in CLINICAL_MORPHOLOGY
    assert any("calculus" in syn for syn in CLINICAL_MORPHOLOGY["choledocholithiasis"])


# ==============================================================================
# 2. Primary Diagnosis Classification & Admission Driver Tests
# ==============================================================================


def test_classifier_identifies_admitting_diagnosis_header():
    """Verify ADMITTING DIAGNOSIS header is recognized and outranks chronic comorbidities."""
    classifier = PrimarySecondaryClassifier(llm=DummyLLM())

    note = (
        "ADMITTING DIAGNOSIS: Ca prostate\n"
        "PAST MEDICAL HISTORY: Essential hypertension on amlodipine, type 2 diabetes.\n"
        "HOSPITAL COURSE: Patient underwent robotic prostatectomy. Home amlodipine continued.\n"
        "DISCHARGE DIAGNOSES: Ca prostate, essential hypertension, type 2 diabetes."
    )

    assessments = [
        ContextAssessment(
            diagnosis="Malignant neoplasm of prostate",
            condition_id="c-prostate",
            section="DISCHARGE_DIAGNOSES",
            current_relevance=True,
            coding_candidate=True,
            status=ConditionStatus.ACTIVE,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            negation=NegationStatus.AFFIRMATIVE,
            evidence="ADMITTING DIAGNOSIS: Ca prostate | robotic prostatectomy",
            reason="Malignant prostate neoplasm requiring surgical resection",
            treated_or_managed=True,
            original_mention="Ca prostate",
        ),
        ContextAssessment(
            diagnosis="Essential hypertension",
            condition_id="c-htn",
            section="PAST_MEDICAL_HISTORY",
            current_relevance=True,
            coding_candidate=True,
            status=ConditionStatus.CHRONIC,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            negation=NegationStatus.AFFIRMATIVE,
            evidence="PAST MEDICAL HISTORY: Essential hypertension on amlodipine",
            reason="Chronic baseline comorbidity continued on home medication",
            treated_or_managed=True,
            original_mention="Essential hypertension",
        ),
        ContextAssessment(
            diagnosis="Type 2 diabetes mellitus",
            condition_id="c-dm",
            section="PAST_MEDICAL_HISTORY",
            current_relevance=True,
            coding_candidate=True,
            status=ConditionStatus.CHRONIC,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            negation=NegationStatus.AFFIRMATIVE,
            evidence="type 2 diabetes",
            reason="Chronic baseline comorbidity",
            treated_or_managed=False,
            original_mention="type 2 diabetes",
        ),
    ]

    res, _ = classifier.classify_conditions(assessments, clinical_text=note)
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.diagnosis == "Malignant neoplasm of prostate"
    assert res.primary_diagnosis.role == DiagnosisRole.PRIMARY

    # Secondary diagnoses must NOT be primary
    sec_names = [d.diagnosis for d in res.secondary_diagnoses]
    assert "Essential hypertension" in sec_names
    assert all(d.role == DiagnosisRole.SECONDARY for d in res.secondary_diagnoses)


def test_classifier_etiological_primary_guardrail():
    """Verify 'Primary hypertension' in condition name does NOT trigger primary role
    when patient was admitted for an acute admission driver (e.g. Pneumonia)."""
    classifier = PrimarySecondaryClassifier(llm=DummyLLM())

    note = (
        "CHIEF COMPLAINT: Severe cough and fever.\n"
        "REASON FOR ADMISSION: Acute community acquired pneumonia.\n"
        "PAST MEDICAL HISTORY: Primary hypertension, well controlled.\n"
        "HOSPITAL COURSE: Started on IV ceftriaxone for pneumonia."
    )

    assessments = [
        ContextAssessment(
            diagnosis="Pneumonia",
            condition_id="c-pna",
            section="REASON_FOR_ADMISSION",
            current_relevance=True,
            coding_candidate=True,
            status=ConditionStatus.ACUTE,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            negation=NegationStatus.AFFIRMATIVE,
            evidence="REASON FOR ADMISSION: Acute community acquired pneumonia",
            reason="Acute occasioning condition",
            treated_or_managed=True,
            original_mention="Acute community acquired pneumonia",
        ),
        ContextAssessment(
            diagnosis="Primary hypertension",
            condition_id="c-phtn",
            section="PAST_MEDICAL_HISTORY",
            current_relevance=True,
            coding_candidate=True,
            status=ConditionStatus.CHRONIC,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            negation=NegationStatus.AFFIRMATIVE,
            evidence="PAST MEDICAL HISTORY: Primary hypertension, well controlled",
            reason="Comorbidity",
            treated_or_managed=False,
            original_mention="Primary hypertension",
        ),
    ]

    res, _ = classifier.classify_conditions(assessments, clinical_text=note)
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.diagnosis == "Pneumonia"
    assert res.primary_diagnosis.role == DiagnosisRole.PRIMARY

    htn_diag = next(d for d in res.secondary_diagnoses if "hypertension" in d.diagnosis.lower())
    assert htn_diag.role == DiagnosisRole.SECONDARY


def test_initial_role_preserved_from_clinical_extractor():
    """Verify that when initial_role=DiagnosisRole.PRIMARY is set on ContextAssessment,
    the classifier honors the clinically established primary driver."""
    classifier = PrimarySecondaryClassifier(llm=DummyLLM())

    assessments = [
        ContextAssessment(
            diagnosis="Acute cholecystitis",
            condition_id="c-chole",
            section="DISCHARGE_DIAGNOSES",
            current_relevance=True,
            coding_candidate=True,
            status=ConditionStatus.ACUTE,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            negation=NegationStatus.AFFIRMATIVE,
            evidence="Acute cholecystitis occasioning laparoscopic cholecystectomy",
            reason="Occasioning admission",
            treated_or_managed=True,
            initial_role=DiagnosisRole.PRIMARY,
        ),
        ContextAssessment(
            diagnosis="Essential hypertension",
            condition_id="c-htn",
            section="PAST_MEDICAL_HISTORY",
            current_relevance=True,
            coding_candidate=True,
            status=ConditionStatus.CHRONIC,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.CURRENT,
            negation=NegationStatus.AFFIRMATIVE,
            evidence="Hypertension on lisinopril",
            reason="Comorbidity",
            treated_or_managed=True,
            initial_role=DiagnosisRole.SECONDARY,
        ),
    ]

    res, _ = classifier.classify_conditions(assessments, clinical_text="")
    assert res.primary_diagnosis is not None
    assert res.primary_diagnosis.diagnosis == "Acute cholecystitis"
    assert res.primary_diagnosis.role == DiagnosisRole.PRIMARY


# ==============================================================================
# 3. Finalize Output & Secondary Non-Promotion Guardrails
# ==============================================================================


def test_no_arbitrary_secondary_promotion_in_finalize_output():
    """Verify that when the clinical primary diagnosis cannot be coded (or abstains),
    the secondary diagnosis (e.g. I10 Hypertension) is strictly NOT promoted to primary."""
    state = {
        "document_id": "test-doc-001",
        "validated_codes": [
            # Only the secondary condition could be validated
            type("MockCoded", (), {
                "raw_term": "Essential hypertension",
                "code": "I10",
                "description": "Essential (primary) hypertension",
                "role": DiagnosisRole.SECONDARY,
                "confidence_score": 0.95,
                "icd10cm": "I10",
                "icdo": None,
                "cpt": None,
                "source_section": "PAST_MEDICAL_HISTORY",
                "source_span": None,
                "matching_status": "MATCHED",
                "evidence": type("MockEv", (), {"quote": "Hypertension on medication", "source_section": "PAST_MEDICAL_HISTORY", "source_span": None})(),
            })(),
        ],
        "classified_diagnoses": [
            # A primary condition existed clinically, but had no verified code
            type("MockClassified", (), {
                "diagnosis_id": "c-unmapped-primary",
                "diagnosis": "Rare obscure primary disease syndrome",
                "raw_term": "Rare obscure primary disease syndrome",
                "normalized_diagnosis": "Rare obscure primary disease syndrome",
                "role": DiagnosisRole.PRIMARY,
                "evidence_quote": "Admitted for rare obscure primary disease syndrome",
                "source_section": "PRINCIPAL_DIAGNOSIS",
            })(),
            type("MockClassified", (), {
                "diagnosis_id": "c-htn",
                "diagnosis": "Essential hypertension",
                "raw_term": "Essential hypertension",
                "normalized_diagnosis": "Essential hypertension",
                "role": DiagnosisRole.SECONDARY,
                "evidence_quote": "Hypertension on medication",
                "source_section": "PAST_MEDICAL_HISTORY",
            })(),
        ],
        "abstentions": [],
    }

    output = finalize_output_node(state)
    result: CodingResult = output["final_result"]

    # CRITICAL: Secondary diagnosis I10 MUST NOT be promoted to primary!
    # If primary had no valid code match, secondary must remain SECONDARY.
    assert all(d.database_code != "I10" or d.role == DiagnosisRole.SECONDARY for d in result.secondary_diagnoses)
    if result.primary_diagnosis:
        assert result.primary_diagnosis.database_code != "I10"
        assert result.primary_diagnosis.role == DiagnosisRole.PRIMARY
    else:
        # If primary abstained, primary_diagnosis is None and abstention is recorded
        assert any("obscure" in a.detail.lower() or "obscure" in (a.raw_term or "").lower() for a in result.abstentions)
        assert len(result.secondary_diagnoses) == 1
        assert result.secondary_diagnoses[0].database_code == "I10"
        assert result.secondary_diagnoses[0].role == DiagnosisRole.SECONDARY
