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
    PipelineStage,
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
        "validated_diagnoses": [
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


# ==============================================================================
# 5. Exact Concept Match & Nonessential Modifier Tests
# ==============================================================================


def test_exact_concept_match_over_first_good_enough_candidate():
    """Verify that the ranking and matching system chooses the BEST exact concept match
    rather than stopping at the first candidate that meets a threshold."""
    from medical_coding.reasoning.concept_matcher import (
        compute_concept_match_score,
        select_best_concept_match,
    )
    from medical_coding.schemas.icd import ICDCandidate

    # Candidate pool ordered with a generic 'good enough' match first
    candidates = [
        ICDCandidate(
            code="I50.9",
            description="Heart failure, unspecified",
            retrieval_score=0.85,
            is_valid_billable=True,
            coding_system="ICD-10-CM",
        ),
        ICDCandidate(
            code="I50.20",
            description="Unspecified systolic (congestive) heart failure",
            retrieval_score=0.75,
            is_valid_billable=True,
            coding_system="ICD-10-CM",
        ),
        ICDCandidate(
            code="I50.21",
            description="Acute systolic (congestive) heart failure",
            retrieval_score=0.60,  # Lower retrieval score, but EXACT concept match
            is_valid_billable=True,
            coding_system="ICD-10-CM",
        ),
    ]

    best_cand, best_score, best_rat = select_best_concept_match(
        candidates=candidates,
        concept_name="Acute systolic heart failure",
        evidence_text="Echo demonstrated EF 20% with acute decompensated heart failure.",
    )

    assert best_cand is not None
    assert best_cand.code == "I50.21", f"Expected exact concept match I50.21, but got {best_cand.code}"
    assert "exact" in best_rat.lower() or best_score >= 0.90


def test_strip_nonessential_modifiers_matches_canonical_concepts():
    """Verify ICD-10-CM nonessential modifiers in parentheses match documented concepts."""
    from medical_coding.reasoning.concept_matcher import (
        compute_concept_match_score,
        normalize_for_concept_matching,
        strip_nonessential_modifiers,
    )

    # I10 has nonessential modifier '(primary)'
    desc_i10 = "Essential (primary) hypertension"
    assert strip_nonessential_modifiers(desc_i10) == "Essential hypertension"
    score_i10, _, is_ex_i10 = compute_concept_match_score(
        candidate_code="I10",
        candidate_description=desc_i10,
        concept_name="Essential hypertension",
    )
    assert is_ex_i10 is True
    assert score_i10 >= 0.95

    # I50.21 has nonessential modifier '(congestive)'
    desc_i5021 = "Acute systolic (congestive) heart failure"
    assert strip_nonessential_modifiers(desc_i5021) == "Acute systolic heart failure"
    score_hf, _, is_ex_hf = compute_concept_match_score(
        candidate_code="I50.21",
        candidate_description=desc_i5021,
        concept_name="Acute systolic heart failure",
    )
    assert is_ex_hf is True
    assert score_hf >= 0.95


def test_concept_recall_preserves_compound_anatomical_phrases():
    """Verify compound phrases where 'and' joins anatomical adjectives do not fragment concepts."""
    from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor

    extractor = EvidenceFirstFactExtractor()
    doc = """
    DISCHARGE SUMMARY
    DISCHARGE DIAGNOSES:
    1. Acute cholecystitis (calculous)
    2. Hydronephrosis with renal and ureteral calculous obstruction
    """
    state = extractor.extract_clinical_state(doc, doc_id="test-recall-doc")
    terms = [c.raw_term.lower() for c in state.all_candidates]

    # Verify parenthetical clinical qualifier is preserved
    assert any("calculous" in t for t in terms), f"Expected calculous in terms: {terms}"

    # Verify 'renal and ureteral' was not split into a bare dangling 'renal' fragment
    assert not any(t.strip() == "renal" for t in terms), f"Bare 'renal' fragment found: {terms}"
    assert any("calculous obstruction" in t or "hydronephrosis" in t for t in terms)


# ==============================================================================
# 5. Semantic Mismatch, Polarity, UHDDS Priority & Audit Consistency Tests
# ==============================================================================


def test_neonatal_diabetes_blocked_for_adult():
    """Verify Chapter 16 perinatal code P70.2 is blocked for adult patient."""
    from medical_coding.reasoning.concept_matcher import compute_concept_match_score
    from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

    # P70.2 is strictly neonatal (age < 28 days)
    is_valid, unsupp = ReverseAttributeChecker.validate_code_attributes(
        code_description="Neonatal diabetes mellitus",
        evidence_text="45-year-old male with long-standing Type 2 diabetes mellitus admitted with acute fever.",
        diagnosis_term="Type 2 diabetes mellitus",
        code="P70.2",
    )
    assert is_valid is False
    assert any("perinatal" in u.lower() or "neonatal" in u.lower() for u in unsupp)

    score, rat, _ = compute_concept_match_score(
        candidate_code="P70.2",
        candidate_description="Neonatal diabetes mellitus",
        concept_name="Type 2 diabetes mellitus",
        evidence_text="45-year-old male with long-standing Type 2 diabetes mellitus admitted with acute fever.",
    )
    assert score == 0.0
    assert "disqualified" in rat.lower()


def test_hypertensive_emergency_blocked_for_hypotensive_patient():
    """Verify hypertensive emergency I16.1 is blocked when patient is hypotensive."""
    from medical_coding.reasoning.concept_matcher import compute_concept_match_score
    from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

    is_valid, unsupp = ReverseAttributeChecker.validate_code_attributes(
        code_description="Hypertensive emergency",
        evidence_text="Patient presented hypotensive with BP 75/45 requiring IV fluid boluses and vasopressors.",
        diagnosis_term="Hypotension",
        code="I16.1",
    )
    assert is_valid is False
    assert any("hypertension" in u.lower() or "polarity" in u.lower() or "emergency" in u.lower() for u in unsupp)

    score, rat, _ = compute_concept_match_score(
        candidate_code="I16.1",
        candidate_description="Hypertensive emergency",
        concept_name="Hypotension",
        evidence_text="Patient presented hypotensive with BP 75/45 requiring IV fluid boluses and vasopressors.",
    )
    assert score == 0.0
    assert "disqualified" in rat.lower()


def test_hepatitis_b_blocked_for_viral_fever():
    """Verify specific pathogen code B19.10 is blocked when evidence only documents generic viral fever."""
    from medical_coding.reasoning.concept_matcher import compute_concept_match_score
    from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

    is_valid, unsupp = ReverseAttributeChecker.validate_code_attributes(
        code_description="Unspecified viral hepatitis B without hepatic coma",
        evidence_text="Patient admitted for acute viral fever with myalgia and chills. LFTs normal.",
        diagnosis_term="Viral fever",
        code="B19.10",
    )
    assert is_valid is False
    assert any("pathogen" in u.lower() or "hepatitis" in u.lower() for u in unsupp)

    score, rat, _ = compute_concept_match_score(
        candidate_code="B19.10",
        candidate_description="Unspecified viral hepatitis B without hepatic coma",
        concept_name="Viral fever",
        evidence_text="Patient admitted for acute viral fever with myalgia and chills.",
    )
    assert score == 0.0
    assert "disqualified" in rat.lower()


def test_bladder_emptying_blocked_for_feverish_feeling():
    """Verify urinary bladder code R39.14 is blocked when derived from constitutional 'feverish feeling'."""
    from medical_coding.reasoning.concept_matcher import compute_concept_match_score
    from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

    is_valid, unsupp = ReverseAttributeChecker.validate_code_attributes(
        code_description="Feeling of incomplete bladder emptying",
        evidence_text="Patient complained of feverish feeling and generalized weakness.",
        diagnosis_term="feverish feeling",
        code="R39.14",
    )
    assert is_valid is False
    assert any("bladder" in u.lower() or "mismatch" in u.lower() for u in unsupp)

    score, rat, _ = compute_concept_match_score(
        candidate_code="R39.14",
        candidate_description="Feeling of incomplete bladder emptying",
        concept_name="feverish feeling",
        evidence_text="Patient complained of feverish feeling and generalized weakness.",
    )
    assert score == 0.0
    assert "disqualified" in rat.lower()


def test_shock_code_blocked_when_contradicted_by_evidence():
    """Verify R57.9 is blocked when evidence explicitly documents absence of shock."""
    from medical_coding.reasoning.concept_matcher import compute_concept_match_score
    from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

    is_valid, unsupp = ReverseAttributeChecker.validate_code_attributes(
        code_description="Shock, unspecified",
        evidence_text="Patient was normotensive without shock; hemodynamically stable on discharge.",
        diagnosis_term="Shock",
        code="R57.9",
    )
    assert is_valid is False
    assert any("shock" in u.lower() and "absent" in u.lower() for u in unsupp)

    score, rat, _ = compute_concept_match_score(
        candidate_code="R57.9",
        candidate_description="Shock, unspecified",
        concept_name="Shock",
        evidence_text="Patient was normotensive without shock; hemodynamically stable on discharge.",
    )
    assert score == 0.0
    assert "disqualified" in rat.lower()


def test_primary_diagnosis_selection_fever_hypotension_over_chronic_diabetes():
    """Verify UHDDS principle: acute admission driver (fever with hypotension) outranks chronic comorbidity (T2DM)."""
    classifier = PrimarySecondaryClassifier(llm=DummyLLM())
    doc = """
    DISCHARGE SUMMARY
    DIAGNOSES:
    1. Type 2 Diabetes Mellitus
    2. Acute Viral Fever with Hypotension
    HPI: Patient admitted with high-grade fever, chills, and acute hypotension (BP 75/40).
    PMH: Known Type 2 diabetes on metformin at baseline.
    """
    conditions = [
        ExtractedClinicalCondition(
            condition_id="c1",
            original_mention="Type 2 Diabetes Mellitus",
            normalized_description="Type 2 Diabetes Mellitus",
            evidence_text="Type 2 Diabetes Mellitus | Known Type 2 diabetes on metformin at baseline.",
            status=ConditionStatus.CHRONIC,
            section="DISCHARGE_DIAGNOSES",
        ),
        ExtractedClinicalCondition(
            condition_id="c2",
            original_mention="Acute Viral Fever with Hypotension",
            normalized_description="Acute Viral Fever with Hypotension",
            evidence_text="Patient admitted with high-grade fever, chills, and acute hypotension (BP 75/40).",
            status=ConditionStatus.ACUTE,
            section="CHIEF_COMPLAINT",
        ),
    ]

    res = classifier.classify_diagnoses(conditions, clinical_text=doc)
    assert res is not None
    primaries = [c for c in res if c.role == DiagnosisRole.PRIMARY]
    assert len(primaries) == 1
    # Acute admission driver must be PRIMARY under UHDDS
    assert "fever" in primaries[0].diagnosis.lower() or "hypotension" in primaries[0].diagnosis.lower()

    secondaries = [c for c in res if c.role == DiagnosisRole.SECONDARY]
    assert any("diabetes" in s.diagnosis.lower() for s in secondaries)


def test_concept_recall_captures_symptoms_and_findings():
    """Verify body ache, petechiae, and constipation are captured during full document extraction."""
    from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor

    extractor = EvidenceFirstFactExtractor()
    doc = """
    HOSPITAL DISCHARGE SUMMARY
    CHIEF COMPLAINT: High fever and generalized body ache for 3 days.
    PHYSICAL EXAMINATION: Petechiae noted on lower extremities.
    HOSPITAL COURSE: Patient developed constipation managed with laxatives.
    DISCHARGE DIAGNOSES:
    1. Acute viral illness with thrombocytopenia
    """
    state = extractor.extract_clinical_state(doc, doc_id="test-findings-recall")
    all_terms = [c.raw_term.lower() for c in state.all_candidates]

    assert any("body ache" in t for t in all_terms), f"Expected 'body ache' in {all_terms}"
    assert any("petechiae" in t for t in all_terms), f"Expected 'petechiae' in {all_terms}"
    assert any("constipation" in t for t in all_terms), f"Expected 'constipation' in {all_terms}"


def test_output_audit_consistency_abstained_candidate_never_output():
    """Verify an abstained candidate is strictly preserved in abstentions and never resurrected in final report."""
    from medical_coding.schemas.validation import AbstentionRecord
    from medical_coding.schemas.icd import RankedSelection

    state = {
        "document_id": "test-audit-consistency",
        "validated_diagnoses": [],
        "abstentions": [
            AbstentionRecord(
                diagnosis_id="diag-dm",
                raw_term="Type 2 diabetes mellitus",
                reason=AbstentionReason.UNSUPPORTED_SPECIFICITY,
                detail="P70.2 failed clinical semantic validation: demographic mismatch",
                stage=PipelineStage.VALIDATION,
            )
        ],
        "audit_trail": [],
        "classified_diagnoses": [
            ExtractedClinicalCondition(
                condition_id="diag-dm",
                original_mention="Type 2 diabetes mellitus",
                normalized_description="Type 2 diabetes mellitus",
                evidence_text="45-year-old adult with Type 2 diabetes",
                section="DISCHARGE_DIAGNOSES",
                role=DiagnosisRole.PRIMARY,
            )
        ],
        "ranked_selections": [
            RankedSelection(
                diagnosis_id="diag-dm",
                decision="ABSTAINED",
                ranking_reason="Demographic mismatch",
                confidence=0.0,
            )
        ],
    }

    result = finalize_output_node(state)
    report_dict = result.get("final_report", {})
    report_abstentions = report_dict.get("abstentions", [])

    # The abstained candidate MUST be present in the report's abstentions
    assert any(a.get("raw_term") == "Type 2 diabetes mellitus" for a in report_abstentions)

    # The abstained candidate must NOT be in primary_diagnosis or secondary_diagnoses as an accepted code
    primary_dx = report_dict.get("primary_diagnosis")
    if primary_dx:
        assert primary_dx.get("database_code") != "P70.2", "P70.2 must not be output as primary code!"
        assert primary_dx.get("raw_term") != "Type 2 diabetes mellitus", "Abstained diabetes must not be output as accepted primary!"


def test_nasal_mucositis_blocked_for_oral_mucositis():
    """Verify that candidate code J34.81 (Nasal mucositis) is disqualified for oral mucositis / general mucositis."""
    from medical_coding.reasoning.concept_matcher import compute_concept_match_score
    from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

    # Oral mucositis -> J34.81 (nasal) must fail reverse attribute check
    is_valid, unsupp = ReverseAttributeChecker.validate_code_attributes(
        code_description="Nasal mucositis (ulcerative)",
        evidence_text="Patient developed severe oral mucositis and stomatitis.",
        diagnosis_term="Oral mucositis",
        code="J34.81",
    )
    assert is_valid is False
    assert any("mismatch" in u.lower() or "nasal" in u.lower() or "site" in u.lower() for u in unsupp)

    # Concept score must be disqualified
    score, rat, _ = compute_concept_match_score(
        candidate_code="J34.81",
        candidate_description="Nasal mucositis (ulcerative)",
        concept_name="Oral mucositis",
        evidence_text="Patient developed severe oral mucositis and stomatitis.",
    )
    assert score == 0.0

    # Oral mucositis -> K12.30 (Oral mucositis) must succeed
    is_valid_oral, _ = ReverseAttributeChecker.validate_code_attributes(
        code_description="Oral mucositis (ulcerative), unspecified",
        evidence_text="Patient developed oral mucositis.",
        diagnosis_term="Oral mucositis",
        code="K12.30",
    )
    assert is_valid_oral is True


def test_compound_with_splitting_hypotension_and_dehydration():
    """Verify compound phrase 'Acute viral fever with hypotension and dehydration' decomposes into independent concepts."""
    from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor

    extractor = EvidenceFirstFactExtractor()
    doc = """
    DISCHARGE SUMMARY
    DIAGNOSES:
    1. Acute viral fever with hypotension and dehydration
    HPI: Admitted with high fever, BP 75/45, and marked dehydration.
    """
    state = extractor.extract_clinical_state(doc, doc_id="test-compound-with")
    extracted_terms = [c.raw_term.lower() for c in state.all_candidates]

    assert any("viral fever" in t or "viral" in t for t in extracted_terms)
    assert any("hypotension" in t for t in extracted_terms)
    assert any("dehydration" in t for t in extracted_terms)


def test_contradictory_shock_resolved_to_ruled_out():
    """Verify that when a heading mentions shock but examination/course shows stability, shock is ruled out and NOT confirmed."""
    from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor

    extractor = EvidenceFirstFactExtractor()
    doc = """
    DISCHARGE SUMMARY
    IMPRESSION: Shock
    EXAMINATION: Patient is alert, warm, normotensive without shock (BP 120/80). Hemodynamically stable.
    HOSPITAL COURSE: Fluid challenge was completed; no evidence of septic shock.
    FINAL DIAGNOSIS: Acute viral fever with dehydration.
    """
    state = extractor.extract_clinical_state(doc, doc_id="test-shock-contradiction")

    # Shock must NOT be in primary_diagnosis or secondary_diagnoses as active/confirmed
    if state.primary_diagnosis:
        assert "shock" not in state.primary_diagnosis.raw_term.lower()
    for sec in state.secondary_diagnoses:
        assert "shock" not in sec.raw_term.lower()

    # Shock should be in ruled_out_conditions
    assert any("shock" in r.raw_term.lower() for r in state.ruled_out_conditions)


def test_ten_concept_catalogue_recall_matrix():
    """Verify strict concept-and-code matching reaches 90-100% across the 10 catalogue-addressable concepts."""
    from medical_coding.reasoning.concept_matcher import compute_concept_match_score

    # Benchmark: 10 catalogue-addressable clinical concepts and their defensible ICD-10-CM codes
    benchmark_cases = [
        # (Concept, Document Evidence, Expected Code, Expected Code Desc, False Positive Code to Reject)
        (
            "Hypotension",
            "Patient was hypotensive with BP 75/45 requiring IV fluid boluses.",
            "I95.9",
            "Hypotension, unspecified",
            "I16.1",  # Hypertensive emergency
        ),
        (
            "Acute viral fever",
            "Patient admitted for acute viral fever with generalized weakness.",
            "B34.9",
            "Viral infection, unspecified",
            "B19.10",  # Viral hepatitis B
        ),
        (
            "Type 2 diabetes mellitus",
            "52-year-old adult patient with baseline Type 2 diabetes on metformin.",
            "E11.9",
            "Type 2 diabetes mellitus without complications",
            "P70.2",  # Neonatal diabetes
        ),
        (
            "Essential hypertension",
            "Past history of essential hypertension controlled on amlodipine.",
            "I10",
            "Essential (primary) hypertension",
            "I16.1",  # Hypertensive emergency
        ),
        (
            "Oral mucositis",
            "Examination revealed oral mucositis and stomatitis of the buccal mucosa.",
            "K12.30",
            "Oral mucositis (ulcerative), unspecified",
            "J34.81",  # Nasal mucositis
        ),
        (
            "Myalgia",
            "Patient complained of severe generalized body aches and myalgia.",
            "M79.10",
            "Myalgia, unspecified site",
            "R39.14",  # Incomplete bladder emptying
        ),
        (
            "Petechiae",
            "Physical examination noted petechiae on lower extremities.",
            "R23.3",
            "Spontaneous ecchymoses",
            "Z33.1",  # Incidental pregnancy
        ),
        (
            "Constipation",
            "Patient developed constipation during hospital stay managed with laxatives.",
            "K59.00",
            "Constipation, unspecified",
            "R57.9",  # Unspecified shock
        ),
        (
            "Dehydration",
            "Patient presented with volume depletion and dehydration.",
            "E86.0",
            "Dehydration",
            "I16.1",
        ),
        (
            "Fever",
            "Patient presented with high-grade fever of 103 F.",
            "R50.9",
            "Fever, unspecified",
            "R39.14",  # Bladder emptying
        ),
    ]

    correct_matches = 0
    total_targets = len(benchmark_cases)

    for concept, ev, expected_code, expected_desc, rejected_code in benchmark_cases:
        # 1. Expected code must score high (best concept match)
        score_exp, rat_exp, _ = compute_concept_match_score(
            candidate_code=expected_code,
            candidate_description=expected_desc,
            concept_name=concept,
            evidence_text=ev,
        )
        assert score_exp >= 0.80, f"Expected code {expected_code} scored too low ({score_exp}): {rat_exp}"

        # 2. Rejected false positive must score 0.0 (disqualified by clinical gates)
        score_rej, rat_rej, _ = compute_concept_match_score(
            candidate_code=rejected_code,
            candidate_description="False positive candidate description",
            concept_name=concept,
            evidence_text=ev,
        )
        assert score_rej == 0.0 or score_rej < score_exp, f"Rejected code {rejected_code} not disqualified: {rat_rej}"

        if score_exp > score_rej and score_exp >= 0.80:
            correct_matches += 1

    strict_recall = (correct_matches / total_targets) * 100.0
    print(f"\nBenchmark Strict Concept-and-Code Recall: {strict_recall:.1f}% ({correct_matches}/{total_targets})")
    assert strict_recall >= 90.0, f"Strict recall {strict_recall}% must be >= 90%!"


# ==============================================================================
# 5. Clinical Semantic Validation & Contradiction Handling Regression Tests
# ==============================================================================


def test_concept_reasoner_detects_shock_contradiction():
    """Verify ClinicalConceptReasoner populates contradictory_attributes and sets RULED_OUT
    when provisional shock is contradicted by hemodynamic stability or normotension."""
    from medical_coding.reasoning.concept_reasoner import ClinicalConceptReasoner
    from medical_coding.schemas.clinical import ExtractedClinicalCondition, EvidenceLocation

    candidate = ExtractedClinicalCondition(
        condition_id="c-shock-1",
        raw_term="Shock",
        normalized_diagnosis="Shock",
        evidence_quote="IMPRESSION: Shock",
        evidence_location=EvidenceLocation(section="IMPRESSION"),
        certainty=Certainty.CONFIRMED,
    )
    doc_text = (
        "IMPRESSION: Shock.\n"
        "PHYSICAL EXAMINATION: Patient is warm, alert, normotensive without shock (BP 125/82).\n"
        "HOSPITAL COURSE: Hemodynamically stable throughout."
    )

    concept = ClinicalConceptReasoner.reason_concept(
        candidate=candidate,
        evidence_text="IMPRESSION: Shock",
        document_text=doc_text,
    )

    assert concept.assertion_status == "RULED_OUT"
    assert "hemodynamic_stability_contradiction" in concept.contradictory_attributes
    assert len(concept.contradiction_evidence) > 0


def test_compatibility_reasoner_rejects_contradicted_concept():
    """Verify CompatibilityReasoner returns compatible=False and score=0.0 when concept has contradictions."""
    from medical_coding.reasoning.concept_reasoner import ClinicalConceptReasoner
    from medical_coding.reasoning.compatibility import CompatibilityReasoner
    from medical_coding.schemas.clinical import ExtractedClinicalCondition, EvidenceLocation

    candidate = ExtractedClinicalCondition(
        condition_id="c-shock-2",
        raw_term="Septic shock",
        normalized_diagnosis="Septic shock",
        evidence_quote="Impression: Septic shock",
        evidence_location=EvidenceLocation(section="IMPRESSION"),
        certainty=Certainty.CONFIRMED,
    )
    doc_text = "Impression: Septic shock. Examination shows patient is normotensive without shock and hemodynamically stable."

    concept = ClinicalConceptReasoner.reason_concept(
        candidate=candidate,
        evidence_text="Impression: Septic shock",
        document_text=doc_text,
    )

    result = CompatibilityReasoner.evaluate_candidate(
        candidate_code="R57.2",
        candidate_description="Septic shock",
        concept=concept,
    )

    assert result.compatible is False
    assert result.score == 0.0
    assert len(result.contradictions) > 0


def test_semantic_validation_rejects_and_realigns_mismatches():
    """Verify CandidateRankingDeterministicValidator rejects anatomical and physiological mismatches
    and realigns to supported candidates from the candidate pool."""
    from medical_coding.validation.deterministic import CandidateRankingDeterministicValidator
    from medical_coding.schemas.icd import ICDCandidate, RankedSelection

    validator = CandidateRankingDeterministicValidator()

    # Case 1: Oral mucositis with candidate pool containing J34.81 (nasal) and K12.30 (oral)
    cand_nasal = ICDCandidate(
        code="J34.81",
        description="Nasal mucositis (ulcerative)",
        retrieval_score=0.95,
        is_valid_billable=True,
    )
    cand_oral = ICDCandidate(
        code="K12.30",
        description="Oral mucositis (ulcerative), unspecified",
        retrieval_score=0.88,
        is_valid_billable=True,
    )

    initial_sel = RankedSelection(
        diagnosis_id="diag-mucositis",
        raw_term="Oral mucositis",
        selected_code="J34.81",
        selected_description=cand_nasal.description,
        selected_candidate=cand_nasal,
        candidate_pool=[cand_nasal, cand_oral],
    )

    validated_sel = validator.validate_ranking_selection(
        selection=initial_sel,
        candidates=[cand_nasal, cand_oral],
        evidence_text="Patient has severe oral mucositis and stomatitis of buccal mucosa.",
        diagnosis_term="Oral mucositis",
    )

    # Must be realigned to K12.30 (oral), NOT J34.81 (nasal)
    assert validated_sel.selected_code == "K12.30"
    assert "oral" in validated_sel.selected_description.lower()




