"""Unit tests for the Primary / Secondary Classification component and deterministic validator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from medical_coding.agents.classifier import PrimarySecondaryClassifier
from medical_coding.schemas.clinical import (
    ConditionClassification,
    ContextAssessment,
    EncounterClassificationResult,
)
from medical_coding.schemas.enums import (
    AbstentionReason,
    Certainty,
    ConditionStatus,
    DiagnosisRole,
    NegationStatus,
    PipelineStage,
    Temporality,
)
from medical_coding.schemas.state import PipelineGraphState
from medical_coding.validation.deterministic import (
    MultiplePrimaryDiagnosesError,
    PrimarySecondaryClassificationValidator,
)

# ============================================================================
# Test Cases 1 - 6: Required Clinical Scenarios
# ============================================================================


def test_scenario_1_clear_single_primary() -> None:
    """Requirement 1: Clear single primary diagnosis.

    Patient admitted for acute appendicitis, underwent laparoscopic appendectomy.
    """
    mock_llm = MagicMock()
    classifier = PrimarySecondaryClassifier(mock_llm)

    assessment = ContextAssessment(
        condition_id="app-01",
        diagnosis="Acute appendicitis",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Patient presented with acute right lower quadrant pain, admitted for acute appendicitis. Laparoscopic appendectomy performed.",
        reason="Acute condition occasioning admission treated with operative intervention.",
        treated_or_managed=True,
        monitored=True,
        affected_clinical_management=True,
        influenced_treatment=True,
        treatment_evidence="Laparoscopic appendectomy performed",
    )

    result, abstentions = classifier.classify_conditions([assessment])

    assert result.has_unique_primary is True
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.diagnosis == "Acute appendicitis"
    assert result.primary_diagnosis.role == DiagnosisRole.PRIMARY
    assert result.primary_diagnosis.is_billable_candidate is True
    assert len(result.secondary_diagnoses) == 0
    assert len(result.excluded_conditions) == 0
    assert result.is_ambiguous_primary is False
    assert len(abstentions) == 0


def test_scenario_2_multiple_diagnoses_not_first_mentioned() -> None:
    """Requirement 2: Multiple diagnoses, ensuring the first-mentioned condition is NOT automatically primary.

    Document mentions Essential hypertension first under PMH, but patient was admitted for
    Acute NSTEMI with coronary catheterization and stent placement, and also had Type 2 diabetes managed.
    """
    mock_llm = MagicMock()
    classifier = PrimarySecondaryClassifier(mock_llm)

    htn = ContextAssessment(
        condition_id="htn-01",
        diagnosis="Essential hypertension",
        current_relevance=False,
        coding_candidate=False,
        status=ConditionStatus.HISTORICAL,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.HISTORICAL,
        evidence="PAST MEDICAL HISTORY: Essential hypertension for 10 years.",
        reason="Historical PMH condition without active inpatient care.",
    )

    nstemi = ContextAssessment(
        condition_id="nstemi-01",
        diagnosis="Acute non-ST elevation myocardial infarction",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="CHIEF COMPLAINT: Admitted for acute NSTEMI with peak troponin 5.2 ng/mL. Emergent coronary catheterization with drug-eluting stent placed.",
        reason="Acute cardiac infarction occasioning admission and catheterization with stenting.",
        treated_or_managed=True,
        monitored=True,
        affected_clinical_management=True,
        influenced_treatment=True,
        treatment_evidence="Coronary catheterization with drug-eluting stent placed",
    )

    dm = ContextAssessment(
        condition_id="dm-01",
        diagnosis="Type 2 diabetes mellitus",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.CHRONIC,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Type 2 diabetes monitored with blood glucose checks, insulin sliding scale adjusted.",
        reason="Pre-existing condition actively monitored and managed during stay.",
        treated_or_managed=True,
        monitored=True,
        treatment_evidence="insulin sliding scale adjusted",
    )

    # Note: HTN is listed first in document order
    result, _ = classifier.classify_conditions([htn, nstemi, dm])

    assert result.has_unique_primary is True
    assert result.primary_diagnosis is not None
    # Must NOT select HTN (the first mentioned)!
    assert result.primary_diagnosis.diagnosis == "Acute non-ST elevation myocardial infarction"
    assert result.primary_diagnosis.role == DiagnosisRole.PRIMARY

    # DM is secondary
    sec_names = [s.diagnosis for s in result.secondary_diagnoses]
    assert "Type 2 diabetes mellitus" in sec_names

    # HTN is excluded
    excl_names = [e.diagnosis for e in result.excluded_conditions]
    assert "Essential hypertension" in excl_names


def test_scenario_3_historical_condition_excluded() -> None:
    """Requirement 3: Historical-only conditions without current relevance must be excluded from billing."""
    mock_llm = MagicMock()
    classifier = PrimarySecondaryClassifier(mock_llm)

    gerd = ContextAssessment(
        condition_id="gerd-01",
        diagnosis="Gastroesophageal reflux disease",
        current_relevance=False,
        coding_candidate=False,
        status=ConditionStatus.HISTORICAL,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.HISTORICAL,
        evidence="Past history of GERD diagnosed 8 years ago, no active symptoms.",
        reason="Historical condition without active evaluation or therapy.",
    )

    result, abstentions = classifier.classify_conditions([gerd])

    assert result.primary_diagnosis is None
    assert len(result.secondary_diagnoses) == 0
    assert len(result.excluded_conditions) == 1
    assert result.excluded_conditions[0].diagnosis == "Gastroesophageal reflux disease"
    assert result.excluded_conditions[0].role == DiagnosisRole.EXCLUDED
    assert result.excluded_conditions[0].is_billable_candidate is False


def test_scenario_4_two_plausible_primary_conditions_abstains() -> None:
    """Requirement 4: Two plausible primary conditions where documentation does not distinguish which occasioned admission.

    Patient presented with simultaneous Acute myocardial infarction AND Acute ischemic stroke,
    both listed as reasons for admission, both receiving emergent acute interventions.
    The agent must NOT arbitrarily choose one, and must return an ambiguity/abstention state.
    """
    mock_llm = MagicMock()
    classifier = PrimarySecondaryClassifier(mock_llm)

    ami = ContextAssessment(
        condition_id="ami-01",
        diagnosis="Acute myocardial infarction",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Reason for admission: Acute myocardial infarction, emergent catheterization and stent placed.",
        reason="Acute condition occasioning emergent admission.",
        treated_or_managed=True,
        monitored=True,
        affected_clinical_management=True,
        treatment_evidence="emergent catheterization and stent placed",
    )

    stroke = ContextAssessment(
        condition_id="cva-01",
        diagnosis="Acute ischemic stroke",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Reason for admission: Acute ischemic stroke, emergent mechanical thrombectomy performed.",
        reason="Acute condition occasioning emergent admission.",
        treated_or_managed=True,
        monitored=True,
        affected_clinical_management=True,
        treatment_evidence="emergent mechanical thrombectomy performed",
    )

    result, abstentions = classifier.classify_conditions([ami, stroke])

    # Must NOT arbitrarily choose one!
    assert result.has_unique_primary is False
    assert result.primary_diagnosis is None
    assert result.is_ambiguous_primary is True
    assert result.abstention_recommended is True
    assert result.abstention_reason == AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY

    # Check abstention record was emitted
    assert len(abstentions) > 0
    assert any(a.reason == AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY for a in abstentions)


def test_scenario_5_no_defensible_primary() -> None:
    """Requirement 5: No defensible primary diagnosis.

    All conditions in record are historical or ruled out; no condition qualifies as reason for admission.
    Agent must not force a primary diagnosis.
    """
    mock_llm = MagicMock()
    classifier = PrimarySecondaryClassifier(mock_llm)

    ruled_out_pe = ContextAssessment(
        condition_id="pe-01",
        diagnosis="Pulmonary embolism",
        current_relevance=False,
        coding_candidate=False,
        status=ConditionStatus.RESOLVED,
        certainty=Certainty.RULED_OUT,
        temporality=Temporality.CURRENT,
        negation=NegationStatus.NEGATED,
        evidence="Suspected pulmonary embolism ruled out by negative CT angiogram.",
        reason="Definitively ruled out by diagnostic imaging.",
    )

    hist_lipid = ContextAssessment(
        condition_id="lipid-01",
        diagnosis="Hyperlipidemia",
        current_relevance=False,
        coding_candidate=False,
        status=ConditionStatus.HISTORICAL,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.HISTORICAL,
        evidence="Past history of hyperlipidemia.",
        reason="Historical condition without active management.",
    )

    result, _ = classifier.classify_conditions([ruled_out_pe, hist_lipid])

    assert result.primary_diagnosis is None
    assert result.has_unique_primary is False
    assert len(result.secondary_diagnoses) == 0
    assert len(result.excluded_conditions) == 2


def test_scenario_6_primary_plus_secondary_conditions() -> None:
    """Requirement 6: Primary diagnosis plus secondary conditions.

    Acute diverticulitis with perforation (primary occasioning admission) plus
    Acute kidney injury (secondary condition treated with IV fluids).
    """
    mock_llm = MagicMock()
    classifier = PrimarySecondaryClassifier(mock_llm)

    diverticulitis = ContextAssessment(
        condition_id="div-01",
        diagnosis="Acute perforated diverticulitis",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Admitted for acute perforated diverticulitis; emergency sigmoid resection performed.",
        reason="Acute condition occasioning urgent admission and surgical intervention.",
        treated_or_managed=True,
        monitored=True,
        affected_clinical_management=True,
        treatment_evidence="emergency sigmoid resection performed",
    )

    aki = ContextAssessment(
        condition_id="aki-01",
        diagnosis="Acute kidney injury",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Acute kidney injury secondary to dehydration, treated with IV fluids.",
        reason="Co-existing acute condition treated and monitored during admission.",
        treated_or_managed=True,
        monitored=True,
        treatment_evidence="treated with IV fluids",
    )

    result, _ = classifier.classify_conditions([diverticulitis, aki])

    assert result.has_unique_primary is True
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.diagnosis == "Acute perforated diverticulitis"
    assert len(result.secondary_diagnoses) == 1
    assert result.secondary_diagnoses[0].diagnosis == "Acute kidney injury"
    assert result.secondary_diagnoses[0].role == DiagnosisRole.SECONDARY


# ============================================================================
# Deterministic Validator Tests & Rejection of Multiple Primaries
# ============================================================================


def test_deterministic_validator_rejects_llm_multiple_primaries() -> None:
    """Requirement: The deterministic validator must strictly reject multiple primary diagnoses from an LLM.

    Demonstrates that even if an LLM erroneously assigns role='PRIMARY' to multiple conditions,
    the deterministic layer catches the violation, demotes them, and emits MULTIPLE_AMBIGUOUS_PRIMARY.
    """
    validator = PrimarySecondaryClassificationValidator()

    # LLM erroneously marked TWO diagnoses as PRIMARY
    invalid_llm_output = [
        ConditionClassification(
            diagnosis_id="diag-01",
            diagnosis="Sepsis",
            role=DiagnosisRole.PRIMARY,  # PRIMARY 1
            is_billable_candidate=True,
            classification_reason="Severe infection.",
        ),
        ConditionClassification(
            diagnosis_id="diag-02",
            diagnosis="Bacterial pneumonia",
            role=DiagnosisRole.PRIMARY,  # PRIMARY 2 (VIOLATION)
            is_billable_candidate=True,
            classification_reason="Lung infection.",
        ),
        ConditionClassification(
            diagnosis_id="diag-03",
            diagnosis="Atrial fibrillation",
            role=DiagnosisRole.SECONDARY,
            is_billable_candidate=True,
            classification_reason="Rate controlled with metoprolol.",
        ),
    ]

    # Validate classification
    result, abstentions = validator.validate_classification(invalid_llm_output)

    # Deterministic layer MUST strip the multiple primaries to prevent invalid billing!
    assert result.primary_diagnosis is None
    assert result.has_unique_primary is False
    assert result.is_ambiguous_primary is True
    assert result.abstention_recommended is True
    assert result.abstention_reason == AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY

    # Check abstention records
    assert len(abstentions) >= 2
    for abst in abstentions:
        assert abst.reason == AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY
        assert (
            "Multiple conflicting primary diagnoses" in abst.detail or "Conflicting" in abst.detail
        )

    # Check strict exception mode
    with pytest.raises(MultiplePrimaryDiagnosesError, match="Multiple primary diagnoses detected"):
        validator.validate_or_raise(invalid_llm_output)


def test_deterministic_validator_demotes_historical_primary() -> None:
    """Requirement: Deterministic validator demotes a historical-only condition if mistakenly marked as primary."""
    validator = PrimarySecondaryClassificationValidator()

    ctx = ContextAssessment(
        condition_id="diag-old",
        diagnosis="Old healed myocardial infarction",
        current_relevance=False,
        coding_candidate=False,
        status=ConditionStatus.HISTORICAL,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.HISTORICAL,
        evidence="MI occurred in 2012, no current intervention.",
        reason="Historical condition.",
    )

    invalid_output = [
        ConditionClassification(
            diagnosis_id="diag-old",
            diagnosis="Old healed myocardial infarction",
            role=DiagnosisRole.PRIMARY,  # INVALID FOR HISTORICAL
            is_billable_candidate=True,
            classification_reason="Historical MI.",
        )
    ]

    result, abstentions = validator.validate_classification(invalid_output, [ctx])

    assert result.primary_diagnosis is None
    assert len(result.excluded_conditions) == 1
    assert result.excluded_conditions[0].role == DiagnosisRole.EXCLUDED
    assert result.excluded_conditions[0].is_billable_candidate is False
    assert any(a.reason == AbstentionReason.EXCLUDED_BY_TEMPORALITY for a in abstentions)


# ============================================================================
# LangGraph Node Execution Tests
# ============================================================================


def test_langgraph_classify_diagnoses_node_execution() -> None:
    """Requirement: LangGraph node execution updating graph state with classified diagnoses."""
    assessment_1 = ContextAssessment(
        condition_id="c-01",
        diagnosis="Acute cholecystitis",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Admitted for acute cholecystitis, underwent laparoscopic cholecystectomy.",
        reason="Operative admission condition.",
        treated_or_managed=True,
        treatment_evidence="laparoscopic cholecystectomy",
    )
    assessment_2 = ContextAssessment(
        condition_id="c-02",
        diagnosis="Hypertension",
        current_relevance=False,
        coding_candidate=False,
        status=ConditionStatus.HISTORICAL,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.HISTORICAL,
        evidence="Past history: hypertension.",
        reason="Historical condition.",
    )

    state: PipelineGraphState = {
        "document_id": "doc-test-123",
        "raw_text": "Sample clinical summary text.",
        "context_assessments": [assessment_1, assessment_2],
    }

    mock_llm = MagicMock()
    classifier = PrimarySecondaryClassifier(mock_llm)
    res_state = classifier.run_node(state)

    assert "classified_diagnoses" in res_state
    assert "classification_result" in res_state
    assert res_state["current_stage"] == PipelineStage.CLASSIFICATION

    enc_res: EncounterClassificationResult = res_state["classification_result"]
    assert enc_res.has_unique_primary is True
    assert enc_res.primary_diagnosis is not None
    assert enc_res.primary_diagnosis.diagnosis == "Acute cholecystitis"

    classified = res_state["classified_diagnoses"]
    assert len(classified) == 2
    primary = next(c for c in classified if c.role == DiagnosisRole.PRIMARY)
    assert primary.raw_term == "Acute cholecystitis"
    assert primary.is_billable_candidate is True


@pytest.mark.asyncio
async def test_async_classification_execution() -> None:
    """Test asynchronous execution of classify_conditions_async."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=json.dumps(
            {
                "has_unique_primary": True,
                "classifications": [
                    {
                        "diagnosis_id": "cond-afib",
                        "diagnosis": "Atrial fibrillation with RVR",
                        "role": "PRIMARY",
                        "is_billable_candidate": True,
                        "classification_reason": "Admitted for RVR treated with diltiazem drip.",
                        "primary_justification": "Chief condition occasioning admission.",
                    }
                ],
            }
        )
    )

    classifier = PrimarySecondaryClassifier(mock_llm)
    assessment = ContextAssessment(
        condition_id="cond-afib",
        diagnosis="Atrial fibrillation with RVR",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Admitted for RVR treated with diltiazem drip.",
        reason="Primary admitting condition.",
        treated_or_managed=True,
    )

    result, _ = await classifier.classify_conditions_async(
        [assessment], "Admitted for RVR treated with diltiazem drip."
    )

    assert result.has_unique_primary is True
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.diagnosis == "Atrial fibrillation with RVR"
    assert result.primary_diagnosis.role == DiagnosisRole.PRIMARY
