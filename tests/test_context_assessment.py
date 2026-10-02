"""Unit tests for the Context and Clinical Relevance Agent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from medical_coding.agents.classifier import ContextAndRelevanceAgent
from medical_coding.schemas.clinical import (
    ContextAssessment,
    EvidenceLocation,
    ExtractedClinicalCondition,
)
from medical_coding.schemas.enums import (
    AbstentionReason,
    Certainty,
    ConditionStatus,
    NegationStatus,
    PipelineStage,
    Temporality,
)
from medical_coding.schemas.state import PipelineGraphState

# ============================================================================
# Test Cases 1 - 7: Specific Clinical Scenarios
# ============================================================================


def test_scenario_pmh_alone_not_coding_candidate() -> None:
    """Requirement: Past Medical History alone must NOT automatically qualify as a secondary diagnosis.

    Example: "Past history: diabetes mellitus." without active management.
    """
    mock_llm = MagicMock()
    agent = ContextAndRelevanceAgent(mock_llm)

    condition = ExtractedClinicalCondition(
        condition_id="pmh-dm-01",
        original_mention="diabetes mellitus",
        normalized_description="Type 2 diabetes mellitus",
        evidence_text="Past medical history: diabetes mellitus for 10 years.",
        evidence_location=EvidenceLocation(section="PAST_MEDICAL_HISTORY", page_number=1),
        section="PAST_MEDICAL_HISTORY",
        status=ConditionStatus.HISTORICAL,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.HISTORICAL,
        treatment_evidence=None,
    )

    assessments = agent._fallback_rule_assessment([condition])

    assert len(assessments) == 1
    asm = assessments[0]
    assert asm.diagnosis == "Type 2 diabetes mellitus"
    assert asm.coding_candidate is False
    assert asm.current_relevance is False
    assert asm.status == ConditionStatus.HISTORICAL
    assert asm.temporality == Temporality.HISTORICAL
    assert "Past Medical History alone" in asm.reason
    assert asm.treated_or_managed is False
    assert asm.monitored is False


def test_scenario_pmh_with_monitoring_and_treatment_is_coding_candidate() -> None:
    """Requirement: PMH with active monitoring and medication adjustment during admission qualifies.

    Example: "Diabetes mellitus was monitored and insulin was adjusted during admission."
    """
    mock_llm = MagicMock()
    agent = ContextAndRelevanceAgent(mock_llm)

    condition = ExtractedClinicalCondition(
        condition_id="pmh-dm-active-02",
        original_mention="Diabetes mellitus",
        normalized_description="Type 2 diabetes mellitus",
        evidence_text="Diabetes mellitus was monitored and insulin was adjusted during admission.",
        evidence_location=EvidenceLocation(section="HOSPITAL_COURSE", page_number=1),
        section="HOSPITAL_COURSE",
        status=ConditionStatus.CHRONIC,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        treatment_evidence="insulin was adjusted",
    )

    assessments = agent._fallback_rule_assessment([condition])

    assert len(assessments) == 1
    asm = assessments[0]
    assert asm.diagnosis == "Type 2 diabetes mellitus"
    assert asm.coding_candidate is True
    assert asm.current_relevance is True
    assert asm.status == ConditionStatus.CHRONIC
    assert asm.temporality == Temporality.CURRENT
    assert asm.treated_or_managed is True
    assert asm.monitored is True
    assert asm.affected_clinical_management is True


def test_scenario_current_active_acute_condition() -> None:
    """Requirement: Current acute condition treated during stay is clinically relevant and a coding candidate."""
    mock_llm = MagicMock()
    agent = ContextAndRelevanceAgent(mock_llm)

    condition = ExtractedClinicalCondition(
        condition_id="uti-01",
        original_mention="acute urinary tract infection",
        normalized_description="Acute urinary tract infection",
        evidence_text="Patient diagnosed with acute urinary tract infection, treated with 5 days of IV ceftriaxone.",
        evidence_location=EvidenceLocation(section="HOSPITAL_COURSE", page_number=1),
        section="HOSPITAL_COURSE",
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        treatment_evidence="treated with 5 days of IV ceftriaxone",
    )

    assessments = agent._fallback_rule_assessment([condition])

    assert len(assessments) == 1
    asm = assessments[0]
    assert asm.diagnosis == "Acute urinary tract infection"
    assert asm.coding_candidate is True
    assert asm.current_relevance is True
    assert asm.status == ConditionStatus.ACUTE
    assert asm.certainty == Certainty.CONFIRMED
    assert asm.treated_or_managed is True
    assert asm.influenced_treatment is True


def test_scenario_ruled_out_pneumonia_not_confirmed() -> None:
    """Requirement: Suspected condition ruled out by imaging/workup must NOT become confirmed.

    Example: "Possible pneumonia was suspected initially but imaging showed no evidence of pneumonia."
    """
    mock_llm = MagicMock()
    agent = ContextAndRelevanceAgent(mock_llm)

    condition = ExtractedClinicalCondition(
        condition_id="pna-ruled-out-01",
        original_mention="possible pneumonia",
        normalized_description="Bacterial pneumonia",
        evidence_text="Possible pneumonia was suspected initially but imaging showed no evidence of pneumonia.",
        evidence_location=EvidenceLocation(section="HOSPITAL_COURSE", page_number=1),
        section="HOSPITAL_COURSE",
        status=ConditionStatus.RESOLVED,
        certainty=Certainty.RULED_OUT,
        temporality=Temporality.CURRENT,
        negation=NegationStatus.NEGATED,
        treatment_evidence=None,
    )

    assessments = agent._fallback_rule_assessment([condition])

    assert len(assessments) == 1
    asm = assessments[0]
    assert asm.diagnosis == "Bacterial pneumonia"
    assert asm.coding_candidate is False
    assert asm.current_relevance is False
    assert asm.certainty == Certainty.RULED_OUT
    assert asm.negation == NegationStatus.NEGATED
    assert "ruled out" in asm.reason.lower()


def test_scenario_suspected_condition_preserves_uncertainty() -> None:
    """Requirement: Suspected/uncertain condition must NOT be converted to CONFIRMED.

    Under inpatient coding guidelines, suspected conditions at discharge evaluated/treated
    may be coded, but certainty must remain SUSPECTED.
    """
    mock_llm = MagicMock()
    agent = ContextAndRelevanceAgent(mock_llm)

    condition = ExtractedClinicalCondition(
        condition_id="pna-suspected-02",
        original_mention="suspected bacterial pneumonia",
        normalized_description="Bacterial pneumonia",
        evidence_text="Suspected bacterial pneumonia treated with empiric levofloxacin, culture pending at discharge.",
        evidence_location=EvidenceLocation(section="DISCHARGE_DIAGNOSIS", page_number=1),
        section="DISCHARGE_DIAGNOSIS",
        status=ConditionStatus.ACTIVE,
        certainty=Certainty.SUSPECTED,
        temporality=Temporality.CURRENT,
        treatment_evidence="treated with empiric levofloxacin",
    )

    assessments = agent._fallback_rule_assessment([condition])

    assert len(assessments) == 1
    asm = assessments[0]
    assert asm.diagnosis == "Bacterial pneumonia"
    assert asm.coding_candidate is True
    assert asm.current_relevance is True
    # MUST retain SUSPECTED, never converted to CONFIRMED
    assert asm.certainty == Certainty.SUSPECTED
    assert asm.treated_or_managed is True


def test_scenario_resolved_condition_during_stay() -> None:
    """Requirement: Acute condition that resolved during stay following active treatment qualifies as coding candidate."""
    mock_llm = MagicMock()
    agent = ContextAndRelevanceAgent(mock_llm)

    condition = ExtractedClinicalCondition(
        condition_id="hypok-01",
        original_mention="acute hypokalemia",
        normalized_description="Hypokalemia",
        evidence_text="Acute hypokalemia, resolved following IV potassium replacement.",
        evidence_location=EvidenceLocation(section="HOSPITAL_COURSE", page_number=1),
        section="HOSPITAL_COURSE",
        status=ConditionStatus.RESOLVED,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        treatment_evidence="IV potassium replacement",
    )

    assessments = agent._fallback_rule_assessment([condition])

    assert len(assessments) == 1
    asm = assessments[0]
    assert asm.diagnosis == "Hypokalemia"
    assert asm.coding_candidate is True
    assert asm.current_relevance is True
    assert asm.status == ConditionStatus.RESOLVED
    assert asm.treated_or_managed is True
    assert "resolved" in asm.reason.lower()


def test_scenario_ambiguous_contradictory_triggers_abstention() -> None:
    """Requirement: Contradictory clinical documentation must trigger explicit abstention rather than forced classification."""
    mock_llm = MagicMock()
    agent = ContextAndRelevanceAgent(mock_llm)

    condition = ExtractedClinicalCondition(
        condition_id="sepsis-ambig-01",
        original_mention="sepsis vs non-infectious SIRS",
        normalized_description="Sepsis",
        evidence_text="Contradictory notes: ED diagnosed severe sepsis, but ID consult definitively concluded no sepsis criteria met.",
        evidence_location=EvidenceLocation(section="CONSULTATION_NOTE", page_number=2),
        section="CONSULTATION_NOTE",
        status=ConditionStatus.UNKNOWN,
        certainty=Certainty.SUSPECTED,
        temporality=Temporality.CURRENT,
        treatment_evidence=None,
    )

    assessments = agent._fallback_rule_assessment([condition])

    assert len(assessments) == 1
    asm = assessments[0]
    assert asm.diagnosis == "Sepsis"
    assert asm.abstain_recommended is True
    assert asm.abstention_reason == AbstentionReason.CONTRADICTORY_DOCUMENTATION
    assert asm.coding_candidate is False
    assert "abstention" in asm.reason.lower() or "contradictory" in asm.reason.lower()


# ============================================================================
# Deterministic Guardrails & LLM Override Tests
# ============================================================================


def test_guardrail_overrides_llm_hallucinated_pmh_eligibility() -> None:
    """Requirement: Even if LLM hallucinates coding_candidate=True for PMH alone, deterministic guardrails override it."""
    mock_llm = MagicMock()
    # LLM improperly marks PMH condition as coding_candidate = true
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "diagnosis": "Essential hypertension",
                "current_relevance": True,
                "coding_candidate": True,  # HALLUCINATED TRUE FOR UNMANAGED PMH
                "status": "ACTIVE",
                "certainty": "CONFIRMED",
                "temporality": "CURRENT",
                "evidence": "Past medical history: hypertension for 5 years.",
                "reason": "Common condition.",
                "treated_or_managed": False,
                "monitored": False,
                "affected_clinical_management": False,
                "influenced_treatment": False,
            }
        ]
    )

    agent = ContextAndRelevanceAgent(mock_llm)
    condition = ExtractedClinicalCondition(
        condition_id="htn-01",
        original_mention="hypertension",
        normalized_description="Essential hypertension",
        evidence_text="Past medical history: hypertension for 5 years.",
        evidence_location=EvidenceLocation(section="PAST_MEDICAL_HISTORY", page_number=1),
        section="PAST_MEDICAL_HISTORY",
        status=ConditionStatus.HISTORICAL,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.HISTORICAL,
    )

    assessments = agent.assess_conditions(
        [condition], "Past medical history: hypertension for 5 years."
    )

    assert len(assessments) == 1
    asm = assessments[0]
    # Guardrail must strictly set coding_candidate to False!
    assert asm.coding_candidate is False
    assert asm.current_relevance is False
    assert asm.status == ConditionStatus.HISTORICAL
    assert asm.temporality == Temporality.HISTORICAL


def test_guardrail_overrides_llm_confirming_ruled_out_condition() -> None:
    """Requirement: Even if LLM erroneously marks a ruled-out condition as confirmed, guardrail enforces RULED_OUT."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "diagnosis": "Pneumonia",
                "current_relevance": True,
                "coding_candidate": True,
                "status": "ACTIVE",
                "certainty": "CONFIRMED",  # ERRONEOUS
                "evidence": "Chest CT showed clear lung fields; pneumonia ruled out.",
                "reason": "Evaluated during stay.",
            }
        ]
    )

    agent = ContextAndRelevanceAgent(mock_llm)
    condition = ExtractedClinicalCondition(
        condition_id="pna-03",
        original_mention="pneumonia",
        normalized_description="Pneumonia",
        evidence_text="Chest CT showed clear lung fields; pneumonia ruled out.",
        evidence_location=EvidenceLocation(section="HOSPITAL_COURSE", page_number=1),
        section="HOSPITAL_COURSE",
    )

    assessments = agent.assess_conditions(
        [condition], "Chest CT showed clear lung fields; pneumonia ruled out."
    )

    assert len(assessments) == 1
    asm = assessments[0]
    assert asm.coding_candidate is False
    assert asm.current_relevance is False
    assert asm.certainty == Certainty.RULED_OUT
    assert asm.negation == NegationStatus.NEGATED


def test_guardrail_preserves_suspected_certainty_when_llm_promotes_to_confirmed() -> None:
    """Requirement: If extracted condition is SUSPECTED and LLM outputs CONFIRMED, guardrail restores SUSPECTED."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "diagnosis": "Aspiration pneumonia",
                "current_relevance": True,
                "coding_candidate": True,
                "status": "ACTIVE",
                "certainty": "CONFIRMED",  # LLM erroneously promoted to confirmed
                "evidence": "Suspected aspiration pneumonia treated with empiric ampicillin-sulbactam.",
                "reason": "Treated with antibiotics.",
                "treated_or_managed": True,
            }
        ]
    )

    agent = ContextAndRelevanceAgent(mock_llm)
    condition = ExtractedClinicalCondition(
        condition_id="asp-pna-01",
        original_mention="suspected aspiration pneumonia",
        normalized_description="Aspiration pneumonia",
        evidence_text="Suspected aspiration pneumonia treated with empiric ampicillin-sulbactam.",
        evidence_location=EvidenceLocation(section="DISCHARGE_DIAGNOSES", page_number=1),
        section="DISCHARGE_DIAGNOSES",
        certainty=Certainty.SUSPECTED,  # Originally suspected
        status=ConditionStatus.ACTIVE,
        treatment_evidence="treated with empiric ampicillin-sulbactam",
    )

    assessments = agent.assess_conditions(
        [condition], "Suspected aspiration pneumonia treated with empiric ampicillin-sulbactam."
    )

    assert len(assessments) == 1
    asm = assessments[0]
    assert asm.coding_candidate is True
    # Certainty must be preserved as SUSPECTED
    assert asm.certainty == Certainty.SUSPECTED


# ============================================================================
# BaseAgent & LangGraph Node Compatibility Tests
# ============================================================================


def test_base_agent_run_contract() -> None:
    """Requirement: ContextAndRelevanceAgent conforms to BaseAgent interface and implements run()."""
    mock_llm = MagicMock()
    agent = ContextAndRelevanceAgent(mock_llm)

    condition = ExtractedClinicalCondition(
        condition_id="chole-01",
        original_mention="acute cholecystitis",
        normalized_description="Acute cholecystitis",
        evidence_text="Laparoscopic cholecystectomy performed for acute cholecystitis.",
        evidence_location=EvidenceLocation(section="OPERATIVE_REPORT", page_number=1),
        section="OPERATIVE_REPORT",
        treatment_evidence="Laparoscopic cholecystectomy performed",
    )

    results = agent.run(
        conditions=[condition],
        clinical_text="Laparoscopic cholecystectomy performed for acute cholecystitis.",
    )
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], ContextAssessment)
    assert results[0].coding_candidate is True


def test_langgraph_run_node_execution() -> None:
    """Requirement: Implement this as a LangChain component compatible with the LangGraph state."""
    mock_llm = MagicMock()
    agent = ContextAndRelevanceAgent(mock_llm)

    condition = ExtractedClinicalCondition(
        condition_id="aki-01",
        original_mention="acute kidney injury",
        normalized_description="Acute kidney injury",
        evidence_text="Acute kidney injury secondary to dehydration, creatinine resolved with IV normal saline.",
        evidence_location=EvidenceLocation(section="HOSPITAL_COURSE", page_number=1),
        section="HOSPITAL_COURSE",
        treatment_evidence="resolved with IV normal saline",
    )

    ambig_condition = ExtractedClinicalCondition(
        condition_id="ambig-01",
        original_mention="pulmonary embolism vs anxiety",
        normalized_description="Pulmonary embolism",
        evidence_text="Conflicting documentation regarding pulmonary embolism, CTA equivocal and unconfirmed.",
        evidence_location=EvidenceLocation(section="HOSPITAL_COURSE", page_number=2),
        section="HOSPITAL_COURSE",
    )

    state: PipelineGraphState = {
        "raw_text": "Sample discharge text.",
        "extracted_conditions": [condition, ambig_condition],
    }

    result_state = agent.run_node(state)

    assert "context_assessments" in result_state
    assert "contextualized_diagnoses" in result_state
    assert "abstentions" in result_state
    assert result_state["current_stage"] == PipelineStage.CONTEXT_ANALYSIS

    assessments = result_state["context_assessments"]
    assert len(assessments) == 2

    aki_asm = next(a for a in assessments if a.diagnosis == "Acute kidney injury")
    assert aki_asm.coding_candidate is True
    assert aki_asm.current_relevance is True

    pe_asm = next(a for a in assessments if a.diagnosis == "Pulmonary embolism")
    assert pe_asm.abstain_recommended is True
    assert pe_asm.coding_candidate is False

    # Check abstention record in state
    abstentions = result_state["abstentions"]
    assert len(abstentions) == 1
    assert abstentions[0].diagnosis_id == "ambig-01"
    assert abstentions[0].reason == AbstentionReason.CONTRADICTORY_DOCUMENTATION


@pytest.mark.asyncio
async def test_async_context_assessment() -> None:
    """Test asynchronous execution of assess_conditions_async."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=json.dumps(
            [
                {
                    "diagnosis": "Atrial fibrillation",
                    "current_relevance": True,
                    "coding_candidate": True,
                    "status": "ACTIVE",
                    "certainty": "CONFIRMED",
                    "temporality": "CURRENT",
                    "evidence": "Rapid ventricular response controlled with IV diltiazem.",
                    "reason": "Treated with IV diltiazem during admission.",
                    "treated_or_managed": True,
                    "monitored": True,
                }
            ]
        )
    )

    agent = ContextAndRelevanceAgent(mock_llm)
    condition = ExtractedClinicalCondition(
        condition_id="afib-01",
        original_mention="atrial fibrillation with RVR",
        normalized_description="Atrial fibrillation",
        evidence_text="Rapid ventricular response controlled with IV diltiazem.",
        evidence_location=EvidenceLocation(section="HOSPITAL_COURSE", page_number=1),
        section="HOSPITAL_COURSE",
        treatment_evidence="IV diltiazem",
    )

    results = await agent.assess_conditions_async(
        [condition], "Rapid ventricular response controlled with IV diltiazem."
    )

    assert len(results) == 1
    assert results[0].diagnosis == "Atrial fibrillation"
    assert results[0].coding_candidate is True
    assert results[0].treated_or_managed is True
