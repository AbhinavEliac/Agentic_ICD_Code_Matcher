"""Unit and end-to-end regression tests verifying that:
1. All valid clinical diagnoses are extracted.
2. Rank 1 candidate is Primary.
3. All remaining valid candidates are Secondary.
4. No secondary diagnoses are dropped or zeroed out.
"""

import pytest
from medical_coding.agents.classifier import PrimarySecondaryClassifier
from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor
from medical_coding.graph.pipeline import process_clinical_document
from medical_coding.models.base import BaseLocalLLM
from medical_coding.schemas.clinical import ContextAssessment
from medical_coding.schemas.enums import (
    Certainty,
    ConditionStatus,
    DiagnosisRole,
    Temporality,
)


class MockRuleLLM(BaseLocalLLM):
    def generate(self, prompt: str) -> str:
        return ""

    def get_model_info(self):
        return {"model_name": "mock-rule"}


def test_classifier_rank1_primary_rest_secondary() -> None:
    """Test that PrimarySecondaryClassifier unconditionally designates Rank 1 as Primary and rest as Secondary."""
    classifier = PrimarySecondaryClassifier(llm=MockRuleLLM())

    c1 = ContextAssessment(
        condition_id="c1",
        diagnosis="Acute pancreatitis",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Admitted for acute pancreatitis with elevated lipase 1200 U/L.",
        reason="Acute condition occasioning admission.",
        treated_or_managed=True,
    )
    c2 = ContextAssessment(
        condition_id="c2",
        diagnosis="Cholelithiasis",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACTIVE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Ultrasound showed multiple gallstones in gallbladder.",
        reason="Co-existing biliary disease.",
        treated_or_managed=True,
    )
    c3 = ContextAssessment(
        condition_id="c3",
        diagnosis="Type 2 diabetes mellitus",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.CHRONIC,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Managed on sliding scale insulin throughout stay.",
        reason="Chronic comorbid condition.",
        treated_or_managed=True,
    )

    result, abstentions = classifier.classify_conditions([c1, c2, c3])

    assert result.has_unique_primary is True
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.diagnosis == "Acute pancreatitis"
    assert result.primary_diagnosis.role == DiagnosisRole.PRIMARY

    assert len(result.secondary_diagnoses) == 2
    sec_names = [s.diagnosis for s in result.secondary_diagnoses]
    assert "Cholelithiasis" in sec_names
    assert "Type 2 diabetes mellitus" in sec_names
    for sec in result.secondary_diagnoses:
        assert sec.role == DiagnosisRole.SECONDARY
        assert sec.is_billable_candidate is True


def test_fact_extractor_rank1_primary_rest_secondary() -> None:
    """Test that EvidenceFirstFactExtractor designates Rank 1 as Primary and all remaining as Secondary."""
    note = """
    DISCHARGE SUMMARY
    DIAGNOSIS:
    1. Acute appendicitis
    2. Dehydration
    3. Hypokalemia
    4. Essential hypertension

    HOSPITAL COURSE:
    Patient admitted for acute appendicitis, underwent laparoscopic appendectomy.
    Dehydration treated with aggressive intravenous fluids.
    Hypokalemia corrected with IV potassium replacement.
    Hypertension continued on home amlodipine 5 mg daily.
    """
    extractor = EvidenceFirstFactExtractor()
    state = extractor.extract_clinical_state(note, "doc-test-rank")

    assert state.primary_diagnosis is not None
    assert state.primary_diagnosis.role == DiagnosisRole.PRIMARY
    assert "appendicitis" in state.primary_diagnosis.normalized_diagnosis.lower()

    sec_terms = [s.normalized_diagnosis.lower() for s in state.secondary_diagnoses]
    assert len(state.secondary_diagnoses) >= 2
    assert any("dehydration" in t for t in sec_terms)
    assert any("hypokalemia" in t for t in sec_terms)
    for s in state.secondary_diagnoses:
        assert s.role == DiagnosisRole.SECONDARY


@pytest.mark.asyncio
async def test_full_pipeline_multi_condition_formatting() -> None:
    """End-to-end pipeline test verifying that Rank 1 is Primary, rest are Secondary, and formatted output is exact."""
    note = """
    DISCHARGE SUMMARY
    ADMITTING DIAGNOSIS: Acute cholecystitis
    SECONDARY DIAGNOSES: Choledocholithiasis, Type 2 diabetes mellitus, Hypertension

    PROCEDURES:
    Laparoscopic cholecystectomy

    HOSPITAL COURSE:
    Patient was admitted with acute RUQ pain and fever, confirmed acute cholecystitis on ultrasound.
    Underwent successful laparoscopic cholecystectomy.
    MRCP revealed choledocholithiasis without cholangitis.
    Blood sugars monitored on sliding scale insulin for Type 2 diabetes mellitus.
    Blood pressure maintained on lisinopril for hypertension.
    Discharged stable.
    """

    final_result = await process_clinical_document(
        source=note,
        document_id="doc-pipeline-multi",
    )
    assert final_result is not None
    assert final_result.primary_diagnosis is not None
    assert final_result.primary_diagnosis.role == DiagnosisRole.PRIMARY
    assert "cholecystitis" in final_result.primary_diagnosis.description.lower()

    # All valid conditions with local code are in secondary diagnoses
    assert len(final_result.secondary_diagnoses) >= 2
    sec_names = [s.description.lower() for s in final_result.secondary_diagnoses]
    assert any("diabetes" in n for n in sec_names)
    assert any("hypertension" in n for n in sec_names)

    # Conditions without local code match are strictly put into abstentions
    abst_terms = [(a.raw_term or "").lower() for a in final_result.abstentions]
    assert any("choledocholithiasis" in t for t in abst_terms)

    output = final_result.format_output()
    assert "primary diagnosis: " in output
    assert "secondary diagnosises:" in output
    assert "code: " in output
    assert "evidence: " in output
    assert "[NO LOCAL CODE]" not in output
    assert "code: None" not in output


@pytest.mark.asyncio
async def test_no_local_code_abstention_guarantee() -> None:
    """Verify that any clinical condition without a valid local database code is strictly abstained."""
    note = """
    DISCHARGE SUMMARY
    PRIMARY DIAGNOSIS: Deviated nasal septum
    SECONDARY DIAGNOSES: Nasal obstruction, Chronic sinusitis
    HOSPITAL COURSE: Patient presented for septoplasty due to deviated nasal septum.
    Also noted nasal obstruction and chronic sinusitis.
    """
    final_result = await process_clinical_document(
        source=note,
        document_id="doc-uncoded-abstain-test",
    )
    assert final_result is not None
    # For every item in secondary diagnoses, code must not be empty or NO_DATABASE_MATCH
    for s in final_result.secondary_diagnoses:
        assert s.code and s.code != "None" and s.code != "[NO LOCAL CODE]"
        assert s.matching_status != "NO_DATABASE_MATCH"

    if final_result.primary_diagnosis:
        assert final_result.primary_diagnosis.code
        assert final_result.primary_diagnosis.code not in ("None", "[NO LOCAL CODE]")
        assert final_result.primary_diagnosis.matching_status != "NO_DATABASE_MATCH"

    output = final_result.format_output()
    assert "[NO LOCAL CODE]" not in output
    assert "code: None" not in output

