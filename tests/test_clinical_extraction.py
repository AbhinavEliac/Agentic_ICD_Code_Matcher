"""Unit tests and clinical case fixtures for the Clinical Information Extraction Agent."""

import json
from unittest.mock import MagicMock

from medical_coding.agents.extractor import ClinicalExtractionAgent
from medical_coding.agents.parser import ExtractionParser
from medical_coding.schemas.enums import (
    Certainty,
    ConditionStatus,
    NegationStatus,
    Temporality,
)

# ============================================================================
# Clinical Case Fixtures
# ============================================================================

CLINICAL_NOTE_CURRENT_UTI = """DISCHARGE SUMMARY
CHIEF COMPLAINT: Dysuria, fever, and urinary frequency.
HOSPITAL COURSE: Patient diagnosed with acute urinary tract infection (UTI), urine culture positive for Escherichia coli. Treated with 5 days of IV ceftriaxone with complete resolution of dysuria and fever.
DISCHARGE DIAGNOSES:
1. Acute urinary tract infection.
"""

CLINICAL_NOTE_HISTORICAL_DIABETES = """DISCHARGE SUMMARY
CHIEF COMPLAINT: Right ankle inversion sprain while jogging.
PAST MEDICAL HISTORY:
1. Type 2 diabetes mellitus diagnosed 15 years ago, managed on metformin.
2. Essential hypertension.
DISCHARGE DIAGNOSES:
1. Grade 2 right ankle sprain.
"""

CLINICAL_NOTE_SUSPECTED_PNEUMONIA_RULED_OUT = """DISCHARGE SUMMARY
HISTORY OF PRESENT ILLNESS: Patient presented with cough, pleuritic chest pain, and low-grade fever, admitted for suspected bacterial pneumonia.
HOSPITAL COURSE: Chest CT demonstrated clear lung fields with no consolidation or infiltrate. Sputum cultures were negative. Viral PCR was positive for rhinovirus. Bacterial pneumonia was definitively ruled out.
DISCHARGE DIAGNOSES:
1. Acute viral upper respiratory infection.
"""

CLINICAL_NOTE_MULTIPLE_DIAGNOSES = """DISCHARGE SUMMARY
DISCHARGE DIAGNOSES:
1. Acute systolic heart failure.
2. Chronic obstructive pulmonary disease exacerbation.
3. Essential hypertension.
4. Acute kidney injury secondary to volume depletion.
HOSPITAL COURSE: Patient received IV furosemide for heart failure and nebulized bronchodilators for COPD. Creatinine improved with hydration.
"""

CLINICAL_NOTE_PMH_ONLY = """DISCHARGE SUMMARY
HOSPITAL COURSE: Patient underwent elective laparoscopic cholecystectomy for symptomatic cholelithiasis.
PAST MEDICAL HISTORY:
1. Longstanding gastroesophageal reflux disease (GERD).
2. Hyperlipidemia.
DISCHARGE DIAGNOSES:
1. Symptomatic cholelithiasis.
"""

CLINICAL_NOTE_FINAL_DISCHARGE_DIAGNOSIS = """DISCHARGE SUMMARY
HISTORY OF PRESENT ILLNESS: 72-year-old female admitted with crushing substernal chest pressure.
HOSPITAL COURSE: Troponins peaked at 4.2 ng/mL. Diagnostic coronary catheterization revealed 90% occlusion of the left anterior descending artery. Stent was successfully placed.
DISCHARGE DIAGNOSES:
1. Acute non-ST-elevation myocardial infarction (NSTEMI).
2. Coronary artery disease.
"""

CLINICAL_NOTE_CONTRADICTORY_MENTIONS = """DISCHARGE SUMMARY
HISTORY OF PRESENT ILLNESS: Patient presented with acute shortness of breath and tachypnea, admitted for suspected acute pulmonary embolism.
HOSPITAL COURSE: CT pulmonary angiography was completely negative for pulmonary embolism; no filling defects observed. Pulmonary embolism was definitively excluded. D-dimer was mildly elevated due to acute pleurisy.
DISCHARGE DIAGNOSES:
1. Acute viral pleurisy.
"""


# ============================================================================
# Unit Tests: Parser and Grounding Rules
# ============================================================================


def test_parser_strips_markdown_and_repairs_syntax() -> None:
    """Verify that parser extracts JSON from markdown fences and repairs single quotes/trailing commas."""
    parser = ExtractionParser()
    doc_text = "Patient was admitted for acute appendicitis. Appendectomy performed."

    # Raw LLM output with conversational preamble, markdown fence, single quotes, and trailing comma
    noisy_llm_output = """Here is the extracted condition:
```json
[
  {
    'original_mention': 'acute appendicitis',
    'normalized_description': 'Acute appendicitis',
    'evidence_text': 'Patient was admitted for acute appendicitis.',
    'certainty': 'confirmed',
    'temporality': 'current',
    'status': 'acute',
    'section': 'DISCHARGE_DIAGNOSES',
  },
]
```
Hope this helps!"""

    conditions, notes, repaired = parser.parse_and_validate(noisy_llm_output, doc_text)

    assert len(conditions) == 1
    assert repaired is True
    cond = conditions[0]
    assert cond.original_mention == "acute appendicitis"
    assert cond.certainty == Certainty.CONFIRMED
    assert cond.temporality == Temporality.CURRENT
    assert cond.status == ConditionStatus.ACUTE
    assert cond.evidence_text == "Patient was admitted for acute appendicitis."


def test_parser_drops_hallucinated_evidence_without_proof() -> None:
    """Requirement: Every diagnosis must have evidence. Drop items where evidence is fabricated."""
    parser = ExtractionParser()
    doc_text = "Patient presented with a simple rash on right arm."

    # LLM hallucinates diabetes with a fake quote that does not exist in doc_text
    hallucinated_llm_output = json.dumps(
        [
            {
                "original_mention": "Type 2 diabetes",
                "normalized_description": "Type 2 diabetes",
                "evidence_text": "Patient has severe uncontrolled diabetes.",  # NOT IN DOC
                "certainty": "CONFIRMED",
                "temporality": "CURRENT",
            }
        ]
    )

    conditions, notes, _ = parser.parse_and_validate(hallucinated_llm_output, doc_text)

    # Condition MUST be discarded because evidence quote is not in document
    assert len(conditions) == 0
    assert any("discarded: evidence quote" in note for note in notes)


# ============================================================================
# Test Cases 1 - 7: Specific Clinical Scenarios
# ============================================================================


def test_scenario_1_current_uti() -> None:
    """Test 1: Current UTI is extracted with CURRENT temporality, CONFIRMED certainty, and treatment."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "original_mention": "acute urinary tract infection (UTI)",
                "normalized_description": "Acute urinary tract infection",
                "entity_type": "DIAGNOSIS",
                "evidence_text": "Patient diagnosed with acute urinary tract infection (UTI), urine culture positive for Escherichia coli.",
                "status": "ACTIVE",
                "certainty": "CONFIRMED",
                "temporality": "CURRENT",
                "negation": "AFFIRMATIVE",
                "anatomical_site": "urinary tract",
                "section": "HOSPITAL_COURSE",
                "treatment_evidence": "Treated with 5 days of IV ceftriaxone",
            }
        ]
    )

    agent = ClinicalExtractionAgent(llm=mock_llm)
    result = agent.extract_clinical_conditions(CLINICAL_NOTE_CURRENT_UTI)

    assert len(result.conditions) == 1
    uti = result.conditions[0]
    assert uti.normalized_description == "Acute urinary tract infection"
    assert uti.temporality == Temporality.CURRENT
    assert uti.certainty == Certainty.CONFIRMED
    assert uti.negation == NegationStatus.AFFIRMATIVE
    assert uti.treatment_evidence == "Treated with 5 days of IV ceftriaxone"


def test_scenario_2_historical_diabetes() -> None:
    """Test 2: Historical diabetes in PMH must have HISTORICAL temporality and not current acute status."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "original_mention": "Type 2 diabetes mellitus",
                "normalized_description": "Type 2 diabetes mellitus",
                "entity_type": "DIAGNOSIS",
                "evidence_text": "Type 2 diabetes mellitus diagnosed 15 years ago, managed on metformin.",
                "status": "HISTORICAL",
                "certainty": "CONFIRMED",
                "temporality": "HISTORICAL",
                "negation": "AFFIRMATIVE",
                "section": "PAST_MEDICAL_HISTORY",
            }
        ]
    )

    agent = ClinicalExtractionAgent(llm=mock_llm)
    result = agent.extract_clinical_conditions(CLINICAL_NOTE_HISTORICAL_DIABETES)

    assert len(result.conditions) == 1
    dm = result.conditions[0]
    assert dm.temporality == Temporality.HISTORICAL
    assert dm.status in (ConditionStatus.HISTORICAL, ConditionStatus.CHRONIC)
    assert dm.section == "PAST_MEDICAL_HISTORY"


def test_scenario_3_suspected_pneumonia_ruled_out() -> None:
    """Test 3: Suspected pneumonia later definitively excluded must be classified as RULED_OUT."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "original_mention": "bacterial pneumonia",
                "normalized_description": "Bacterial pneumonia",
                "entity_type": "DIAGNOSIS",
                "evidence_text": "Bacterial pneumonia was definitively ruled out.",
                "status": "RESOLVED",
                "certainty": "RULED_OUT",
                "temporality": "CURRENT",
                "negation": "NEGATED",
                "section": "HOSPITAL_COURSE",
            }
        ]
    )

    agent = ClinicalExtractionAgent(llm=mock_llm)
    result = agent.extract_clinical_conditions(CLINICAL_NOTE_SUSPECTED_PNEUMONIA_RULED_OUT)

    assert len(result.conditions) == 1
    pna = result.conditions[0]
    assert pna.certainty == Certainty.RULED_OUT
    assert pna.negation == NegationStatus.NEGATED


def test_scenario_4_multiple_diagnoses() -> None:
    """Test 4: Multiple distinct active diagnoses extracted with separate evidence grounding."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "original_mention": "Acute systolic heart failure",
                "normalized_description": "Acute systolic heart failure",
                "evidence_text": "1. Acute systolic heart failure.",
                "status": "ACTIVE",
                "certainty": "CONFIRMED",
                "temporality": "CURRENT",
                "section": "DISCHARGE_DIAGNOSES",
            },
            {
                "original_mention": "Chronic obstructive pulmonary disease exacerbation",
                "normalized_description": "COPD exacerbation",
                "evidence_text": "2. Chronic obstructive pulmonary disease exacerbation.",
                "status": "ACTIVE",
                "certainty": "CONFIRMED",
                "temporality": "CURRENT",
                "section": "DISCHARGE_DIAGNOSES",
            },
            {
                "original_mention": "Essential hypertension",
                "normalized_description": "Essential hypertension",
                "evidence_text": "3. Essential hypertension.",
                "status": "CHRONIC",
                "certainty": "CONFIRMED",
                "temporality": "CURRENT",
                "section": "DISCHARGE_DIAGNOSES",
            },
            {
                "original_mention": "Acute kidney injury",
                "normalized_description": "Acute kidney injury",
                "evidence_text": "4. Acute kidney injury secondary to volume depletion.",
                "status": "ACTIVE",
                "certainty": "CONFIRMED",
                "temporality": "CURRENT",
                "section": "DISCHARGE_DIAGNOSES",
            },
        ]
    )

    agent = ClinicalExtractionAgent(llm=mock_llm)
    result = agent.extract_clinical_conditions(CLINICAL_NOTE_MULTIPLE_DIAGNOSES)

    assert len(result.conditions) == 4
    descriptions = {c.normalized_description for c in result.conditions}
    assert "Acute systolic heart failure" in descriptions
    assert "Acute kidney injury" in descriptions


def test_scenario_5_diagnosis_in_past_medical_history_only() -> None:
    """Test 5: Condition appearing in past medical history only must be assigned HISTORICAL temporality."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "original_mention": "gastroesophageal reflux disease (GERD)",
                "normalized_description": "GERD",
                "evidence_text": "1. Longstanding gastroesophageal reflux disease (GERD).",
                "status": "CHRONIC",
                "certainty": "CONFIRMED",
                "temporality": "CURRENT",  # Even if LLM naively claimed CURRENT, PMH rule reconciles it to HISTORICAL
                "section": "PAST_MEDICAL_HISTORY",
            }
        ]
    )

    agent = ClinicalExtractionAgent(llm=mock_llm)
    result = agent.extract_clinical_conditions(CLINICAL_NOTE_PMH_ONLY)

    assert len(result.conditions) == 1
    gerd = result.conditions[0]
    # Agent reconciliation must enforce HISTORICAL temporality for PMH-only conditions
    assert gerd.temporality == Temporality.HISTORICAL


def test_scenario_6_diagnosis_in_final_discharge_diagnosis() -> None:
    """Test 6: Diagnosis in final discharge diagnoses has DISCHARGE_DIAGNOSES section and CONFIRMED certainty."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "original_mention": "Acute non-ST-elevation myocardial infarction (NSTEMI)",
                "normalized_description": "NSTEMI",
                "evidence_text": "1. Acute non-ST-elevation myocardial infarction (NSTEMI).",
                "status": "ACUTE",
                "certainty": "CONFIRMED",
                "temporality": "CURRENT",
                "section": "DISCHARGE_DIAGNOSES",
                "treatment_evidence": "Stent was successfully placed",
            }
        ]
    )

    agent = ClinicalExtractionAgent(llm=mock_llm)
    result = agent.extract_clinical_conditions(CLINICAL_NOTE_FINAL_DISCHARGE_DIAGNOSIS)

    assert len(result.conditions) == 1
    nstemi = result.conditions[0]
    assert nstemi.certainty == Certainty.CONFIRMED
    assert nstemi.temporality == Temporality.CURRENT
    assert nstemi.section == "DISCHARGE_DIAGNOSES"
    assert nstemi.treatment_evidence == "Stent was successfully placed"


def test_scenario_7_contradictory_mentions_reconciled() -> None:
    """Test 7: Mentioned as suspected initially in HPI but later ruled out in Hospital Course.

    Reconciliation must favor the definitive RULED_OUT certainty.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = json.dumps(
        [
            {
                "original_mention": "acute pulmonary embolism",
                "normalized_description": "Pulmonary embolism",
                "evidence_text": "Patient presented with acute shortness of breath and tachypnea, admitted for suspected acute pulmonary embolism.",
                "status": "ACTIVE",
                "certainty": "SUSPECTED",
                "temporality": "CURRENT",
                "negation": "AFFIRMATIVE",
                "section": "HISTORY_OF_PRESENT_ILLNESS",
            },
            {
                "original_mention": "pulmonary embolism",
                "normalized_description": "Pulmonary embolism",
                "evidence_text": "Pulmonary embolism was definitively excluded.",
                "status": "RESOLVED",
                "certainty": "RULED_OUT",
                "temporality": "CURRENT",
                "negation": "NEGATED",
                "section": "HOSPITAL_COURSE",
            },
        ]
    )

    agent = ClinicalExtractionAgent(llm=mock_llm)
    result = agent.extract_clinical_conditions(CLINICAL_NOTE_CONTRADICTORY_MENTIONS)

    # Reconciled into a single concept where RULED_OUT overrides SUSPECTED
    assert len(result.conditions) == 1
    pe = result.conditions[0]
    assert pe.certainty == Certainty.RULED_OUT
    assert pe.negation == NegationStatus.NEGATED
    assert pe.status == ConditionStatus.RESOLVED


def test_agent_retry_policy_on_malformed_first_response() -> None:
    """Verify that agent retries when first generation is malformed and succeeds on retry."""
    mock_llm = MagicMock()
    # First response is malformed / empty, second response succeeds
    mock_llm.invoke.side_effect = [
        "Sorry, I cannot produce valid JSON right now.",
        json.dumps(
            [
                {
                    "original_mention": "Grade 2 right ankle sprain",
                    "normalized_description": "Right ankle sprain",
                    "evidence_text": "1. Grade 2 right ankle sprain.",
                    "status": "ACUTE",
                    "certainty": "CONFIRMED",
                    "temporality": "CURRENT",
                    "section": "DISCHARGE_DIAGNOSES",
                }
            ]
        ),
    ]

    agent = ClinicalExtractionAgent(llm=mock_llm, max_retries=2)
    result = agent.extract_clinical_conditions(CLINICAL_NOTE_HISTORICAL_DIABETES)

    assert mock_llm.invoke.call_count == 2
    assert len(result.conditions) == 1
    assert result.conditions[0].normalized_description == "Right ankle sprain"
