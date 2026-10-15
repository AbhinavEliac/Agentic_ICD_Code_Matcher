"""Unit tests for demo-fix-1 improvements:
1) Reasoning for conditions carrying 'Primary' in their name vs UHDDS Primary Diagnosis.
2) Multi-round re-checking of abstentions with positive evidence via BM25 and vector search.
3) Strengthened clinical diagnosis normalization logic and query variations.
"""

from unittest.mock import MagicMock
import pytest

from medical_coding.agents.classifier import PrimarySecondaryClassifier
from medical_coding.retrieval.hybrid import HybridICDRetriever
from medical_coding.retrieval.lexical import BM25ICDRetriever
from medical_coding.retrieval.vector import FAISSICDRetriever
from medical_coding.models.factory import FastLocalEmbeddings
from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.schemas.clinical import ContextAssessment
from medical_coding.schemas.enums import (
    Acuity,
    Certainty,
    ConditionStatus,
    DiagnosisRole,
    ExecutionStatus,
    NegationStatus,
    PipelineStage,
    Temporality,
    AbstentionReason,
)
from medical_coding.schemas.icd import ICDCodeRecord
from medical_coding.schemas.validation import AbstentionRecord
from medical_coding.utils.text import (
    extract_search_query_variations,
    normalize_diagnosis_name,
)
from medical_coding.graph.nodes import finalize_output_node


# ==============================================================================
# 1. NORMALIZATION TESTS
# ==============================================================================


def test_normalize_diagnosis_name_strips_preambles_and_medications() -> None:
    """Test that medical preambles (K/C/O, H/O, S/P, C/O) and medications are stripped."""
    # Case 1: K/C/O and medication attachment
    raw1 = "1. K/C/O Essential (primary) hypertension on Tab Telmisartan 40mg"
    norm1 = normalize_diagnosis_name(raw1)
    assert "essential" in norm1.lower()
    assert "primary" in norm1.lower()
    assert "hypertension" in norm1.lower()
    assert "k/c/o" not in norm1.lower()
    assert "telmisartan" not in norm1.lower()

    # Case 2: H/O and clinical impression
    raw2 = "Impression: H/O Acute cholecystitis with cholelithiasis"
    norm2 = normalize_diagnosis_name(raw2)
    assert "cholecystitis" in norm2.lower()
    assert "impression" not in norm2.lower()
    assert "h/o" not in norm2.lower()

    # Case 3: S/P and bullet
    raw3 = "• S/P Appendectomy"
    norm3 = normalize_diagnosis_name(raw3)
    assert "appendectomy" in norm3.lower()
    assert "s/p" not in norm3.lower()

    # Case 4: C/O with abbreviation expansion
    raw4 = "C/O SOB and CP secondary to CHF"
    norm4 = normalize_diagnosis_name(raw4)
    assert "c/o" not in norm4.lower()
    assert "shortness of breath" in norm4.lower()
    assert "congestive heart failure" in norm4.lower()


def test_extract_search_query_variations() -> None:
    """Test generating prioritized query variations for multi-round retrieval."""
    raw = "Primary hypertension"
    variations = extract_search_query_variations(raw)
    assert len(variations) >= 3
    # Should include etiological synonym variations
    assert any("essential" in v for v in variations)
    assert any(v == "hypertension" for v in variations)

    # Test with clinical noise and evidence
    raw2 = "Known case of T2DM with peripheral neuropathy on Metformin"
    variations2 = extract_search_query_variations(raw2, evidence="Patient has diabetic peripheral neuropathy")
    assert any("type 2 diabetes mellitus" in v for v in variations2)
    assert any("neuropathy" in v for v in variations2)


# ==============================================================================
# 2. 'PRIMARY' IN DISEASE NAME VS UHDDS PRIMARY DIAGNOSIS TESTS
# ==============================================================================


def test_primary_hypertension_is_secondary_when_acute_appendicitis_occasions_admission() -> None:
    """Verify condition carrying 'Primary' in name is classified as Secondary when acute admission driver exists."""
    mock_llm = MagicMock()
    classifier = PrimarySecondaryClassifier(mock_llm)

    acute_app = ContextAssessment(
        condition_id="app-1",
        diagnosis="Acute appendicitis",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Patient presented with acute right lower quadrant pain, admitted for acute appendicitis. Underwent urgent appendectomy.",
        reason="Acute condition occasioning admission.",
        treated_or_managed=True,
        monitored=True,
    )

    pri_htn = ContextAssessment(
        condition_id="htn-1",
        diagnosis="Primary hypertension",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.CHRONIC,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Past history of Primary hypertension, maintained on daily amlodipine during hospitalization.",
        reason="Chronic baseline condition monitored during stay.",
        treated_or_managed=True,
        monitored=True,
    )

    result, abstentions = classifier.classify_conditions(
        [acute_app, pri_htn],
        clinical_text="Admitted for acute appendicitis. History: Primary hypertension on amlodipine.",
    )

    assert result.has_unique_primary is True
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.diagnosis == "Acute appendicitis"
    assert result.primary_diagnosis.role == DiagnosisRole.PRIMARY

    # Primary hypertension must be SECONDARY, not PRIMARY
    sec_names = [s.diagnosis for s in result.secondary_diagnoses]
    assert "Primary hypertension" in sec_names
    for sec in result.secondary_diagnoses:
        if sec.diagnosis == "Primary hypertension":
            assert sec.role == DiagnosisRole.SECONDARY


def test_llm_confusion_guardrail_reclassifies_primary_hypertension_to_secondary() -> None:
    """If LLM erroneously assigns PRIMARY to 'Primary hypertension' over Acute MI, guardrail enforces UHDDS."""
    mock_llm = MagicMock()
    llm_response = """
    {
      "classifications": [
        {
          "diagnosis_id": "htn-1",
          "diagnosis": "Primary hypertension",
          "role": "PRIMARY",
          "classification_reason": "Contains Primary in name"
        },
        {
          "diagnosis_id": "mi-1",
          "diagnosis": "Acute myocardial infarction",
          "role": "SECONDARY",
          "classification_reason": "Treated with coronary stent"
        }
      ]
    }
    """
    mock_llm.invoke.return_value = llm_response
    mock_llm.generate.return_value = llm_response
    classifier = PrimarySecondaryClassifier(mock_llm)

    mi = ContextAssessment(
        condition_id="mi-1",
        diagnosis="Acute myocardial infarction",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Patient presented with crushing chest pain, admitted for acute STEMI. Underwent emergent PCI.",
        reason="Acute condition occasioning admission.",
        treated_or_managed=True,
    )
    htn = ContextAssessment(
        condition_id="htn-1",
        diagnosis="Primary hypertension",
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.CHRONIC,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        evidence="Known primary hypertension on lisinopril.",
        reason="Chronic comorbidity.",
        treated_or_managed=True,
    )

    result, _ = classifier.classify_conditions([mi, htn])
    assert result.has_unique_primary is True
    # Guardrail must demote Primary hypertension and make Acute MI the Primary diagnosis
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.diagnosis == "Acute myocardial infarction"
    assert result.primary_diagnosis.role == DiagnosisRole.PRIMARY


# ==============================================================================
# 3. MULTI-ROUND RE-CHECKING FOR ABSTENTIONS WITH EVIDENCE TESTS
# ==============================================================================


@pytest.fixture
def mini_icd_system() -> tuple[LocalICDCatalog, HybridICDRetriever]:
    catalog = LocalICDCatalog()
    records = [
        ICDCodeRecord(
            code="I10",
            unformatted_code="I10",
            description="Essential primary hypertension",
            is_valid_billable=True,
            category="I10",
            synonyms=["primary hypertension", "essential hypertension", "HTN"],
        ),
        ICDCodeRecord(
            code="K35.80",
            unformatted_code="K3580",
            description="Unspecified acute appendicitis",
            is_valid_billable=True,
            category="K35",
            synonyms=["acute appendicitis"],
        ),
    ]
    for r in records:
        catalog.add_record(r)
    catalog._initialized = True

    bm25 = BM25ICDRetriever()
    bm25.build_index(records)
    embeddings = FastLocalEmbeddings(dim=384)
    faiss_ret = FAISSICDRetriever(embeddings=embeddings)
    faiss_ret.build_index(records)

    hybrid = HybridICDRetriever(
        lexical_retriever=bm25,
        vector_retriever=faiss_ret,
        catalog=catalog,
        min_score_threshold=0.35,
    )
    return catalog, hybrid


def test_retrieve_with_retry_finds_code_via_normalized_variations(
    mini_icd_system: tuple[LocalICDCatalog, HybridICDRetriever],
) -> None:
    """Verify retrieve_with_retry resolves complex phrasing via normalization variations."""
    _, hybrid = mini_icd_system
    noisy_query = "K/C/O High blood pressure (primary) on Tab Telmisartan 40mg"
    results = hybrid.retrieve_with_retry(
        noisy_query,
        evidence="Patient has primary hypertension managed with medication",
        top_k=3,
    )
    assert len(results) > 0
    assert results[0].code == "I10"


def test_finalize_output_node_rechecks_abstentions_with_positive_evidence(
    monkeypatch: pytest.MonkeyPatch,
    mini_icd_system: tuple[LocalICDCatalog, HybridICDRetriever],
) -> None:
    """Verify finalize_output_node re-checks abstentions that carry positive evidence and promotes them."""
    catalog, hybrid = mini_icd_system
    import medical_coding.graph.nodes as nodes_mod
    monkeypatch.setattr(nodes_mod, "get_or_initialize_retrieval_system", lambda: (catalog, hybrid))

    # Construct pipeline state where an abstention exists with positive evidence
    abstention_with_ev = AbstentionRecord(
        diagnosis_id="htn-abstained",
        raw_term="K/C/O Primary hypertension on medication",
        reason=AbstentionReason.NO_MATCHING_ICD_CANDIDATE,
        detail="No matching code on first pass",
        stage=nodes_mod.PipelineStage.VALIDATION,
    )

    state = {
        "document_id": "test-doc-01",
        "raw_text": "History: K/C/O Primary hypertension on medication. Given amlodipine.",
        "validated_diagnoses": [],
        "classified_diagnoses": [],
        "abstentions": [abstention_with_ev],
        "ranked_selections": [],
        "candidate_pool": {},
    }

    out = finalize_output_node(state)
    res = out["final_result"]

    # The abstained condition with evidence must be recovered and promoted!
    coded_all = []
    if res.primary_diagnosis:
        coded_all.append(res.primary_diagnosis)
    coded_all.extend(res.secondary_diagnoses)

    assert any(c.database_code == "I10" for c in coded_all)
    # Abstention list must be cleared of the recovered condition
    assert not any(a.raw_term == "K/C/O Primary hypertension on medication" for a in res.abstentions)


# ==============================================================================
# 4. DUPLICATE AND REPETITIVE EVIDENCE CLEANING TESTS
# ==============================================================================


def test_deduplicate_evidence_spans_and_clean_repetition() -> None:
    """Verify that repetitive evidence strings, substrings, and bare headers are cleanly removed."""
    from medical_coding.utils.text import (
        clean_and_deduplicate_evidence_quote,
        deduplicate_evidence_spans,
    )

    # 1. Bare diagnosis header with narrative evidence -> header dropped
    raw_ev = "Acute cholecystitis | Patient was admitted with acute RUQ pain and fever, confirmed acute cholecystitis on ultrasound. | Underwent successful laparoscopic cholecystectomy."
    spans = deduplicate_evidence_spans(raw_ev, diagnosis_name="Acute cholecystitis")
    assert len(spans) == 2
    assert "Acute cholecystitis" not in spans
    assert "Patient was admitted with acute RUQ pain and fever, confirmed acute cholecystitis on ultrasound." in spans
    assert "Underwent successful laparoscopic cholecystectomy." in spans

    # 2. Identical repeats
    raw_ev2 = "Blood pressure maintained on lisinopril for hypertension. | Blood pressure maintained on lisinopril for hypertension."
    spans2 = deduplicate_evidence_spans(raw_ev2)
    assert len(spans2) == 1
    assert spans2[0] == "Blood pressure maintained on lisinopril for hypertension."

    # 3. Subsumed substring
    raw_ev3 = ["Patient presented with chest pain", "Patient presented with chest pain and shortness of breath"]
    spans3 = deduplicate_evidence_spans(raw_ev3)
    assert len(spans3) == 1
    assert spans3[0] == "Patient presented with chest pain and shortness of breath"

    # 4. Clean quote output
    clean_q = clean_and_deduplicate_evidence_quote(raw_ev, diagnosis_name="Acute cholecystitis")
    assert clean_q == "Patient was admitted with acute RUQ pain and fever, confirmed acute cholecystitis on ultrasound. | Underwent successful laparoscopic cholecystectomy."


def test_coding_result_deduplicates_secondaries_and_cleans_evidences() -> None:
    """Verify that CodingResult drops duplicate secondary diagnoses, drops secondaries matching primary, and cleans repetitive evidence."""
    from medical_coding.schemas.response import CodedDiagnosisResponse, CodingResult

    primary = CodedDiagnosisResponse(
        code="K81.0",
        description="Acute cholecystitis",
        role=DiagnosisRole.PRIMARY,
        acuity=Acuity.ACUTE,
        certainty=Certainty.CONFIRMED,
        evidence_quote="Acute cholecystitis | Patient admitted for Acute cholecystitis with fever.",
        confidence_score=0.95,
    )

    # Duplicate of primary (same code)
    dup_pri_sec = CodedDiagnosisResponse(
        code="K81.0",
        description="Acute cholecystitis",
        role=DiagnosisRole.SECONDARY,
        acuity=Acuity.ACUTE,
        certainty=Certainty.CONFIRMED,
        evidence_quote="Patient admitted for Acute cholecystitis with fever.",
        confidence_score=0.90,
    )

    # Two identical secondary entries with same code K80.80
    sec1 = CodedDiagnosisResponse(
        code="K80.80",
        description="Other cholelithiasis without obstruction",
        role=DiagnosisRole.SECONDARY,
        acuity=Acuity.CHRONIC,
        certainty=Certainty.CONFIRMED,
        evidence_quote="Gallstones noted on ultrasound.",
        confidence_score=0.75,
    )
    sec2 = CodedDiagnosisResponse(
        code="K80.80",
        description="Other cholelithiasis without obstruction",
        role=DiagnosisRole.SECONDARY,
        acuity=Acuity.CHRONIC,
        certainty=Certainty.CONFIRMED,
        evidence_quote="Gallstones noted on ultrasound. | Cholelithiasis visualized.",
        confidence_score=0.85,
    )

    # Valid different secondary
    sec_htn = CodedDiagnosisResponse(
        code="I10",
        description="Essential (primary) hypertension",
        role=DiagnosisRole.SECONDARY,
        acuity=Acuity.CHRONIC,
        certainty=Certainty.CONFIRMED,
        evidence_quote="Hypertension on lisinopril. | Hypertension on lisinopril.",
        confidence_score=0.90,
    )

    abstention_for_coded_cond = AbstentionRecord(
        diagnosis_id="chole-abstained",
        raw_term="Acute cholecystitis",  # Already coded in primary!
        reason=AbstentionReason.NO_MATCHING_ICD_CANDIDATE,
        detail="Old abstention record",
        stage=PipelineStage.VALIDATION,
    )

    abstention_valid = AbstentionRecord(
        diagnosis_id="rare-abstained",
        raw_term="Uncoded rare condition",
        reason=AbstentionReason.NO_MATCHING_ICD_CANDIDATE,
        detail="No matching code",
        stage=PipelineStage.VALIDATION,
    )

    result = CodingResult(
        document_id="doc-dedup-test",
        status=ExecutionStatus.SUCCESS,
        primary_diagnosis=primary,
        secondary_diagnoses=[dup_pri_sec, sec1, sec2, sec_htn],
        abstentions=[abstention_for_coded_cond, abstention_valid],
        procedures=["Laparoscopic cholecystectomy [CPT: 47600]", "Laparoscopic cholecystectomy"],
        processing_time_ms=10.0,
    )

    # 1. Primary evidence should be deduplicated (bare header removed)
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.evidence_quote == "Patient admitted for Acute cholecystitis with fever."

    # 2. Secondary list must NOT contain duplicate of primary
    assert not any(s.code == "K81.0" for s in result.secondary_diagnoses)

    # 3. K80.80 must appear exactly ONCE (deduplicated)
    k80_matches = [s for s in result.secondary_diagnoses if s.code == "K80.80"]
    assert len(k80_matches) == 1
    # Evidence merged cleanly
    assert "Gallstones noted on ultrasound." in k80_matches[0].evidence_quote
    assert "Cholelithiasis visualized." in k80_matches[0].evidence_quote
    # Confidence score is max (0.85)
    assert k80_matches[0].confidence_score == 0.85

    # 4. Hypertension repetitive evidence should be cleaned
    htn_match = next(s for s in result.secondary_diagnoses if s.code == "I10")
    assert htn_match.evidence_quote == "Hypertension on lisinopril."

    # 5. Abstention list must drop 'Acute cholecystitis' because it is coded
    assert len(result.abstentions) == 1
    assert result.abstentions[0].raw_term == "Uncoded rare condition"

    # 6. Procedures must be deduplicated
    assert len(result.procedures) == 1

    # 7. Formatted output must have no duplicates
    formatted = result.format_output()
    assert formatted.count("code: K80.80") == 1
    assert formatted.count("code: I10") == 1
    assert "Hypertension on lisinopril. | Hypertension on lisinopril." not in formatted

