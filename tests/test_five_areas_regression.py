"""Regression test suite for the Five Clinical Improvement Areas on noise-fix branch.

Validates:
Area 1: Complete Clinical Inventory Generation (compound phrases, decomposition, laterality preservation).
Area 2: First-Class Assertion, Negation & Entity Type Decisions (CMS Guideline I.B.4 integral symptom suppression, entity typing).
Area 3: Semantic Code Validation Beyond Excel Existence (CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY, bleeding & laterality attribute checks).
Area 4: Decoupled UHDDS Role Classification from Code Ranking & Clean Output Separation.
Area 5: Trustworthy Evidence Provenance, Distinct Document IDs, and Multi-Metric Telemetry.
"""

import pytest

from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor
from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.graph.pipeline import process_clinical_document_sync
from medical_coding.schemas.clinical import ClassifiedDiagnosis
from medical_coding.schemas.enums import ClinicalEntityType, DiagnosisRole, NegationStatus, Certainty
from medical_coding.schemas.icd import ICDCandidate, RankedSelection
from medical_coding.validation.clinical_gate import HardClinicalCandidateGate
from medical_coding.validation.deterministic import DeterministicValidator
from medical_coding.validation.reverse_attributes import ReverseAttributeChecker


class TestFiveAreasRegression:
    """Test suite covering all 5 improvement areas."""

    # -------------------------------------------------------------------------
    # AREA 1: Complete Clinical Inventory Generation
    # -------------------------------------------------------------------------
    def test_area_1_compound_decomposition_and_etiology_preservation(self) -> None:
        """Verify compound expressions like 'X secondary to Y' extract both conditions."""
        extractor = EvidenceFirstFactExtractor()
        text = """
        DISCHARGE SUMMARY
        DISCHARGE DIAGNOSES:
        1. Acute kidney injury secondary to dehydration - Patient managed with IV fluid resuscitation.
        2. Essential hypertension - Maintained on home amlodipine.
        """
        state = extractor.extract_clinical_state(text, "DOC-AREA1-01")
        all_terms = [c.normalized_diagnosis.lower() for c in state.all_candidates]

        # Both the acute manifestation and the underlying etiology must be extracted
        assert any("acute kidney injury" in t for t in all_terms)
        assert any("dehydration" in t for t in all_terms)
        assert any("hypertension" in t for t in all_terms)

    def test_area_1_laterality_deduplication_preservation(self) -> None:
        """Verify distinct anatomical lateralities (Right vs Left) are NOT collapsed into one candidate."""
        extractor = EvidenceFirstFactExtractor()
        text = """
        DISCHARGE SUMMARY
        DISCHARGE DIAGNOSES:
        1. Right breast carcinoma - Underwent right lumpectomy.
        2. Left breast fibroadenoma - Excision biopsy performed.
        """
        state = extractor.extract_clinical_state(text, "DOC-AREA1-02")
        terms = [c.normalized_diagnosis.lower() for c in state.all_candidates]

        assert any("right" in t for t in terms)
        assert any("left" in t for t in terms)
        # Verify 2 distinct candidates
        assert len(state.all_candidates) >= 2

    # -------------------------------------------------------------------------
    # AREA 2: First-Class Assertion, Negation & Entity Type Decisions
    # -------------------------------------------------------------------------
    def test_area_2_entity_type_classification(self) -> None:
        """Verify entity classifier properly distinguishes symptoms, organisms, and lab findings."""
        assert HardClinicalCandidateGate.classify_entity_type("dyspnea") == ClinicalEntityType.SYMPTOM
        assert HardClinicalCandidateGate.classify_entity_type("chest pain") == ClinicalEntityType.SYMPTOM
        assert HardClinicalCandidateGate.classify_entity_type("flank pain") == ClinicalEntityType.SYMPTOM
        assert HardClinicalCandidateGate.classify_entity_type("Escherichia coli") == ClinicalEntityType.CLINICAL_FINDING
        assert HardClinicalCandidateGate.classify_entity_type("leukocytosis") == ClinicalEntityType.LAB_RESULT
        assert HardClinicalCandidateGate.classify_entity_type("no acute complications") == ClinicalEntityType.NEGATED_CONDITION

    def test_area_2_cms_guideline_integral_symptom_suppression(self) -> None:
        """Verify CMS Guideline I.B.4: Integral symptoms of confirmed primary are suppressed from secondary coding."""
        extractor = EvidenceFirstFactExtractor()
        text = """
        DISCHARGE SUMMARY
        CHIEF COMPLAINT: Severe shortness of breath and peripheral edema.
        DISCHARGE DIAGNOSES:
        1. Acute systolic heart failure - Admitted with decompensation. Echo EF 20%. Treated with IV furosemide diuresis.
        2. Shortness of breath - Accompanied heart failure decompensation.
        3. Peripheral edema - Resolved with diuresis.
        4. Type 2 diabetes mellitus - Controlled on metformin.
        """
        state = extractor.extract_clinical_state(text, "DOC-AREA2-01")

        # Primary must be heart failure
        assert state.primary_diagnosis is not None
        assert "heart failure" in state.primary_diagnosis.normalized_diagnosis.lower()

        # Integral symptoms (shortness of breath, peripheral edema) must NOT be in secondary_diagnoses
        sec_names = [s.normalized_diagnosis.lower() for s in state.secondary_diagnoses]
        assert not any("shortness of breath" in s for s in sec_names)
        assert not any("peripheral edema" in s for s in sec_names)
        assert any("diabetes" in s for s in sec_names)

    # -------------------------------------------------------------------------
    # AREA 3: Semantic Code Validation Beyond Excel Existence
    # -------------------------------------------------------------------------
    def test_area_3_reverse_attributes_rejects_unsupported_bleeding(self) -> None:
        """Verify ReverseAttributeChecker rejects code specifying bleeding when evidence states without bleeding."""
        code_desc = "Acute gastritis with bleeding"
        evidence_text = "Patient diagnosed with acute gastritis. Pertinent negatives: no hematemesis, melena, or bleeding."
        is_valid, unsupp = ReverseAttributeChecker.validate_code_attributes(
            code_description=code_desc,
            evidence_text=evidence_text,
            diagnosis_term="Acute gastritis",
        )
        assert not is_valid
        assert "bleeding" in unsupp

    def test_area_3_deterministic_validator_realigns_unsupported_attributes(self) -> None:
        """Verify DeterministicValidator realigns to 'without bleeding' candidate if available."""
        from medical_coding.schemas.icd import ICDCodeRecord

        catalog = LocalICDCatalog()
        catalog.add_record(
            ICDCodeRecord(code="K29.01", description="Acute gastritis with bleeding", is_valid_billable=True, category="K29")
        )
        catalog.add_record(
            ICDCodeRecord(code="K29.00", description="Acute gastritis without bleeding", is_valid_billable=True, category="K29")
        )
        catalog._initialized = True

        from medical_coding.schemas.clinical import ContextualizedDiagnosis, EvidenceSnippet
        from medical_coding.schemas.enums import Acuity, Temporality

        evidence_str = "Patient admitted with acute gastritis. Pertinent negatives: no bleeding, melena, or hematemesis."
        validator = DeterministicValidator(catalog=catalog)

        cand_with_bleed = ICDCandidate(
            code="K29.01",
            description="Acute gastritis with bleeding",
            is_valid_billable=True,
            retrieval_score=0.9,
            retrieval_method="hybrid",
        )
        cand_without_bleed = ICDCandidate(
            code="K29.00",
            description="Acute gastritis without bleeding",
            is_valid_billable=True,
            retrieval_score=0.85,
            retrieval_method="hybrid",
        )

        cond = ClassifiedDiagnosis(
            diagnosis_id="diag-1",
            raw_term="Acute gastritis",
            context=ContextualizedDiagnosis(
                diagnosis_id="diag-1",
                raw_term="Acute gastritis",
                evidence=EvidenceSnippet(quote=evidence_str, source_section="DISCHARGE_DIAGNOSES"),
                extraction_confidence=1.0,
                acuity=Acuity.ACUTE,
                certainty=Certainty.CONFIRMED,
                temporality=Temporality.CURRENT,
                clinical_justification="Acute gastritis",
            ),
            role=DiagnosisRole.PRIMARY,
            is_billable_candidate=True,
            classification_reason="Admission occasioning diagnosis",
        )

        selection = RankedSelection(
            diagnosis_id="diag-1",
            raw_term="Acute gastritis",
            selected_code="K29.01",
            selected_description="Acute gastritis with bleeding",
            selected_candidate=cand_with_bleed,
            candidate_pool=[cand_with_bleed, cand_without_bleed],
            ranking_score=0.9,
            supporting_evidence=[evidence_str],
        )

        validated, abst = validator.validate_code(cond, selection)
        # Must realign to K29.00 (without bleeding)
        assert validated is not None
        assert validated.code == "K29.00"

    # -------------------------------------------------------------------------
    # AREA 4: Decouple UHDDS Role from Code Ranking & Clean Separation
    # -------------------------------------------------------------------------
    def test_area_4_secondary_section_never_carries_primary_diagnosis(self) -> None:
        """Verify final output never repeats the primary diagnosis inside the secondary diagnoses list."""
        text = """
        DISCHARGE SUMMARY
        CHIEF COMPLAINT: Fever and severe flank pain.
        DISCHARGE DIAGNOSES:
        1. Acute emphysematous pyelonephritis - Admitted and underwent DJ stenting.
        2. Essential hypertension - Maintained on amlodipine.
        HOSPITAL COURSE:
        Patient was admitted for acute pyelonephritis. Responded to antibiotic therapy.
        """
        res = process_clinical_document_sync(source=text, document_id="DOC-AREA4-01")

        assert res.primary_diagnosis is not None
        pri_desc = (res.primary_diagnosis.description or "").lower()
        assert "pyelonephritis" in pri_desc or "kidney" in pri_desc

        # Secondary section must not duplicate pyelonephritis
        for sec in res.secondary_diagnoses:
            sec_desc = (sec.description or "").lower()
            assert "pyelonephritis" not in sec_desc
            if res.primary_diagnosis.code:
                assert sec.code != res.primary_diagnosis.code

    # -------------------------------------------------------------------------
    # AREA 5: Trustworthy Provenance & Distinct Document IDs
    # -------------------------------------------------------------------------
    def test_area_5_distinct_document_ids_in_consecutive_runs(self) -> None:
        """Verify pipeline preserves unique document identifiers across consecutive runs."""
        text_1 = "DISCHARGE SUMMARY\nDISCHARGE DIAGNOSES:\n1. Essential hypertension - On lisinopril."
        text_2 = "DISCHARGE SUMMARY\nDISCHARGE DIAGNOSES:\n1. Type 2 diabetes mellitus - On metformin."

        res_1 = process_clinical_document_sync(source=text_1, document_id="ENC-CUSTOM-ALPHA")
        res_2 = process_clinical_document_sync(source=text_2, document_id="ENC-CUSTOM-BETA")

        assert res_1.document_id == "ENC-CUSTOM-ALPHA"
        assert res_2.document_id == "ENC-CUSTOM-BETA"
        assert res_1.document_id != res_2.document_id
