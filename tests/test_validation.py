"""Unit tests for deterministic ICD validation and rule enforcement."""

from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.schemas.clinical import ClassifiedDiagnosis
from medical_coding.schemas.enums import AbstentionReason, DiagnosisRole
from medical_coding.schemas.icd import ICDCandidate, RankedSelection
from medical_coding.validation.deterministic import DeterministicValidator


def test_validation_passes_valid_billable_code(
    mock_catalog: LocalICDCatalog,
    sample_classified_primary: ClassifiedDiagnosis,
    sample_ranked_selection: RankedSelection,
) -> None:
    validator = DeterministicValidator(catalog=mock_catalog)
    validated, abstention = validator.validate_code(
        condition=sample_classified_primary,
        selection=sample_ranked_selection,
    )

    assert abstention is None
    assert validated is not None
    assert validated.code == "I50.21"
    assert validated.role == DiagnosisRole.PRIMARY
    assert len(validated.checks) == 5
    assert all(c.passed for c in validated.checks)


def test_validation_rejects_nonexistent_code(
    mock_catalog: LocalICDCatalog,
    sample_classified_primary: ClassifiedDiagnosis,
) -> None:
    validator = DeterministicValidator(catalog=mock_catalog)
    fake_candidate = ICDCandidate(
        code="Z99.9999",  # Does not exist in local catalog
        description="Fictitious condition",
        retrieval_score=0.99,
        retrieval_method="hybrid",
        is_valid_billable=True,
    )
    selection = RankedSelection(
        diagnosis_id=sample_classified_primary.diagnosis_id,
        raw_term=sample_classified_primary.raw_term,
        selected_candidate=fake_candidate,
        ranking_score=0.99,
        selection_justification="Candidate tested for existence.",
        decision="ACCEPTED",
    )

    validated, abstention = validator.validate_code(
        condition=sample_classified_primary,
        selection=selection,
    )
    assert validated is None
    assert abstention is not None
    assert abstention.reason == AbstentionReason.INVALID_ICD_CODE


def test_validation_rejects_category_header_requiring_specificity(
    mock_catalog: LocalICDCatalog,
    sample_classified_primary: ClassifiedDiagnosis,
) -> None:
    validator = DeterministicValidator(catalog=mock_catalog)
    # I50 is non-billable category header in our mock catalog
    header_candidate = ICDCandidate(
        code="I50",
        description="Heart failure category header",
        retrieval_score=0.85,
        retrieval_method="hybrid",
        is_valid_billable=False,
    )
    selection = RankedSelection(
        diagnosis_id=sample_classified_primary.diagnosis_id,
        raw_term=sample_classified_primary.raw_term,
        selected_candidate=header_candidate,
        ranking_score=0.85,
        selection_justification="Category header selected.",
        decision="ACCEPTED",
    )

    validated, abstention = validator.validate_code(
        condition=sample_classified_primary,
        selection=selection,
    )
    assert validated is None
    assert abstention is not None
    assert abstention.reason == AbstentionReason.SPECIFICITY_REQUIRED


def test_validation_detects_multiple_primaries_and_abstains(
    mock_catalog: LocalICDCatalog,
    sample_classified_primary: ClassifiedDiagnosis,
    sample_ranked_selection: RankedSelection,
) -> None:
    validator = DeterministicValidator(catalog=mock_catalog)

    # Create a second conflicting primary
    cond2 = sample_classified_primary.model_copy(deep=True)
    cond2.diagnosis_id = "diag-002"
    cond2.raw_term = "type 2 diabetes"
    cand2 = ICDCandidate(
        code="E11.9",
        description="Type 2 diabetes mellitus",
        retrieval_score=0.91,
        retrieval_method="hybrid",
        is_valid_billable=True,
    )
    sel2 = RankedSelection(
        diagnosis_id=cond2.diagnosis_id,
        raw_term=cond2.raw_term,
        selected_candidate=cand2,
        ranking_score=0.91,
        selection_justification="Candidate matching diabetes.",
        decision="ACCEPTED",
    )

    validated_list, abstentions = validator.validate_encounter_set(
        conditions=[sample_classified_primary, cond2],
        selections=[sample_ranked_selection, sel2],
    )

    # Invariant: Must abstain multiple conflicting primaries
    assert len(validated_list) == 0
    assert any(a.reason == AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY for a in abstentions)
