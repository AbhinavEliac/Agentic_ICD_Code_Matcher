"""Pytest shared fixtures and test configuration."""

import pytest
from fastapi.testclient import TestClient

from medical_coding.api.app import create_app
from medical_coding.config.settings import Settings
from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.schemas.clinical import (
    ClassifiedDiagnosis,
    ContextualizedDiagnosis,
    EvidenceSnippet,
    ExtractedDiagnosis,
)
from medical_coding.schemas.enums import (
    Acuity,
    Certainty,
    DiagnosisRole,
    NegationStatus,
    Temporality,
)
from medical_coding.schemas.icd import ICDCandidate, ICDCodeRecord, RankedSelection


@pytest.fixture
def test_settings() -> Settings:
    """Provide isolated in-memory test settings."""
    return Settings(
        environment="test",
        debug=True,
        offline_mode=True,
        log_level="DEBUG",
        max_batch_concurrency=5,
    )


@pytest.fixture
def test_client(test_settings: Settings) -> TestClient:
    """FastAPI test client fixture."""
    app = create_app(settings=test_settings)
    return TestClient(app)


@pytest.fixture
def mock_catalog() -> LocalICDCatalog:
    """Provide an in-memory populated LocalICDCatalog for deterministic testing."""
    catalog = LocalICDCatalog()
    # Pre-populate sample verified codes
    records = [
        ICDCodeRecord(
            code="I50.21",
            unformatted_code="I5021",
            short_description="Acute systolic heart failure",
            long_description="Acute systolic (congestive) heart failure",
            is_valid_billable=True,
            category="I50",
            excludes1=["I50.22", "I50.23"],
        ),
        ICDCodeRecord(
            code="I50.9",
            unformatted_code="I509",
            short_description="Heart failure, unspecified",
            long_description="Heart failure, unspecified",
            is_valid_billable=True,
            category="I50",
        ),
        ICDCodeRecord(
            code="I50",
            unformatted_code="I50",
            short_description="Heart failure category header",
            long_description="Heart failure category header (non-billable)",
            is_valid_billable=False,  # Category header, not billable
            category="I50",
        ),
        ICDCodeRecord(
            code="E11.9",
            unformatted_code="E119",
            short_description="Type 2 diabetes mellitus without complications",
            long_description="Type 2 diabetes mellitus without complications",
            is_valid_billable=True,
            category="E11",
        ),
    ]
    for r in records:
        catalog._records_by_clean[r.unformatted_code] = r
        catalog._records_by_formatted[r.code] = r
    catalog._initialized = True
    return catalog


@pytest.fixture
def sample_classified_primary() -> ClassifiedDiagnosis:
    """Sample primary active condition fixture."""
    evidence = EvidenceSnippet(
        quote="Patient admitted for acute exacerbation of systolic heart failure.",
        source_section="Discharge Diagnosis",
    )
    extracted = ExtractedDiagnosis(
        diagnosis_id="diag-001",
        raw_term="acute systolic heart failure",
        evidence=evidence,
        extraction_confidence=0.98,
    )
    context = ContextualizedDiagnosis(
        diagnosis_id=extracted.diagnosis_id,
        raw_term=extracted.raw_term,
        evidence=evidence,
        extraction_confidence=0.98,
        negation=NegationStatus.AFFIRMATIVE,
        temporality=Temporality.CURRENT,
        certainty=Certainty.CONFIRMED,
        acuity=Acuity.ACUTE,
        clinical_justification="Patient required immediate diuresis during stay.",
    )
    return ClassifiedDiagnosis(
        diagnosis_id=extracted.diagnosis_id,
        raw_term=extracted.raw_term,
        context=context,
        role=DiagnosisRole.PRIMARY,
        is_billable_candidate=True,
        classification_reason="Chief complaint occasioning hospital admission.",
        primary_justification="Chief clinical condition treated throughout hospitalization.",
    )


@pytest.fixture
def sample_ranked_selection() -> RankedSelection:
    """Sample ranked candidate selection fixture."""
    candidate = ICDCandidate(
        code="I50.21",
        description="Acute systolic (congestive) heart failure",
        retrieval_score=0.95,
        retrieval_method="hybrid",
        is_valid_billable=True,
    )
    return RankedSelection(
        diagnosis_id="diag-001",
        raw_term="acute systolic heart failure",
        selected_candidate=candidate,
        ranking_score=0.95,
        selection_justification="Matches clinical acuity and cardiac chamber specification.",
        candidate_pool=[candidate],
        decision="ACCEPTED",
    )
