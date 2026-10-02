"""Unit tests for SQLAlchemy database models, connection, and repository operations."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from medical_coding.database.models import Base
from medical_coding.database.repository import MedicalCodingRepository
from medical_coding.schemas.enums import (
    AbstentionReason,
    Acuity,
    Certainty,
    DiagnosisRole,
    ExecutionStatus,
    PipelineStage,
)
from medical_coding.schemas.response import CodedDiagnosisResponse, CodingResult
from medical_coding.schemas.validation import AbstentionRecord


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> MedicalCodingRepository:
    """Create an isolated SQLite database repository for testing."""
    test_db = tmp_path / "test_medical_coding.db"
    test_db_url = f"sqlite:///{test_db.as_posix()}"

    # Override get_engine to use our temporary sqlite db
    test_engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    test_session_factory = sessionmaker(bind=test_engine, expire_on_commit=False)

    monkeypatch.setattr("medical_coding.database.connection._ENGINE", test_engine)
    monkeypatch.setattr("medical_coding.database.connection._SESSION_FACTORY", test_session_factory)

    return MedicalCodingRepository()


def test_save_and_retrieve_document(repo: MedicalCodingRepository) -> None:
    """Test saving a full CodingResult with primary, secondary, and abstention records."""
    result = CodingResult(
        document_id="ENC-TEST-001",
        status=ExecutionStatus.SUCCESS,
        primary_diagnosis=CodedDiagnosisResponse(
            code="I50.21",
            description="Acute systolic heart failure",
            role=DiagnosisRole.PRIMARY,
            acuity=Acuity.ACUTE,
            certainty=Certainty.CONFIRMED,
            evidence_quote="admitted with acute systolic heart failure",
            confidence_score=0.95,
            is_terminal_billable=True,
        ),
        secondary_diagnoses=[
            CodedDiagnosisResponse(
                code="I10",
                description="Essential primary hypertension",
                role=DiagnosisRole.SECONDARY,
                acuity=Acuity.CHRONIC,
                certainty=Certainty.CONFIRMED,
                evidence_quote="history of essential hypertension",
                confidence_score=0.88,
                is_terminal_billable=True,
            )
        ],
        abstentions=[
            AbstentionRecord(
                raw_term="chest pain",
                reason=AbstentionReason.EXCLUDED_BY_NEGATION,
                detail="Patient explicitly denies chest pain.",
                stage=PipelineStage.CONTEXT_ANALYSIS,
            )
        ],
        processing_time_ms=142.5,
        models_used={"llm": "mock", "retriever": "hybrid"},
    )

    raw_text = "Patient was admitted with acute systolic heart failure. History of essential hypertension. Denies chest pain."
    saved = repo.save_coding_result(
        result=result,
        raw_text=raw_text,
        filename="discharge_summary.pdf",
        source_type="pdf",
        page_count=2,
        file_size_bytes=4096,
        metadata={"department": "Cardiology"},
    )

    assert saved["document_id"] == "ENC-TEST-001"
    assert saved["status"] == "SUCCESS"
    assert saved["primary_code"] == "I50.21"
    assert saved["secondary_count"] == 1
    assert saved["abstention_count"] == 1
    assert saved["page_count"] == 2
    assert saved["metadata"]["department"] == "Cardiology"

    # Query back by ID
    retrieved = repo.get_document_by_id("ENC-TEST-001")
    assert retrieved is not None
    assert retrieved["document_id"] == "ENC-TEST-001"
    assert len(retrieved["diagnoses"]) == 2
    assert len(retrieved["abstentions"]) == 1
    assert retrieved["diagnoses"][0]["code"] == "I50.21"
    assert retrieved["diagnoses"][0]["role"] == "PRIMARY"
    assert retrieved["diagnoses"][1]["code"] == "I10"
    assert retrieved["diagnoses"][1]["role"] == "SECONDARY"
    assert retrieved["abstentions"][0]["reason"] == "EXCLUDED_BY_NEGATION"


def test_list_and_filter_documents(repo: MedicalCodingRepository) -> None:
    """Test searching and filtering stored documents."""
    res1 = CodingResult(
        document_id="ENC-001",
        status=ExecutionStatus.SUCCESS,
        primary_diagnosis=CodedDiagnosisResponse(
            code="I50.21",
            description="Acute systolic heart failure",
            role=DiagnosisRole.PRIMARY,
            acuity=Acuity.ACUTE,
            certainty=Certainty.CONFIRMED,
            evidence_quote="acute systolic heart failure",
            confidence_score=0.92,
        ),
        processing_time_ms=100.0,
    )
    res2 = CodingResult(
        document_id="ENC-002",
        status=ExecutionStatus.PARTIAL_SUCCESS,
        primary_diagnosis=CodedDiagnosisResponse(
            code="E11.9",
            description="Type 2 diabetes mellitus",
            role=DiagnosisRole.PRIMARY,
            acuity=Acuity.CHRONIC,
            certainty=Certainty.CONFIRMED,
            evidence_quote="type 2 diabetes",
            confidence_score=0.85,
        ),
        processing_time_ms=80.0,
    )

    repo.save_coding_result(res1, raw_text="Cardiology patient", filename="cardio.pdf")
    repo.save_coding_result(res2, raw_text="Endocrinology patient", filename="endo.pdf")

    # List all
    all_docs = repo.list_documents()
    assert len(all_docs) == 2

    # Filter by status
    success_docs = repo.list_documents(status="SUCCESS")
    assert len(success_docs) == 1
    assert success_docs[0]["document_id"] == "ENC-001"

    # Filter by search term
    search_docs = repo.list_documents(search="Cardiology")
    assert len(search_docs) == 1
    assert search_docs[0]["document_id"] == "ENC-001"

    # Filter by ICD code
    code_docs = repo.list_documents(code_filter="E11")
    assert len(code_docs) == 1
    assert code_docs[0]["document_id"] == "ENC-002"


def test_analytics_summary(repo: MedicalCodingRepository) -> None:
    """Test calculation of analytics aggregates and quality indicators."""
    res = CodingResult(
        document_id="ENC-ANALYTICS",
        status=ExecutionStatus.SUCCESS,
        primary_diagnosis=CodedDiagnosisResponse(
            code="I50.21",
            description="Acute systolic heart failure",
            role=DiagnosisRole.PRIMARY,
            acuity=Acuity.ACUTE,
            certainty=Certainty.CONFIRMED,
            evidence_quote="heart failure",
            confidence_score=0.9,
            is_terminal_billable=True,
        ),
        secondary_diagnoses=[
            CodedDiagnosisResponse(
                code="I10",
                description="Essential primary hypertension",
                role=DiagnosisRole.SECONDARY,
                acuity=Acuity.CHRONIC,
                certainty=Certainty.CONFIRMED,
                evidence_quote="hypertension",
                confidence_score=0.9,
                is_terminal_billable=True,
            )
        ],
        abstentions=[
            AbstentionRecord(
                raw_term="unspecified symptom",
                reason=AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
                detail="Low confidence",
                stage=PipelineStage.RANKING,
            )
        ],
        processing_time_ms=120.0,
    )
    repo.save_coding_result(res, raw_text="Note content")

    stats = repo.get_analytics_summary()
    assert stats["total_documents"] == 1
    assert stats["total_diagnoses"] == 2
    assert stats["billable_count"] == 2
    assert stats["billable_ratio_percent"] == 100.0
    assert stats["average_processing_time_ms"] == 120.0
    assert stats["status_breakdown"]["SUCCESS"] == 1
    assert len(stats["top_primary_codes"]) == 1
    assert stats["top_primary_codes"][0]["code"] == "I50.21"


def test_delete_document_cascades(repo: MedicalCodingRepository) -> None:
    """Test deleting a document cascades and deletes related diagnoses and abstentions."""
    res = CodingResult(
        document_id="ENC-DELETE",
        status=ExecutionStatus.SUCCESS,
        primary_diagnosis=CodedDiagnosisResponse(
            code="I50.21",
            description="Acute systolic heart failure",
            role=DiagnosisRole.PRIMARY,
            acuity=Acuity.ACUTE,
            certainty=Certainty.CONFIRMED,
            evidence_quote="heart failure",
            confidence_score=0.9,
        ),
        abstentions=[
            AbstentionRecord(
                raw_term="test",
                reason=AbstentionReason.EXCLUDED_BY_NEGATION,
                detail="test detail",
                stage=PipelineStage.CONTEXT_ANALYSIS,
            )
        ],
        processing_time_ms=50.0,
    )
    repo.save_coding_result(res, raw_text="Content")

    assert repo.get_document_by_id("ENC-DELETE") is not None
    deleted = repo.delete_document("ENC-DELETE")
    assert deleted is True
    assert repo.get_document_by_id("ENC-DELETE") is None


def test_export_documents_df(repo: MedicalCodingRepository) -> None:
    """Test exporting documents as a DataFrame."""
    res = CodingResult(
        document_id="ENC-DF",
        status=ExecutionStatus.SUCCESS,
        primary_diagnosis=CodedDiagnosisResponse(
            code="I50.21",
            description="Acute systolic heart failure",
            role=DiagnosisRole.PRIMARY,
            acuity=Acuity.ACUTE,
            certainty=Certainty.CONFIRMED,
            evidence_quote="heart failure",
            confidence_score=0.9,
        ),
        processing_time_ms=50.0,
    )
    repo.save_coding_result(res, raw_text="Content", filename="doc.pdf")

    df = repo.get_documents_df()
    assert len(df) == 1
    assert df.iloc[0]["document_id"] == "ENC-DF"
    assert df.iloc[0]["primary_code"] == "I50.21"
