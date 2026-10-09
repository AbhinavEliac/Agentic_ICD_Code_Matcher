"""SQLAlchemy models for persistent storage of clinical documents and ICD-10 coding results."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base class for medical coding database entities."""

    pass


class ProcessedDocument(Base):
    """Database representation of an ingested and coded clinical document encounter."""

    __tablename__ = "processed_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), default="clinical_note.txt", nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), default="text", nullable=False)  # "pdf", "text", "batch_pdf"
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="SUCCESS", index=True, nullable=False)

    # Primary Diagnosis snapshot
    primary_code: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    primary_description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    primary_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Counts & Metrics
    secondary_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    abstention_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Surgical Oncology & Procedures
    procedures_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    operative_findings_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    oncology_context_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    # Metadata & Auditing
    metadata_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    diagnoses: Mapped[list["DiagnosisRecord"]] = relationship(
        "DiagnosisRecord",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DiagnosisRecord.id",
    )
    abstentions: Mapped[list["AbstentionRecordModel"]] = relationship(
        "AbstentionRecordModel",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="AbstentionRecordModel.id",
    )
    validation_checks: Mapped[list["ValidationCheckModel"]] = relationship(
        "ValidationCheckModel",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ValidationCheckModel.id",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert document record to dictionary representation."""
        import json

        meta = {}
        try:
            meta = json.loads(self.metadata_json) if self.metadata_json else {}
        except Exception:
            pass

        procedures = []
        if getattr(self, "procedures_json", None):
            try:
                procedures = json.loads(self.procedures_json)
            except Exception:
                procedures = []

        operative_findings = []
        if getattr(self, "operative_findings_json", None):
            try:
                operative_findings = json.loads(self.operative_findings_json)
            except Exception:
                operative_findings = []

        oncology_context = {}
        if getattr(self, "oncology_context_json", None):
            try:
                oncology_context = json.loads(self.oncology_context_json)
            except Exception:
                oncology_context = {}

        return {
            "id": self.id,
            "document_id": self.document_id,
            "filename": self.filename,
            "source_type": self.source_type,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "raw_text": self.raw_text,
            "status": self.status,
            "primary_code": self.primary_code,
            "primary_description": self.primary_description,
            "primary_confidence": self.primary_confidence,
            "secondary_count": self.secondary_count,
            "abstention_count": self.abstention_count,
            "procedures": procedures,
            "operative_findings": operative_findings,
            "oncology_context": oncology_context,
            "processing_time_ms": self.processing_time_ms,
            "metadata": meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "abstentions": [a.to_dict() for a in self.abstentions],
            "validation_checks": [v.to_dict() for v in self.validation_checks],
        }


class DiagnosisRecord(Base):
    """Database representation of an ICD-10 diagnosis assigned to an encounter."""

    __tablename__ = "diagnosis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("processed_documents.document_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="SECONDARY", index=True, nullable=False)  # "PRIMARY", "SECONDARY"
    acuity: Mapped[str] = mapped_column(String(32), default="UNSPECIFIED", nullable=False)
    certainty: Mapped[str] = mapped_column(String(32), default="CONFIRMED", nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    is_terminal_billable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    evidence_quote: Mapped[str] = mapped_column(Text, default="", nullable=False)
    icd10cm: Mapped[str | None] = mapped_column(String(32), nullable=True)
    icdo: Mapped[str | None] = mapped_column(String(32), nullable=True)
    cpt: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    document: Mapped["ProcessedDocument"] = relationship(
        "ProcessedDocument",
        back_populates="diagnoses",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnosis record to dictionary representation."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "code": self.code,
            "icd10cm": self.icd10cm,
            "icdo": self.icdo,
            "cpt": self.cpt,
            "description": self.description,
            "role": self.role,
            "acuity": self.acuity,
            "certainty": self.certainty,
            "confidence_score": self.confidence_score,
            "is_terminal_billable": self.is_terminal_billable,
            "evidence_quote": self.evidence_quote,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AbstentionRecordModel(Base):
    """Database representation of an explicit abstention or excluded condition."""

    __tablename__ = "abstention_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("processed_documents.document_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    raw_term: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reason: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    stage: Mapped[str] = mapped_column(String(64), default="EVALUATE_CONFIDENCE", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    document: Mapped["ProcessedDocument"] = relationship(
        "ProcessedDocument",
        back_populates="abstentions",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert abstention record to dictionary representation."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "raw_term": self.raw_term,
            "reason": self.reason,
            "detail": self.detail,
            "stage": self.stage,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ValidationCheckModel(Base):
    """Database representation of a deterministic rule audit check."""

    __tablename__ = "validation_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("processed_documents.document_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    diagnosis_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rule_name: Mapped[str] = mapped_column(String(64), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    details: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    document: Mapped["ProcessedDocument"] = relationship(
        "ProcessedDocument",
        back_populates="validation_checks",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert validation check to dictionary representation."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "diagnosis_code": self.diagnosis_code,
            "rule_name": self.rule_name,
            "passed": self.passed,
            "details": self.details,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PipelineThreadModel(Base):
    """Database representation of an end-to-end pipeline execution thread for history and auditing."""

    __tablename__ = "pipeline_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    document_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="RUNNING", index=True, nullable=False)
    current_step_name: Mapped[str] = mapped_column(String(64), default="Starting", nullable=False)
    current_step_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_steps: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    failed_step: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_source: Mapped[str] = mapped_column(String(255), default="clinical_note.txt", nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list["PipelineExecutionStepModel"]] = relationship(
        "PipelineExecutionStepModel",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="PipelineExecutionStepModel.step_index",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline thread record to dictionary representation."""
        import json

        result_data = None
        if self.result_json:
            try:
                result_data = json.loads(self.result_json)
            except Exception:
                result_data = None

        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "document_id": self.document_id,
            "status": self.status,
            "current_step_name": self.current_step_name,
            "current_step_index": self.current_step_index,
            "total_steps": self.total_steps,
            "progress_pct": self.progress_pct,
            "failed_step": self.failed_step,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "input_source": self.input_source,
            "raw_text": self.raw_text,
            "duration_ms": self.duration_ms,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "result": result_data,
            "steps": [s.to_dict() for s in self.steps],
        }


class PipelineExecutionStepModel(Base):
    """Database representation of an individual execution stage/node within a pipeline thread."""

    __tablename__ = "pipeline_execution_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("pipeline_threads.thread_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    step_name: Mapped[str] = mapped_column(String(64), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="SUCCESS", nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    log_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    details_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    thread: Mapped["PipelineThreadModel"] = relationship(
        "PipelineThreadModel",
        back_populates="steps",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert step record to dictionary representation."""
        import json

        details = {}
        if self.details_json:
            try:
                details = json.loads(self.details_json)
            except Exception:
                details = {}

        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "step_name": self.step_name,
            "step_index": self.step_index,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "log_message": self.log_message,
            "details": details,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }
