"""Repository functions for saving, querying, and analyzing clinical coding records."""

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import selectinload

from medical_coding.database.connection import get_db_session, init_db
from medical_coding.database.models import (
    AbstentionRecordModel,
    DiagnosisRecord,
    PipelineExecutionStepModel,
    PipelineThreadModel,
    ProcessedDocument,
    ValidationCheckModel,
)
from medical_coding.schemas.response import CodingResult
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalCodingRepository:
    """Repository managing CRUD operations and analytics for medical coding documents."""

    def __init__(self) -> None:
        init_db()

    def save_coding_result(
        self,
        result: CodingResult,
        raw_text: str = "",
        filename: str = "clinical_note.txt",
        source_type: str = "text",
        page_count: int = 1,
        file_size_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a complete CodingResult and document payload to the database."""
        meta_dict = metadata or {}
        if result.metadata:
            meta_dict.update(result.metadata)

        char_cnt = len(raw_text)
        word_cnt = len(raw_text.split())

        with get_db_session() as session:
            # Check if document_id already exists to prevent duplicate key constraint
            existing = session.execute(
                select(ProcessedDocument).where(ProcessedDocument.document_id == result.document_id)
            ).scalar_one_or_none()

            if existing:
                # Update existing record
                doc = existing
                doc.filename = filename
                doc.source_type = source_type
                doc.file_size_bytes = file_size_bytes
                doc.page_count = page_count
                doc.word_count = word_cnt
                doc.char_count = char_cnt
                doc.raw_text = raw_text
                doc.status = result.status.value if hasattr(result.status, "value") else str(result.status)
                doc.primary_code = result.primary_diagnosis.code if result.primary_diagnosis else None
                doc.primary_description = (
                    result.primary_diagnosis.description if result.primary_diagnosis else None
                )
                doc.primary_confidence = (
                    result.primary_diagnosis.confidence_score if result.primary_diagnosis else None
                )
                doc.secondary_count = len(result.secondary_diagnoses)
                doc.abstention_count = len(result.abstentions)
                doc.processing_time_ms = result.processing_time_ms
                doc.metadata_json = json.dumps(meta_dict)

                # Clear previous diagnoses, abstentions, checks for clean re-save
                doc.diagnoses.clear()
                doc.abstentions.clear()
                doc.validation_checks.clear()
            else:
                doc = ProcessedDocument(
                    document_id=result.document_id,
                    filename=filename,
                    source_type=source_type,
                    file_size_bytes=file_size_bytes,
                    page_count=page_count,
                    word_count=word_cnt,
                    char_count=char_cnt,
                    raw_text=raw_text,
                    status=result.status.value if hasattr(result.status, "value") else str(result.status),
                    primary_code=result.primary_diagnosis.code if result.primary_diagnosis else None,
                    primary_description=(
                        result.primary_diagnosis.description if result.primary_diagnosis else None
                    ),
                    primary_confidence=(
                        result.primary_diagnosis.confidence_score if result.primary_diagnosis else None
                    ),
                    secondary_count=len(result.secondary_diagnoses),
                    abstention_count=len(result.abstentions),
                    processing_time_ms=result.processing_time_ms,
                    metadata_json=json.dumps(meta_dict),
                    created_at=datetime.now(UTC),
                )
                session.add(doc)

            # Insert Primary Diagnosis if present
            if result.primary_diagnosis:
                p_diag = result.primary_diagnosis
                doc.diagnoses.append(
                    DiagnosisRecord(
                        document_id=result.document_id,
                        code=p_diag.code,
                        description=p_diag.description,
                        role="PRIMARY",
                        acuity=p_diag.acuity.value if hasattr(p_diag.acuity, "value") else str(p_diag.acuity),
                        certainty=p_diag.certainty.value if hasattr(p_diag.certainty, "value") else str(p_diag.certainty),
                        confidence_score=p_diag.confidence_score,
                        is_terminal_billable=p_diag.is_terminal_billable,
                        evidence_quote=p_diag.evidence_quote,
                        icd10cm=p_diag.icd10cm or p_diag.code,
                        icdo=p_diag.icdo,
                        cpt=p_diag.cpt,
                        created_at=datetime.now(UTC),
                    )
                )

            # Insert Secondary Diagnoses
            for s_diag in result.secondary_diagnoses:
                doc.diagnoses.append(
                    DiagnosisRecord(
                        document_id=result.document_id,
                        code=s_diag.code,
                        description=s_diag.description,
                        role="SECONDARY",
                        acuity=s_diag.acuity.value if hasattr(s_diag.acuity, "value") else str(s_diag.acuity),
                        certainty=s_diag.certainty.value if hasattr(s_diag.certainty, "value") else str(s_diag.certainty),
                        confidence_score=s_diag.confidence_score,
                        is_terminal_billable=s_diag.is_terminal_billable,
                        evidence_quote=s_diag.evidence_quote,
                        icd10cm=s_diag.icd10cm or s_diag.code,
                        icdo=s_diag.icdo,
                        cpt=s_diag.cpt,
                        created_at=datetime.now(UTC),
                    )
                )

            # Insert Abstentions
            for abst in result.abstentions:
                doc.abstentions.append(
                    AbstentionRecordModel(
                        document_id=result.document_id,
                        raw_term=abst.raw_term,
                        reason=abst.reason.value if hasattr(abst.reason, "value") else str(abst.reason),
                        detail=abst.detail,
                        stage=abst.stage.value if hasattr(abst.stage, "value") else str(abst.stage),
                        created_at=datetime.now(UTC),
                    )
                )

            # Add validation check records if available in metadata or result
            validation_checks_data = meta_dict.get("validation_checks", [])
            for chk in validation_checks_data:
                if isinstance(chk, dict):
                    doc.validation_checks.append(
                        ValidationCheckModel(
                            document_id=result.document_id,
                            diagnosis_code=chk.get("diagnosis_code"),
                            rule_name=chk.get("rule_name", "GeneralValidation"),
                            passed=bool(chk.get("passed", True)),
                            details=str(chk.get("details", "")),
                            created_at=datetime.now(UTC),
                        )
                    )

            session.flush()
            result_dict = doc.to_dict()

        logger.info(
            "Persisted coding result for document_id=%s (primary=%s, secondary_cnt=%d)",
            result.document_id,
            result_dict.get("primary_code"),
            result_dict.get("secondary_count", 0),
        )
        return result_dict

    def get_document_by_id(self, document_id: str) -> dict[str, Any] | None:
        """Fetch a single document by its encounter ID with full nested relationships."""
        with get_db_session() as session:
            stmt = (
                select(ProcessedDocument)
                .options(
                    selectinload(ProcessedDocument.diagnoses),
                    selectinload(ProcessedDocument.abstentions),
                    selectinload(ProcessedDocument.validation_checks),
                )
                .where(ProcessedDocument.document_id == document_id)
            )
            doc = session.execute(stmt).scalar_one_or_none()
            return doc.to_dict() if doc else None

    def list_documents(
        self,
        search: str | None = None,
        status: str | None = None,
        code_filter: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query and paginate processed documents with optional full-text and status filtering."""
        with get_db_session() as session:
            stmt = (
                select(ProcessedDocument)
                .options(
                    selectinload(ProcessedDocument.diagnoses),
                    selectinload(ProcessedDocument.abstentions),
                )
                .order_by(desc(ProcessedDocument.created_at))
            )

            if status and status != "ALL":
                stmt = stmt.where(ProcessedDocument.status == status)

            if search:
                term = f"%{search.strip()}%"
                stmt = stmt.where(
                    or_(
                        ProcessedDocument.document_id.ilike(term),
                        ProcessedDocument.filename.ilike(term),
                        ProcessedDocument.raw_text.ilike(term),
                        ProcessedDocument.primary_description.ilike(term),
                    )
                )

            if code_filter:
                code_term = f"%{code_filter.strip()}%"
                stmt = stmt.where(
                    or_(
                        ProcessedDocument.primary_code.ilike(code_term),
                        ProcessedDocument.document_id.in_(
                            select(DiagnosisRecord.document_id).where(
                                DiagnosisRecord.code.ilike(code_term)
                            )
                        ),
                    )
                )

            stmt = stmt.limit(limit).offset(offset)
            docs = session.execute(stmt).scalars().all()
            return [d.to_dict() for d in docs]

    def delete_document(self, document_id: str) -> bool:
        """Delete an encounter document and all associated cascade records."""
        with get_db_session() as session:
            doc = session.execute(
                select(ProcessedDocument).where(ProcessedDocument.document_id == document_id)
            ).scalar_one_or_none()
            if not doc:
                return False
            session.delete(doc)
            return True

    def get_analytics_summary(self) -> dict[str, Any]:
        """Aggregate high-level metrics, top diagnoses, and quality indicators."""
        with get_db_session() as session:
            total_documents = session.scalar(select(func.count(ProcessedDocument.id))) or 0
            total_diagnoses = session.scalar(select(func.count(DiagnosisRecord.id))) or 0
            billable_count = session.scalar(
                select(func.count(DiagnosisRecord.id)).where(DiagnosisRecord.is_terminal_billable.is_(True))
            ) or 0
            avg_latency = session.scalar(select(func.avg(ProcessedDocument.processing_time_ms))) or 0.0

            # Status breakdown
            status_rows = session.execute(
                select(ProcessedDocument.status, func.count(ProcessedDocument.id)).group_by(
                    ProcessedDocument.status
                )
            ).all()
            status_breakdown = {row[0]: row[1] for row in status_rows}

            # Top 10 Primary Codes
            top_primary_rows = session.execute(
                select(
                    ProcessedDocument.primary_code,
                    ProcessedDocument.primary_description,
                    func.count(ProcessedDocument.id).label("count"),
                )
                .where(ProcessedDocument.primary_code.isnot(None))
                .group_by(ProcessedDocument.primary_code, ProcessedDocument.primary_description)
                .order_by(desc("count"))
                .limit(10)
            ).all()
            top_primary_codes = [
                {"code": r[0], "description": r[1], "count": r[2]} for r in top_primary_rows
            ]

            # Top 10 Secondary Codes
            top_secondary_rows = session.execute(
                select(
                    DiagnosisRecord.code,
                    DiagnosisRecord.description,
                    func.count(DiagnosisRecord.id).label("count"),
                )
                .where(DiagnosisRecord.role == "SECONDARY")
                .group_by(DiagnosisRecord.code, DiagnosisRecord.description)
                .order_by(desc("count"))
                .limit(10)
            ).all()
            top_secondary_codes = [
                {"code": r[0], "description": r[1], "count": r[2]} for r in top_secondary_rows
            ]

            # Abstention reason distribution
            abstention_rows = session.execute(
                select(
                    AbstentionRecordModel.reason,
                    func.count(AbstentionRecordModel.id).label("count"),
                )
                .group_by(AbstentionRecordModel.reason)
                .order_by(desc("count"))
            ).all()
            abstention_distribution = {r[0]: r[1] for r in abstention_rows}

            billable_ratio = (
                round((billable_count / total_diagnoses) * 100.0, 1) if total_diagnoses > 0 else 0.0
            )

            return {
                "total_documents": total_documents,
                "total_diagnoses": total_diagnoses,
                "billable_count": billable_count,
                "billable_ratio_percent": billable_ratio,
                "average_processing_time_ms": round(float(avg_latency), 2),
                "status_breakdown": status_breakdown,
                "top_primary_codes": top_primary_codes,
                "top_secondary_codes": top_secondary_codes,
                "abstention_distribution": abstention_distribution,
            }

    def get_documents_df(self) -> pd.DataFrame:
        """Export all documents and primary coding decisions as a pandas DataFrame."""
        docs = self.list_documents(limit=1000)
        if not docs:
            return pd.DataFrame(
                columns=[
                    "document_id",
                    "filename",
                    "source_type",
                    "status",
                    "primary_code",
                    "primary_description",
                    "primary_confidence",
                    "secondary_count",
                    "abstention_count",
                    "page_count",
                    "word_count",
                    "processing_time_ms",
                    "created_at",
                ]
            )

        records = []
        for d in docs:
            records.append(
                {
                    "document_id": d["document_id"],
                    "filename": d["filename"],
                    "source_type": d["source_type"],
                    "status": d["status"],
                    "primary_code": d["primary_code"] or "N/A",
                    "primary_description": d["primary_description"] or "N/A",
                    "primary_confidence": d["primary_confidence"] or 0.0,
                    "secondary_count": d["secondary_count"],
                    "abstention_count": d["abstention_count"],
                    "page_count": d["page_count"],
                    "word_count": d["word_count"],
                    "processing_time_ms": d["processing_time_ms"],
                    "created_at": d["created_at"],
                }
            )
        return pd.DataFrame(records)

    def purge_all_documents(self) -> int:
        """Delete all documents (primarily used for test cleanup)."""
        with get_db_session() as session:
            count = session.query(ProcessedDocument).delete()
            return count

    # -------------------------------------------------------------------------
    # Pipeline Thread & Step Lifecycle Management (Persistence & Audit)
    # -------------------------------------------------------------------------
    def create_pipeline_thread(
        self,
        thread_id: str,
        document_id: str,
        input_source: str = "clinical_note.txt",
        raw_text: str = "",
        total_steps: int = 10,
    ) -> dict[str, Any]:
        """Initialize and persist a new pipeline execution thread with RUNNING status."""
        with get_db_session() as session:
            existing = session.execute(
                select(PipelineThreadModel).where(PipelineThreadModel.thread_id == thread_id)
            ).scalar_one_or_none()

            if existing:
                thread = existing
                thread.document_id = document_id
                thread.status = "RUNNING"
                thread.current_step_name = "Starting"
                thread.current_step_index = 0
                thread.total_steps = total_steps
                thread.progress_pct = 0.0
                thread.failed_step = None
                thread.error_message = None
                thread.error_traceback = None
                thread.input_source = input_source
                thread.raw_text = raw_text
                thread.duration_ms = 0.0
                thread.start_time = datetime.now(UTC)
                thread.end_time = None
                thread.steps.clear()
            else:
                thread = PipelineThreadModel(
                    thread_id=thread_id,
                    document_id=document_id,
                    status="RUNNING",
                    current_step_name="Starting",
                    current_step_index=0,
                    total_steps=total_steps,
                    progress_pct=0.0,
                    failed_step=None,
                    error_message=None,
                    error_traceback=None,
                    input_source=input_source,
                    raw_text=raw_text,
                    duration_ms=0.0,
                    start_time=datetime.now(UTC),
                )
                session.add(thread)

            session.flush()
            return thread.to_dict()

    def record_pipeline_step(
        self,
        thread_id: str,
        step_index: int,
        step_name: str,
        status: str = "SUCCESS",
        duration_ms: float = 0.0,
        log_message: str = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Record an execution node step, updating the parent thread's current step and progress."""
        with get_db_session() as session:
            thread = session.execute(
                select(PipelineThreadModel)
                .options(selectinload(PipelineThreadModel.steps))
                .where(PipelineThreadModel.thread_id == thread_id)
            ).scalar_one_or_none()

            if not thread:
                return None

            # Calculate progress percentage
            pct = round(min(100.0, (step_index / max(1, thread.total_steps)) * 100.0), 1)
            thread.current_step_name = step_name
            thread.current_step_index = step_index
            thread.progress_pct = pct

            step = PipelineExecutionStepModel(
                thread_id=thread_id,
                step_name=step_name,
                step_index=step_index,
                status=status,
                duration_ms=round(duration_ms, 2),
                log_message=log_message,
                details_json=json.dumps(details or {}),
                start_time=datetime.now(UTC),
                end_time=datetime.now(UTC),
            )
            thread.steps.append(step)
            session.flush()
            return step.to_dict()

    def finish_pipeline_thread_success(
        self,
        thread_id: str,
        result_data: dict[str, Any] | CodingResult,
        duration_ms: float,
    ) -> dict[str, Any] | None:
        """Finalize a thread with SUCCESS status and archive the full output payload."""
        payload_str = (
            result_data.model_dump_json()
            if hasattr(result_data, "model_dump_json")
            else json.dumps(result_data)
        )
        with get_db_session() as session:
            thread = session.execute(
                select(PipelineThreadModel)
                .options(selectinload(PipelineThreadModel.steps))
                .where(PipelineThreadModel.thread_id == thread_id)
            ).scalar_one_or_none()

            if not thread:
                return None

            thread.status = "SUCCESS"
            thread.progress_pct = 100.0
            thread.current_step_name = "Completed"
            thread.current_step_index = thread.total_steps
            thread.duration_ms = round(duration_ms, 2)
            thread.result_json = payload_str
            thread.end_time = datetime.now(UTC)
            session.flush()
            return thread.to_dict()

    def finish_pipeline_thread_failure(
        self,
        thread_id: str,
        failed_step: str,
        error_message: str,
        error_traceback: str,
        duration_ms: float,
    ) -> dict[str, Any] | None:
        """Finalize a thread with FAILED status, recording the exact step and exception trace."""
        with get_db_session() as session:
            thread = session.execute(
                select(PipelineThreadModel)
                .options(selectinload(PipelineThreadModel.steps))
                .where(PipelineThreadModel.thread_id == thread_id)
            ).scalar_one_or_none()

            if not thread:
                return None

            thread.status = "FAILED"
            thread.failed_step = failed_step
            thread.error_message = error_message
            thread.error_traceback = error_traceback
            thread.duration_ms = round(duration_ms, 2)
            thread.end_time = datetime.now(UTC)
            session.flush()
            return thread.to_dict()

    def get_pipeline_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Fetch a specific execution thread with its ordered steps and result."""
        with get_db_session() as session:
            thread = session.execute(
                select(PipelineThreadModel)
                .options(selectinload(PipelineThreadModel.steps))
                .where(PipelineThreadModel.thread_id == thread_id)
            ).scalar_one_or_none()
            return thread.to_dict() if thread else None

    def list_pipeline_threads(
        self,
        limit: int = 50,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List historical pipeline execution threads with optional status filtering."""
        with get_db_session() as session:
            stmt = (
                select(PipelineThreadModel)
                .options(selectinload(PipelineThreadModel.steps))
                .order_by(desc(PipelineThreadModel.start_time))
            )
            if status and status != "ALL":
                stmt = stmt.where(PipelineThreadModel.status == status)

            stmt = stmt.limit(limit)
            threads = session.execute(stmt).scalars().all()
            return [t.to_dict() for t in threads]

    def get_latest_pipeline_thread(self) -> dict[str, Any] | None:
        """Fetch the most recent pipeline thread (running or finished)."""
        with get_db_session() as session:
            stmt = (
                select(PipelineThreadModel)
                .options(selectinload(PipelineThreadModel.steps))
                .order_by(desc(PipelineThreadModel.start_time))
                .limit(1)
            )
            thread = session.execute(stmt).scalar_one_or_none()
            return thread.to_dict() if thread else None

    def delete_pipeline_thread(self, thread_id: str) -> bool:
        """Delete a thread and all associated step logs."""
        with get_db_session() as session:
            thread = session.execute(
                select(PipelineThreadModel).where(PipelineThreadModel.thread_id == thread_id)
            ).scalar_one_or_none()
            if not thread:
                return False
            session.delete(thread)
            return True

