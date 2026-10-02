"""LangGraph state re-exports and pipeline initialization helpers."""

from typing import Any

from medical_coding.schemas.enums import PipelineStage
from medical_coding.schemas.state import PipelineGraphState


def create_initial_state(
    document_id: str,
    raw_text: str = "",
    pdf_path: str | None = None,
    pdf_bytes: bytes | None = None,
    file_path: str | None = None,
    file_bytes: bytes | None = None,
    source_type: str = "text",
    metadata: dict[str, Any] | None = None,
) -> PipelineGraphState:
    """Create a clean, standardized initial state dictionary for LangGraph execution."""
    # Reconcile file_bytes / pdf_bytes and file_path / pdf_path
    resolved_bytes = file_bytes if file_bytes is not None else pdf_bytes
    resolved_path = file_path if file_path is not None else pdf_path

    return {
        "document_id": document_id,
        "raw_text": raw_text,
        "pdf_path": resolved_path,
        "pdf_bytes": resolved_bytes,
        "file_path": resolved_path,
        "file_bytes": resolved_bytes,
        "detected_filetype": None,
        "source_type": source_type,
        "metadata": metadata or {},
        "extracted_diagnoses": [],
        "extracted_conditions": [],
        "context_assessments": [],
        "contextualized_diagnoses": [],
        "classified_diagnoses": [],
        "candidate_pool": {},
        "ranked_selections": [],
        "validated_diagnoses": [],
        "abstentions": [],
        "audit_trail": [],
        "current_stage": PipelineStage.INGESTION,
        "errors": [],
        "is_aborted": False,
        "final_result": None,
    }


__all__ = ["PipelineGraphState", "create_initial_state"]
