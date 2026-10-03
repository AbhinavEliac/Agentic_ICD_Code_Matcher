"""Asynchronous document and batch execution pipeline orchestrating LangGraph workflows."""

import asyncio
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from medical_coding.config.settings import Settings, get_settings
from medical_coding.graph.state import create_initial_state
from medical_coding.graph.workflow import get_compiled_graph
from medical_coding.pdf.concurrency import BoundedDocumentGate
from medical_coding.schemas.enums import ExecutionStatus
from medical_coding.schemas.response import BatchJobStatus, CodingResult
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


import traceback
from collections.abc import Callable

from medical_coding.database.repository import MedicalCodingRepository

NODE_META: dict[str, tuple[int, str, str]] = {
    "validate_document": (1, "Document Validation", "Validating format and invariants"),
    "extract_text": (2, "Multimodal Normalization", "Extracting text and normalizing sections"),
    "extract_diagnoses": (3, "Clinical Extraction Agent", "Extracting diagnoses and evidence quotes"),
    "analyze_context": (4, "Clinical Context Analysis", "Evaluating negation, certainty, and acuity"),
    "classify_diagnoses": (5, "Classification Agent", "Determining primary vs secondary roles"),
    "retrieve_candidates": (6, "Multi-System Retrieval", "Querying ICD-10-CM, ICD-O, and CPT catalogs"),
    "rank_candidates": (7, "Candidate Ranking Agent", "Scoring and ranking candidate matches"),
    "validate_codes": (8, "Deterministic Invariant Validation", "Enforcing HIPAA leaf & Excludes1 checks"),
    "evaluate_confidence": (9, "Confidence & Abstention Assessment", "Evaluating confidence score thresholds"),
    "finalize_output": (10, "Final Payload Generation", "Generating structured multi-system payload"),
}


def _summarize_node_output(node_name: str, output: Any) -> str:
    """Produce a concise, human-readable summary for a completed pipeline node."""
    if not isinstance(output, dict):
        return f"{node_name} completed"

    if node_name == "validate_document":
        doc_id = output.get("document_id", "")
        return f"Document payload verified (encounter {doc_id})"
    elif node_name == "extract_text":
        txt = output.get("normalized_text", "")
        pages = output.get("extracted_pages", [])
        return f"Text normalized ({len(txt)} chars, {len(pages)} page(s) parsed)"
    elif node_name == "extract_diagnoses":
        diags = output.get("extracted_diagnoses", [])
        return f"Clinical Extraction Agent found {len(diags)} clinical condition(s)"
    elif node_name == "analyze_context":
        return "Clinical Context Agent evaluated negation, temporality, certainty, and acuity"
    elif node_name == "classify_diagnoses":
        pri = output.get("primary_diagnosis")
        pri_code = getattr(pri, "code", None) or (pri.get("code") if isinstance(pri, dict) else "Identified")
        return f"Classification complete (Primary designated: {pri_code})"
    elif node_name == "retrieve_candidates":
        pool = output.get("candidate_pool", {})
        total_cands = sum(len(v) for v in pool.values()) if isinstance(pool, dict) else 0
        return f"Hybrid Multi-System Retrieval fetched {total_cands} candidate codes"
    elif node_name == "rank_candidates":
        ranked = output.get("ranked_selections", [])
        return f"Candidate Ranking Agent scored and selected {len(ranked)} candidate codes"
    elif node_name == "validate_codes":
        v_checks = output.get("validation_checks", [])
        return f"Deterministic Validation passed ({len(v_checks)} HIPAA/Excludes1 checks verified)"
    elif node_name == "evaluate_confidence":
        abst = output.get("abstentions", [])
        return f"Confidence evaluation complete ({len(abst)} explicit abstention(s))"
    elif node_name == "finalize_output":
        return "Final multi-system payload constructed with audit trail"
    return f"{node_name} completed successfully"


def _sanitize_details(output: Any) -> dict[str, Any]:
    """Extract lightweight, JSON-serializable audit summary from node output dict."""
    if not isinstance(output, dict):
        return {}
    summary: dict[str, Any] = {}
    for k, v in output.items():
        if k in ("raw_text", "normalized_text", "pdf_bytes", "file_bytes"):
            continue
        if isinstance(v, (int, float, str, bool, type(None))):
            summary[k] = v
        elif isinstance(v, list):
            summary[k] = len(v)
        elif isinstance(v, dict):
            summary[k] = {sk: type(sv).__name__ for sk, sv in list(v.items())[:5]}
    return summary


async def process_clinical_document(
    source: Path | bytes | str = "",
    document_id: str | None = None,
    pdf_path: str | Path | None = None,
    pdf_bytes: bytes | None = None,
    metadata: dict[str, Any] | None = None,
    graph: Any = None,
    thread_id: str | None = None,
    progress_callback: Callable[[int, int, str, str, float], None] | None = None,
    persist_thread: bool = True,
) -> CodingResult:
    """Execute end-to-end medical coding pipeline for a single clinical document or text.

    Args:
        source: Text string, PDF file Path, or raw PDF bytes.
        document_id: Optional unique encounter ID.
        pdf_path: Optional explicit path to PDF source.
        pdf_bytes: Optional explicit raw PDF file bytes.
        metadata: Optional dictionary of encounter context.
        graph: Optional compiled LangGraph workflow app instance.
        thread_id: Optional unique thread ID for tracking and history persistence.
        progress_callback: Optional callback func(step_idx, total_steps, node_name, log_message, duration_ms).
        persist_thread: Whether to archive execution thread and step logs in SQLite.

    Returns:
        Validated CodingResult conforming to Pydantic schema with internal audit trail.
    """
    start_time = time.perf_counter()
    doc_id = document_id or f"doc-{uuid4().hex[:8]}"
    active_thread_id = thread_id or f"thread-{uuid4().hex[:10]}"
    app = graph or get_compiled_graph()
    repo = MedicalCodingRepository() if persist_thread else None

    raw_text = ""
    resolved_pdf_path = str(pdf_path) if pdf_path else None
    resolved_pdf_bytes = pdf_bytes
    source_type = "text"

    if isinstance(source, str):
        # Determine if string points to an existing file on disk
        if len(source) < 300:
            try:
                p = Path(source)
                if p.exists() and p.is_file():
                    resolved_pdf_path = str(p)
                    source_type = "file"
                else:
                    raw_text = source
            except Exception:
                raw_text = source
        else:
            raw_text = source
    elif isinstance(source, bytes):
        resolved_pdf_bytes = source
        source_type = "file_bytes"
    elif isinstance(source, Path):
        resolved_pdf_path = str(source)
        source_type = "file"

    if metadata and metadata.get("source_type"):
        source_type = str(metadata["source_type"])

    initial_state = create_initial_state(
        document_id=doc_id,
        raw_text=raw_text,
        pdf_path=resolved_pdf_path,
        pdf_bytes=resolved_pdf_bytes,
        file_path=resolved_pdf_path,
        file_bytes=resolved_pdf_bytes,
        source_type=source_type,
        metadata=metadata,
    )

    input_source = resolved_pdf_path or (metadata.get("filename") if metadata else None) or "clinical_note.txt"
    if repo:
        try:
            repo.create_pipeline_thread(
                thread_id=active_thread_id,
                document_id=doc_id,
                input_source=str(input_source),
                raw_text=raw_text[:4000],
                total_steps=10,
            )
        except Exception as exc:
            logger.warning("Failed to initialize pipeline thread record in repository: %s", exc)

    logger.info(
        "Starting pipeline execution for doc_id=%s (thread_id=%s, source_type=%s)",
        doc_id,
        active_thread_id,
        source_type,
    )

    accumulated_state: dict[str, Any] = dict(initial_state)
    current_node_name = "validate_document"
    current_step_idx = 1
    step_start_time = time.perf_counter()

    try:
        async for chunk in app.astream(initial_state, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                step_end_time = time.perf_counter()
                duration_ms = (step_end_time - step_start_time) * 1000.0
                step_start_time = step_end_time

                current_node_name = node_name
                step_idx, step_label, default_desc = NODE_META.get(
                    node_name, (current_step_idx + 1, node_name, "Executing stage")
                )
                current_step_idx = step_idx

                if isinstance(node_output, dict):
                    accumulated_state.update(node_output)

                log_msg = _summarize_node_output(node_name, node_output)
                details = _sanitize_details(node_output)

                if repo:
                    try:
                        repo.record_pipeline_step(
                            thread_id=active_thread_id,
                            step_index=step_idx,
                            step_name=node_name,
                            status="SUCCESS",
                            duration_ms=duration_ms,
                            log_message=log_msg,
                            details=details,
                        )
                    except Exception as step_err:
                        logger.warning("Error recording step in DB: %s", step_err)

                if progress_callback:
                    try:
                        progress_callback(step_idx, 10, node_name, log_msg, duration_ms)
                    except Exception:
                        pass

    except Exception as exc:
        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        tb_str = traceback.format_exc()
        logger.exception("Pipeline failed at node %s: %s", current_node_name, exc)

        if repo:
            try:
                repo.record_pipeline_step(
                    thread_id=active_thread_id,
                    step_index=current_step_idx,
                    step_name=current_node_name,
                    status="FAILED",
                    duration_ms=0.0,
                    log_message=f"Node {current_node_name} execution failed: {exc}",
                    details={"traceback": tb_str, "error": str(exc)},
                )
                repo.finish_pipeline_thread_failure(
                    thread_id=active_thread_id,
                    failed_step=current_node_name,
                    error_message=str(exc),
                    error_traceback=tb_str,
                    duration_ms=total_elapsed_ms,
                )
            except Exception as finish_err:
                logger.warning("Error recording failure in DB: %s", finish_err)

        if progress_callback:
            try:
                progress_callback(current_step_idx, 10, current_node_name, f"FAILED: {exc}", 0.0)
            except Exception:
                pass

        return CodingResult(
            document_id=doc_id,
            status=ExecutionStatus.ERROR,
            primary_diagnosis=None,
            secondary_diagnoses=[],
            abstentions=[],
            processing_time_ms=round(total_elapsed_ms, 2),
            models_used={"llm": "local_gguf", "retrieval": "local_faiss_bm25"},
            metadata={
                "thread_id": active_thread_id,
                "failed_step": current_node_name,
                "failed_step_index": current_step_idx,
                "error": str(exc),
                "traceback": tb_str,
            },
        )

    total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    result: CodingResult | None = accumulated_state.get("final_result")

    if result is None:
        result = CodingResult(
            document_id=doc_id,
            status=ExecutionStatus.ABSTAINED
            if accumulated_state.get("abstentions")
            else ExecutionStatus.ERROR,
            primary_diagnosis=None,
            secondary_diagnoses=[],
            abstentions=accumulated_state.get("abstentions", []),
            processing_time_ms=round(total_elapsed_ms, 2),
            models_used={"llm": "local_gguf", "retrieval": "local_faiss_bm25"},
            metadata={"thread_id": active_thread_id, "warning": "No final_result emitted by workflow"},
        )
    else:
        result.processing_time_ms = round(total_elapsed_ms, 2)
        if not result.metadata:
            result.metadata = {}
        result.metadata["thread_id"] = active_thread_id

    if repo:
        try:
            repo.finish_pipeline_thread_success(
                thread_id=active_thread_id,
                result_data=result,
                duration_ms=total_elapsed_ms,
            )
        except Exception as success_err:
            logger.warning("Error recording thread success in DB: %s", success_err)

    logger.info(
        "Completed pipeline execution for doc_id=%s (thread_id=%s) in %.2fms (status=%s, primary=%s, secondaries=%d)",
        doc_id,
        active_thread_id,
        total_elapsed_ms,
        result.status,
        result.primary_diagnosis.code if result.primary_diagnosis else "None",
        len(result.secondary_diagnoses),
    )

    return result


async def process_clinical_document_batch(
    documents: list[Path | bytes | str | dict[str, Any]],
    job_id: str | None = None,
    max_concurrency: int | None = None,
    settings: Settings | None = None,
) -> BatchJobStatus:
    """Asynchronously process a batch of clinical documents (e.g. >= 10 PDFs).

    ARCHITECTURAL PARALLELISM GUARANTEE:
    - Multiple documents (>= 10) are ingested, parsed, and validated in parallel.
    - Diagnosis-level candidate retrieval is parallelized within each document.
    - Model-level concurrency is bounded via LLMLifecycleManager to avoid spawning
      independent GPT4All instances.

    Args:
        documents: List of file paths, bytes, text strings, or dicts with source and metadata.
        job_id: Optional batch identifier.
        max_concurrency: Maximum simultaneous document-level pipeline tasks (default: 10).
        settings: Application settings.

    Returns:
        BatchJobStatus summarizing overall completion, errors, and individual CodingResults.
    """
    cfg = settings or get_settings()
    concurrency_limit = max_concurrency or cfg.max_document_concurrency
    batch_id = job_id or f"batch-{uuid4().hex[:8]}"

    gate = BoundedDocumentGate(max_concurrency=concurrency_limit, settings=cfg)
    app = get_compiled_graph()

    logger.info(
        "Beginning batch execution job_id=%s with %d documents (concurrency limit: %d)",
        batch_id,
        len(documents),
        concurrency_limit,
    )

    async def _process_bounded(item: Any, idx: int) -> CodingResult:
        doc_source = item
        doc_id = f"{batch_id}-doc-{idx + 1}"
        meta: dict[str, Any] = {"batch_index": idx, "batch_id": batch_id}

        if isinstance(item, dict):
            doc_source = item.get("source") or item.get("text") or item.get("file_path")
            doc_id = item.get("document_id") or doc_id
            meta.update(item.get("metadata", {}))

        async with gate.get_semaphore():
            return await process_clinical_document(
                source=doc_source,
                document_id=doc_id,
                metadata=meta,
                graph=app,
            )

    tasks = [_process_bounded(doc, i) for i, doc in enumerate(documents)]
    results: list[CodingResult] = await asyncio.gather(*tasks)

    completed = sum(
        1
        for r in results
        if r.status
        in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS, ExecutionStatus.ABSTAINED)
    )
    failed = sum(1 for r in results if r.status == ExecutionStatus.ERROR)

    batch_status = BatchJobStatus(
        job_id=batch_id,
        total_documents=len(documents),
        completed_documents=completed,
        failed_documents=failed,
        status="completed" if failed == 0 else "failed" if completed == 0 else "completed",
        results=results,
    )

    logger.info(
        "Batch job_id=%s finished: %d total, %d completed, %d failed",
        batch_id,
        batch_status.total_documents,
        batch_status.completed_documents,
        batch_status.failed_documents,
    )

    return batch_status
