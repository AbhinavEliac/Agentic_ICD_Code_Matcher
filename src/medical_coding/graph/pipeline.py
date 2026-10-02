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


async def process_clinical_document(
    source: Path | bytes | str = "",
    document_id: str | None = None,
    pdf_path: str | Path | None = None,
    pdf_bytes: bytes | None = None,
    metadata: dict[str, Any] | None = None,
    graph: Any = None,
) -> CodingResult:
    """Execute end-to-end medical coding pipeline for a single clinical document or text.

    Args:
        source: Text string, PDF file Path, or raw PDF bytes.
        document_id: Optional unique encounter ID.
        pdf_path: Optional explicit path to PDF source.
        pdf_bytes: Optional explicit raw PDF file bytes.
        metadata: Optional dictionary of encounter context.
        graph: Optional compiled LangGraph workflow app instance.

    Returns:
        Validated CodingResult conforming to Pydantic schema with internal audit trail.
    """
    start_time = time.perf_counter()
    doc_id = document_id or f"doc-{uuid4().hex[:8]}"
    app = graph or get_compiled_graph()

    raw_text = ""
    resolved_pdf_path = str(pdf_path) if pdf_path else None
    resolved_pdf_bytes = pdf_bytes
    source_type = "text"

    if resolved_pdf_path:
        source_type = "pdf_file"
    elif resolved_pdf_bytes:
        source_type = "pdf_bytes"
    elif isinstance(source, bytes):
        resolved_pdf_bytes = source
        source_type = "pdf_bytes"
    elif isinstance(source, Path):
        resolved_pdf_path = str(source)
        source_type = "pdf_file"
    elif isinstance(source, str):
        # Determine if string points to an existing file
        p = Path(source)
        if p.exists() and p.is_file():
            resolved_pdf_path = str(p)
            source_type = "pdf_file"
        else:
            raw_text = source
            source_type = "text"

    initial_state = create_initial_state(
        document_id=doc_id,
        raw_text=raw_text,
        pdf_path=resolved_pdf_path,
        pdf_bytes=resolved_pdf_bytes,
        source_type=source_type,
        metadata=metadata,
    )

    logger.info("Starting pipeline execution for doc_id=%s (source_type=%s)", doc_id, source_type)
    final_state = await app.ainvoke(initial_state)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    result: CodingResult | None = final_state.get("final_result")
    if result is None:
        # Fallback empty result
        result = CodingResult(
            document_id=doc_id,
            status=ExecutionStatus.ABSTAINED
            if final_state.get("abstentions")
            else ExecutionStatus.ERROR,
            primary_diagnosis=None,
            secondary_diagnoses=[],
            abstentions=final_state.get("abstentions", []),
            processing_time_ms=elapsed_ms,
            models_used={"llm": "local_gguf", "retrieval": "local_faiss_bm25"},
            metadata=final_state.get("metadata", {}),
        )
    else:
        result.processing_time_ms = round(elapsed_ms, 2)

    logger.info(
        "Completed pipeline execution for doc_id=%s in %.2fms (status=%s, primary=%s, secondaries=%d)",
        doc_id,
        elapsed_ms,
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
