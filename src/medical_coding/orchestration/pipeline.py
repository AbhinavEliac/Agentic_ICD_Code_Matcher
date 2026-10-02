"""High-level asynchronous orchestrator for the medical coding pipeline with bounded parallelism."""

import asyncio
import time
from pathlib import Path
from typing import Any

from medical_coding.config.settings import Settings, get_settings
from medical_coding.graph.state import create_initial_state
from medical_coding.graph.workflow import get_compiled_graph
from medical_coding.pdf.concurrency import BoundedDocumentGate
from medical_coding.schemas.clinical import ClinicalDocument
from medical_coding.schemas.enums import ExecutionStatus
from medical_coding.schemas.response import CodingResult
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalCodingPipeline:
    """Orchestrates document coding by executing the compiled LangGraph workflow.

    Enforces document-level asynchronous parallelism while bounding model-level inference.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.graph = get_compiled_graph()
        self.gate = BoundedDocumentGate(
            max_concurrency=self.settings.max_document_concurrency,
            settings=self.settings,
        )

    async def run_document(
        self,
        document_id: str,
        text: str = "",
        pdf_path: str | Path | None = None,
        pdf_bytes: bytes | None = None,
        file_path: str | Path | None = None,
        file_bytes: bytes | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CodingResult:
        """Execute the end-to-end coding pipeline for a single clinical document, image, or text.

        Args:
            document_id: Unique encounter identifier.
            text: Clinical document text.
            pdf_path: Optional path to PDF source file.
            pdf_bytes: Optional raw PDF file bytes.
            file_path: Optional path to any document/image file (.pdf, .txt, .png, .jpg).
            file_bytes: Optional raw file bytes for any supported format.
            metadata: Optional dictionary of clinical metadata.

        Returns:
            Deterministic CodingResult adhering to ICD guidelines.
        """
        start_time = time.perf_counter()
        resolved_bytes = file_bytes if file_bytes is not None else pdf_bytes
        resolved_path = file_path if file_path is not None else pdf_path

        initial_state = create_initial_state(
            document_id=document_id,
            raw_text=text,
            pdf_path=str(resolved_path) if resolved_path else None,
            pdf_bytes=resolved_bytes,
            file_path=str(resolved_path) if resolved_path else None,
            file_bytes=resolved_bytes,
            source_type="file" if (resolved_path or resolved_bytes) else "text",
            metadata=metadata or {},
        )

        try:
            logger.info("Invoking coding graph for document_id=%s", document_id)
            final_state = await self.graph.ainvoke(initial_state)

            result: CodingResult | None = final_state.get("final_result")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if result is None:
                logger.warning("Pipeline completed without final_result for %s", document_id)
                return CodingResult(
                    document_id=document_id,
                    status=ExecutionStatus.ABSTAINED
                    if final_state.get("abstentions")
                    else ExecutionStatus.ERROR,
                    primary_diagnosis=None,
                    secondary_diagnoses=[],
                    abstentions=final_state.get("abstentions", []),
                    processing_time_ms=elapsed_ms,
                    models_used={"llm": "local_gguf", "retrieval": "local_faiss_bm25"},
                    metadata={"error": "Workflow did not emit final_result"},
                )

            result.processing_time_ms = round(elapsed_ms, 2)
            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception("Pipeline invocation failed for document %s: %s", document_id, exc)
            return CodingResult(
                document_id=document_id,
                status=ExecutionStatus.ERROR,
                primary_diagnosis=None,
                secondary_diagnoses=[],
                abstentions=[],
                processing_time_ms=elapsed_ms,
                models_used={},
                metadata={"error": str(exc)},
            )

    async def run_batch_documents(
        self,
        documents: list[ClinicalDocument | dict[str, Any]],
        max_concurrency: int | None = None,
    ) -> list[CodingResult]:
        """Process a collection of clinical documents in parallel with bounded concurrency.

        Args:
            documents: List of ClinicalDocument objects or dictionaries with document specifications.
            max_concurrency: Optional override for document-level concurrency ceiling.

        Returns:
            List of CodingResult objects for each document.
        """
        gate = (
            BoundedDocumentGate(max_concurrency=max_concurrency, settings=self.settings)
            if max_concurrency
            else self.gate
        )

        async def _run_bounded(doc: Any) -> CodingResult:
            async with gate.get_semaphore():
                if isinstance(doc, ClinicalDocument):
                    return await self.run_document(
                        document_id=doc.document_id,
                        text=doc.text,
                        metadata=doc.metadata,
                    )
                if isinstance(doc, dict):
                    return await self.run_document(
                        document_id=doc.get("document_id", "doc-unknown"),
                        text=doc.get("text", ""),
                        pdf_path=doc.get("pdf_path"),
                        pdf_bytes=doc.get("pdf_bytes"),
                        file_path=doc.get("file_path"),
                        file_bytes=doc.get("file_bytes"),
                        metadata=doc.get("metadata"),
                    )
                raise TypeError(f"Unsupported document item type in batch: {type(doc)}")

        tasks = [_run_bounded(doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        return list(results)
