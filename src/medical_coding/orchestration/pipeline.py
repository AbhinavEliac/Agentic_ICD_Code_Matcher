"""High-level asynchronous orchestrator for the medical coding pipeline with bounded parallelism."""

import asyncio
from pathlib import Path
from typing import Any

from medical_coding.config.settings import Settings, get_settings
from medical_coding.graph.workflow import get_compiled_graph
from medical_coding.pdf.concurrency import BoundedDocumentGate
from medical_coding.schemas.clinical import ClinicalDocument
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
        thread_id: str | None = None,
        progress_callback: Any = None,
        persist_thread: bool = True,
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
            thread_id: Optional unique thread ID for tracking and history persistence.
            progress_callback: Optional callback func(step_idx, total_steps, node_name, log_message, duration_ms).
            persist_thread: Whether to archive execution thread and step logs in SQLite.

        Returns:
            Deterministic CodingResult adhering to ICD guidelines.
        """
        resolved_bytes = file_bytes if file_bytes is not None else pdf_bytes
        resolved_path = file_path if file_path is not None else pdf_path

        from medical_coding.graph.pipeline import process_clinical_document

        return await process_clinical_document(
            source=text,
            document_id=document_id,
            pdf_path=resolved_path,
            pdf_bytes=resolved_bytes,
            metadata=metadata,
            graph=self.graph,
            thread_id=thread_id,
            progress_callback=progress_callback,
            persist_thread=persist_thread,
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
