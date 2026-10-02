"""Asynchronous batch document processor with bounded concurrency, section mapping, and fault tolerance."""

import asyncio
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any
from uuid import uuid4

from medical_coding.config.settings import Settings, get_settings
from medical_coding.pdf.concurrency import BoundedDocumentGate
from medical_coding.pdf.extractor import PDFExtractor
from medical_coding.pdf.models import (
    BatchProcessingResult,
    ClinicalPDFDocument,
    DocumentProcessingResult,
    PDFExtractionStatus,
)
from medical_coding.pdf.section_detector import SectionDetector
from medical_coding.schemas.enums import ExecutionStatus
from medical_coding.schemas.response import BatchJobStatus, CodingResult
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class BatchPDFProcessor:
    """Orchestrates asynchronous PDF document ingestion, text normalization, sectioning, and batching."""

    def __init__(
        self,
        extractor: PDFExtractor | None = None,
        section_detector: SectionDetector | None = None,
        max_concurrency: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.extractor = extractor or PDFExtractor(
            max_pages=self.settings.pdf_max_pages,
        )
        self.section_detector = section_detector or SectionDetector()
        self.max_concurrency = max_concurrency or self.settings.max_document_concurrency
        self.gate = BoundedDocumentGate(
            max_concurrency=self.max_concurrency, settings=self.settings
        )

    async def process_document(
        self,
        pdf_source: Path | bytes | str,
        doc_id: str | None = None,
        filename: str | None = None,
    ) -> DocumentProcessingResult:
        """Process an individual clinical PDF into a structured ClinicalPDFDocument asynchronously.

        Pipeline:
        1. Extract text page-by-page (PyMuPDF -> fallback pdfplumber).
        2. Detect scanned or insufficient text conditions.
        3. Segment normalized text into clinical sections (diagnoses, hospital course, etc.).
        4. Assemble immutable ClinicalPDFDocument representation.

        Fault Isolation:
        Errors are captured per document. Malformed or scanned PDFs return structured failures
        without raising unhandled exceptions.
        """
        start_time = time.perf_counter()

        # Determine identifier and filename
        document_id = doc_id or str(uuid4())
        if filename:
            source_name = filename
        elif isinstance(pdf_source, (str, Path)):
            source_name = Path(pdf_source).name
        else:
            source_name = f"doc_{document_id[:8]}.pdf"

        logger.info("Processing document '%s' (doc_id=%s)...", source_name, document_id)

        try:
            # Execute extraction bounded by document gate
            sem = self.gate.get_semaphore()
            async with sem:
                extract_res = await self.extractor.extract_async(pdf_source)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Handle non-success states (scanned, empty, malformed, insufficient text)
            if extract_res.status != PDFExtractionStatus.SUCCESS:
                logger.warning(
                    "Document '%s' extraction returned status [%s]: %s",
                    source_name,
                    extract_res.status.value,
                    extract_res.error_message,
                )
                return DocumentProcessingResult(
                    document_id=document_id,
                    source_name=source_name,
                    status=extract_res.status,
                    document=None,
                    error_message=extract_res.error_message,
                    extraction_time_ms=elapsed_ms,
                    extractor_used=extract_res.extractor_used,
                    page_count=len(extract_res.pages),
                    char_count=len(extract_res.full_normalized_text),
                    needs_ocr=(extract_res.status == PDFExtractionStatus.NEEDS_OCR),
                )

            # Build page offset index for mapping sections to page numbers
            page_offsets = [
                (p.page_number, p.start_char_offset, p.end_char_offset) for p in extract_res.pages
            ]

            # Detect clinical sections
            sections = self.section_detector.detect_sections(
                full_text=extract_res.full_normalized_text,
                page_offsets=page_offsets,
            )

            clinical_doc = ClinicalPDFDocument(
                document_id=document_id,
                source_filename=source_name,
                total_pages=len(extract_res.pages),
                pages=extract_res.pages,
                full_normalized_text=extract_res.full_normalized_text,
                sections=sections,
                is_scanned=extract_res.is_scanned,
                metadata={
                    "extractor": extract_res.extractor_used,
                    "extraction_ms": elapsed_ms,
                    "section_count": len(sections),
                },
            )

            logger.info(
                "Document '%s' successfully processed: %d pages, %d chars, %d sections (%.2f ms)",
                source_name,
                len(extract_res.pages),
                len(extract_res.full_normalized_text),
                len(sections),
                elapsed_ms,
            )

            return DocumentProcessingResult(
                document_id=document_id,
                source_name=source_name,
                status=PDFExtractionStatus.SUCCESS,
                document=clinical_doc,
                error_message=None,
                extraction_time_ms=elapsed_ms,
                extractor_used=extract_res.extractor_used,
                page_count=len(extract_res.pages),
                char_count=len(extract_res.full_normalized_text),
                needs_ocr=False,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception("Unexpected error processing document '%s': %s", source_name, exc)
            return DocumentProcessingResult(
                document_id=document_id,
                source_name=source_name,
                status=PDFExtractionStatus.EXTRACTION_ERROR,
                document=None,
                error_message=f"Extraction failure: {exc}",
                extraction_time_ms=elapsed_ms,
                extractor_used="unknown",
                page_count=0,
                char_count=0,
                needs_ocr=False,
            )

    async def process_documents_batch(
        self,
        pdf_sources: list[Path | bytes | str],
        job_id: str | None = None,
        filenames: list[str] | None = None,
    ) -> BatchProcessingResult:
        """Process a batch of clinical PDFs asynchronously with bounded concurrency (>= 10 documents).

        Guarantees:
        - Bounded concurrency: strictly respects max_document_concurrency.
        - Fault isolation: one corrupted PDF will not cancel or crash other documents.
        - Independent results: each document produces a complete DocumentProcessingResult.
        """
        start_time = time.perf_counter()
        batch_id = job_id or str(uuid4())
        total = len(pdf_sources)

        logger.info(
            "Launching async PDF batch '%s' with %d documents (concurrency limit: %d)...",
            batch_id,
            total,
            self.max_concurrency,
        )

        tasks = []
        for idx, src in enumerate(pdf_sources):
            fn = filenames[idx] if filenames and idx < len(filenames) else None
            task = asyncio.create_task(
                self.process_document(
                    pdf_source=src,
                    filename=fn,
                )
            )
            tasks.append(task)

        # Await all tasks cooperatively; return_exceptions=True prevents batch collapse
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[DocumentProcessingResult] = []
        for idx, item in enumerate(raw_results):
            if isinstance(item, Exception):
                logger.error("Task for PDF #%d raised unhandled exception: %s", idx, item)
                results.append(
                    DocumentProcessingResult(
                        document_id=str(uuid4()),
                        source_name=f"document_{idx}.pdf",
                        status=PDFExtractionStatus.EXTRACTION_ERROR,
                        error_message=str(item),
                    )
                )
            elif isinstance(item, DocumentProcessingResult):
                results.append(item)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        elapsed_sec = max(elapsed_ms / 1000.0, 0.001)
        throughput = total / elapsed_sec

        success_count = sum(1 for r in results if r.status == PDFExtractionStatus.SUCCESS)
        ocr_count = sum(1 for r in results if r.needs_ocr)
        failed_count = total - success_count

        logger.info(
            "Batch '%s' complete: %d/%d succeeded, %d need OCR, %d failed in %.2f ms (%.2f docs/sec)",
            batch_id,
            success_count,
            total,
            ocr_count,
            failed_count,
            elapsed_ms,
            throughput,
        )

        return BatchProcessingResult(
            batch_id=batch_id,
            total_documents=total,
            successful_documents=success_count,
            failed_documents=failed_count,
            needs_ocr_documents=ocr_count,
            results=results,
            total_elapsed_ms=elapsed_ms,
            throughput_docs_per_sec=throughput,
        )

    # Legacy integration method for existing API endpoints
    async def process_single_pdf(
        self,
        pdf_path: Path,
        pipeline_executor: Callable[[str, str], Coroutine[Any, Any, CodingResult]],
    ) -> CodingResult:
        """Legacy helper running PDF extraction and passing text to pipeline executor."""
        doc_res = await self.process_document(pdf_path)
        doc_id = doc_res.document_id

        if doc_res.status != PDFExtractionStatus.SUCCESS or doc_res.document is None:
            return CodingResult(
                document_id=doc_id,
                status=ExecutionStatus.ERROR
                if doc_res.status != PDFExtractionStatus.NEEDS_OCR
                else ExecutionStatus.ABSTAINED,
                primary_diagnosis=None,
                secondary_diagnoses=[],
                abstentions=[],
                processing_time_ms=doc_res.extraction_time_ms,
                models_used={},
                metadata={
                    "error": doc_res.error_message or "Extraction failed",
                    "file": str(pdf_path),
                },
            )

        # Pass high-yield coding text rather than raw entire document
        coding_text = doc_res.document.get_coding_text()
        return await pipeline_executor(doc_id, coding_text)

    async def process_batch(
        self,
        pdf_paths: list[Path],
        pipeline_executor: Callable[[str, str], Coroutine[Any, Any, CodingResult]],
        job_id: str | None = None,
    ) -> BatchJobStatus:
        """Legacy helper processing a batch of PDFs through the coding pipeline."""
        batch_id = job_id or str(uuid4())
        total = len(pdf_paths)
        tasks = [
            asyncio.create_task(self.process_single_pdf(path, pipeline_executor))
            for path in pdf_paths
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        completed = sum(1 for r in results if r.status != ExecutionStatus.ERROR)
        failed = sum(1 for r in results if r.status == ExecutionStatus.ERROR)

        return BatchJobStatus(
            job_id=batch_id,
            total_documents=total,
            completed_documents=completed,
            failed_documents=failed,
            status="completed" if failed == 0 else "failed" if completed == 0 else "completed",
            results=results,
        )


async def process_document(
    pdf_source: Path | bytes | str,
    doc_id: str | None = None,
    filename: str | None = None,
    processor: BatchPDFProcessor | None = None,
) -> DocumentProcessingResult:
    """Convenience top-level function for processing a single PDF."""
    proc = processor or BatchPDFProcessor()
    return await proc.process_document(pdf_source=pdf_source, doc_id=doc_id, filename=filename)


async def process_documents_batch(
    pdf_sources: list[Path | bytes | str],
    job_id: str | None = None,
    filenames: list[str] | None = None,
    processor: BatchPDFProcessor | None = None,
) -> BatchProcessingResult:
    """Convenience top-level function for processing a batch of PDFs (supporting >= 10 documents)."""
    proc = processor or BatchPDFProcessor()
    return await proc.process_documents_batch(
        pdf_sources=pdf_sources,
        job_id=job_id,
        filenames=filenames,
    )
