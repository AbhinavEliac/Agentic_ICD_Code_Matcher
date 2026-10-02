"""Domain models for PDF document extraction, sections, pages, and processing outcomes."""

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class PDFExtractionStatus(StrEnum):
    """Explicit extraction outcome states for clinical PDFs."""

    SUCCESS = "SUCCESS"
    NEEDS_OCR = "NEEDS_OCR"
    INSUFFICIENT_TEXT = "INSUFFICIENT_TEXT"
    EMPTY_PDF = "EMPTY_PDF"
    MALFORMED_PDF = "MALFORMED_PDF"
    EXTRACTION_ERROR = "EXTRACTION_ERROR"


class ExtractedPage(BaseModel):
    """Data representation of a single extracted PDF page."""

    page_number: int = Field(ge=1, description="1-based page index.")
    raw_text: str = Field(description="Raw text extracted from this page.")
    normalized_text: str = Field(description="Normalized text for this page.")
    char_count: int = Field(ge=0, description="Character count of normalized page text.")
    word_count: int = Field(ge=0, description="Word count of normalized page text.")
    has_images: bool = Field(
        default=False, description="True if page contains embedded raster images."
    )
    start_char_offset: int = Field(
        default=0,
        ge=0,
        description="Start character index of this page in full document normalized text.",
    )
    end_char_offset: int = Field(
        default=0,
        ge=0,
        description="End character index of this page in full document normalized text.",
    )


class ExtractedSection(BaseModel):
    """Detected clinical section with boundary offsets and high-yield designation."""

    section_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this clinical section.",
    )
    section_type: str = Field(
        description="Canonical section category (e.g. 'DISCHARGE_DIAGNOSES', 'HOSPITAL_COURSE').",
    )
    header_text: str = Field(description="Original section header string.")
    page_number: int = Field(ge=1, description="Page where section begins.")
    start_char: int = Field(ge=0, description="Start character offset in full document text.")
    end_char: int = Field(ge=0, description="End character offset in full document text.")
    content_text: str = Field(description="Clean text content within this section.")
    is_high_yield_coding: bool = Field(
        default=False,
        description="True if section is critical for ICD coding (e.g. diagnoses, hospital course, HPI).",
    )


class ClinicalPDFDocument(BaseModel):
    """Structured representation of an ingested clinical discharge summary PDF."""

    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique document identifier.",
    )
    source_filename: str = Field(
        default="unknown.pdf",
        description="Origin filename or path string.",
    )
    total_pages: int = Field(ge=0, description="Total count of pages parsed.")
    pages: list[ExtractedPage] = Field(
        default_factory=list,
        description="Ordered sequence of page representations.",
    )
    full_normalized_text: str = Field(
        description="Full concatenated, normalized text across all pages.",
    )
    sections: list[ExtractedSection] = Field(
        default_factory=list,
        description="Detected clinical sections with character spans.",
    )
    is_scanned: bool = Field(
        default=False,
        description="True if page analysis detected image-only or scanned content.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extraction metadata (e.g. extractor used, timestamps).",
    )

    def get_coding_text(self) -> str:
        """Return targeted clinical text from high-yield coding sections.

        ARCHITECTURAL INVARIANT:
        Avoids sending entire 30+ page PDFs blindly to reasoning agents.
        If structured coding sections exist (diagnoses, hospital course, HPI),
        only those are concatenated. Otherwise falls back to full normalized text.
        """
        high_yield = [s for s in self.sections if s.is_high_yield_coding and s.content_text.strip()]
        if high_yield:
            parts = [
                f"=== {s.header_text} (Page {s.page_number}) ===\n{s.content_text}"
                for s in high_yield
            ]
            return "\n\n".join(parts)
        return self.full_normalized_text

    def get_sections_by_type(self, section_type: str) -> list[ExtractedSection]:
        """Filter sections by their canonical type."""
        target = section_type.strip().upper()
        return [s for s in self.sections if s.section_type.upper() == target]

    def find_page_number_for_char(self, char_offset: int) -> int:
        """Find the 1-based page number containing the specified character offset."""
        for page in self.pages:
            if page.start_char_offset <= char_offset <= page.end_char_offset:
                return page.page_number
        return self.pages[-1].page_number if self.pages else 1


class DocumentProcessingResult(BaseModel):
    """Independent outcome container for an individual PDF processing job."""

    document_id: str = Field(description="Unique encounter/document ID.")
    source_name: str = Field(description="Source filename or identifier.")
    status: PDFExtractionStatus = Field(description="Explicit extraction outcome status.")
    document: ClinicalPDFDocument | None = Field(
        default=None,
        description="Parsed ClinicalPDFDocument if extraction succeeded.",
    )
    error_message: str | None = Field(
        default=None,
        description="Descriptive message when extraction fails or requires OCR.",
    )
    extraction_time_ms: float = Field(ge=0.0, default=0.0, description="Extraction duration in ms.")
    extractor_used: str = Field(default="pymupdf", description="'pymupdf' or 'pdfplumber'")
    page_count: int = Field(ge=0, default=0, description="Pages parsed.")
    char_count: int = Field(ge=0, default=0, description="Characters extracted.")
    needs_ocr: bool = Field(
        default=False, description="True if document is scanned and requires OCR."
    )


class BatchProcessingResult(BaseModel):
    """Aggregated processing results for an asynchronous batch of PDFs."""

    batch_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique batch identifier.",
    )
    total_documents: int = Field(ge=0, description="Total count of PDFs submitted.")
    successful_documents: int = Field(ge=0, description="Count of successfully parsed documents.")
    failed_documents: int = Field(ge=0, description="Count of documents that encountered errors.")
    needs_ocr_documents: int = Field(
        ge=0, description="Count of documents flagged as scanned/needing OCR."
    )
    results: list[DocumentProcessingResult] = Field(
        default_factory=list,
        description="Collection of independent document results.",
    )
    total_elapsed_ms: float = Field(
        ge=0.0, default=0.0, description="Total batch elapsed time in ms."
    )
    throughput_docs_per_sec: float = Field(
        ge=0.0,
        default=0.0,
        description="Effective batch throughput in documents per second.",
    )
