"""Unit and integration tests for PDF extraction, normalization, section detection, and batch concurrency."""

import io
from pathlib import Path

import PIL.Image
import pymupdf
import pytest

from medical_coding.pdf.extractor import PDFExtractor
from medical_coding.pdf.models import PDFExtractionStatus
from medical_coding.pdf.normalizer import DocumentNormalizer
from medical_coding.pdf.processor import (
    BatchPDFProcessor,
    process_document,
    process_documents_batch,
)
from medical_coding.pdf.section_detector import SectionDetector


def generate_sample_clinical_pdf(
    patient_name: str = "DOE, JOHN",
    diagnoses: list[str] | None = None,
    num_pages: int = 1,
) -> bytes:
    """Generate a clean synthetic clinical discharge summary PDF in memory using PyMuPDF."""
    doc = pymupdf.open()
    diag_list = diagnoses or [
        "1. Acute systolic (congestive) heart failure (I50.21)",
        "2. Type 2 diabetes mellitus without complications (E11.9)",
        "3. Essential hypertension (I10)",
    ]
    diag_text = "\n".join(diag_list)

    page1_text = f"""DISCHARGE SUMMARY

PATIENT NAME: {patient_name}
MRN: 987654321
ADMISSION DATE: 2026-09-20
DISCHARGE DATE: 2026-09-28

DISCHARGE DIAGNOSES:
{diag_text}

HISTORY OF PRESENT ILLNESS:
The patient is a 68-year-old male who presented with severe progressive dyspnea on exertion,
orthopnea, and 3+ bilateral lower extremity edema over the past 5 days.

HOSPITAL COURSE:
Patient was admitted to cardiology service and promptly initiated on intravenous furosemide diuresis.
Echocardiogram revealed severely depressed left ventricular ejection fraction of 25%.
Patient experienced significant clinical improvement with 4 kg fluid loss.

PAST MEDICAL HISTORY:
1. Longstanding hypertension.
2. Type 2 diabetes mellitus diagnosed 10 years ago.
3. No prior history of myocardial infarction.
"""

    page = doc.new_page()
    page.insert_text((40, 50), page1_text, fontsize=10)

    for p in range(1, num_pages):
        extra_page = doc.new_page()
        extra_page.insert_text(
            (40, 50),
            f"DISCHARGE MEDICATIONS (Page {p + 1}):\n1. Furosemide 40 mg daily\n2. Lisinopril 10 mg daily\n3. Metformin 500 mg BID\n\nDISCHARGE INSTRUCTIONS:\nFollow up with cardiology clinic in 2 weeks.",
            fontsize=10,
        )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def generate_scanned_image_pdf() -> bytes:
    """Generate an image-only PDF containing raster pixels and minimal/no text."""
    doc = pymupdf.open()
    page = doc.new_page()

    # Create dummy raster image
    img = PIL.Image.new("RGB", (200, 200), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    page.insert_image(pymupdf.Rect(50, 50, 400, 400), stream=buf.getvalue())
    # Add minimal header text (below 20 chars)
    page.insert_text((20, 20), "Scan", fontsize=8)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def generate_insufficient_text_pdf() -> bytes:
    """Generate a PDF with under 50 characters of total text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Short note.", fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ============================================================================
# Tests: Normalization and Section Detection
# ============================================================================


def test_whitespace_normalization() -> None:
    """Verify whitespace normalizer cleans control chars, de-hyphenates, and preserves paragraphs."""
    normalizer = DocumentNormalizer()
    raw = "Patient  with   hyper-\n  tension    and dysp-\nnea.\x00\x08\n\n\n\nPlan:\n1. Diuresis"
    normalized = normalizer.normalize(raw)

    assert "hypertension" in normalized
    assert "dyspnea" in normalized
    assert "\x00" not in normalized
    assert "\x08" not in normalized
    # Max two consecutive newlines between paragraphs
    assert "\n\n\n" not in normalized
    assert "Plan:\n1. Diuresis" in normalized


def test_section_detector_identifies_clinical_boundaries() -> None:
    """Verify SectionDetector detects headers, sets offsets, and flags high-yield coding sections."""
    detector = SectionDetector()
    text = """PATIENT HEADER: John Doe MRN 123
DISCHARGE DIAGNOSES:
1. Acute systolic heart failure
2. Essential hypertension

HOSPITAL COURSE:
Patient received diuresis and improved clinically.

DISCHARGE MEDICATIONS:
1. Furosemide 40mg
"""
    sections = detector.detect_sections(text)
    assert len(sections) >= 3

    types = [s.section_type for s in sections]
    assert "DISCHARGE_DIAGNOSES" in types
    assert "HOSPITAL_COURSE" in types
    assert "MEDICATIONS" in types

    # Check high-yield flags
    diag_sec = next(s for s in sections if s.section_type == "DISCHARGE_DIAGNOSES")
    assert diag_sec.is_high_yield_coding is True
    assert "Acute systolic heart failure" in diag_sec.content_text

    med_sec = next(s for s in sections if s.section_type == "MEDICATIONS")
    assert med_sec.is_high_yield_coding is False


# ============================================================================
# Tests: Single and Batch PDF Processing
# ============================================================================


@pytest.mark.asyncio
async def test_process_single_pdf_success() -> None:
    """Verify processing a single valid PDF into a structured ClinicalPDFDocument."""
    pdf_bytes = generate_sample_clinical_pdf(patient_name="SMITH, JANE", num_pages=2)
    result = await process_document(pdf_source=pdf_bytes, filename="jane_smith.pdf")

    assert result.status == PDFExtractionStatus.SUCCESS
    assert result.document is not None
    assert result.page_count == 2
    assert result.char_count > 100
    assert result.needs_ocr is False

    doc = result.document
    assert doc.source_filename == "jane_smith.pdf"
    assert len(doc.pages) == 2
    assert len(doc.sections) >= 3

    # Verify high-yield coding text extraction
    coding_text = doc.get_coding_text()
    assert "DISCHARGE DIAGNOSES" in coding_text
    assert "Acute systolic" in coding_text


@pytest.mark.asyncio
async def test_process_ten_pdfs_batch() -> None:
    """Requirement: Process a batch of 10 PDFs asynchronously without creating 10 LLM instances."""
    batch_size = 10
    pdf_sources = [
        generate_sample_clinical_pdf(patient_name=f"PATIENT_{i:02d}", num_pages=1)
        for i in range(batch_size)
    ]
    filenames = [f"patient_{i:02d}.pdf" for i in range(batch_size)]

    processor = BatchPDFProcessor(max_concurrency=10)
    batch_result = await processor.process_documents_batch(
        pdf_sources=pdf_sources,
        filenames=filenames,
        job_id="test-batch-10-docs",
    )

    assert batch_result.total_documents == 10
    assert batch_result.successful_documents == 10
    assert batch_result.failed_documents == 0
    assert len(batch_result.results) == 10

    # Ensure every document has a unique ID and successful status
    doc_ids = {r.document_id for r in batch_result.results}
    assert len(doc_ids) == 10
    for r in batch_result.results:
        assert r.status == PDFExtractionStatus.SUCCESS
        assert r.document is not None
        assert r.page_count == 1


# ============================================================================
# Tests: Edge Cases, Malformed, Empty, and Scanned PDFs
# ============================================================================


@pytest.mark.asyncio
async def test_failed_nonexistent_pdf_file() -> None:
    """Verify that a non-existent PDF file path returns EXTRACTION_ERROR without crashing."""
    missing_path = Path("./nonexistent_directory/missing_file_123.pdf")
    result = await process_document(pdf_source=missing_path)

    assert result.status == PDFExtractionStatus.EXTRACTION_ERROR
    assert result.document is None
    assert "not found" in (result.error_message or "").lower()


@pytest.mark.asyncio
async def test_empty_pdf() -> None:
    """Verify that an empty 0-byte stream returns EMPTY_PDF status."""
    empty_bytes = b""
    result = await process_document(pdf_source=empty_bytes, filename="empty.pdf")

    assert result.status == PDFExtractionStatus.EMPTY_PDF
    assert result.document is None
    assert "empty" in (result.error_message or "").lower()


@pytest.mark.asyncio
async def test_malformed_pdf_corrupted_bytes() -> None:
    """Verify that corrupted binary data returns MALFORMED_PDF without raising unhandled exceptions."""
    corrupted_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nGARBAGE_DATA_CORRUPT_EOF"
    result = await process_document(pdf_source=corrupted_bytes, filename="corrupt.pdf")

    assert result.status == PDFExtractionStatus.MALFORMED_PDF
    assert result.document is None
    assert result.error_message is not None


@pytest.mark.asyncio
async def test_insufficient_extracted_text() -> None:
    """Verify that a document with minimal text (< 50 chars) returns INSUFFICIENT_TEXT."""
    short_pdf = generate_insufficient_text_pdf()
    result = await process_document(pdf_source=short_pdf, filename="short.pdf")

    assert result.status == PDFExtractionStatus.INSUFFICIENT_TEXT
    assert result.document is None
    assert result.char_count < 50


@pytest.mark.asyncio
async def test_scanned_pdf_detects_needs_ocr() -> None:
    """Requirement: Gracefully detect scanned raster PDFs and return NEEDS_OCR without inventing text."""
    scanned_bytes = generate_scanned_image_pdf()
    extractor = PDFExtractor(min_page_chars_for_scanned=20, min_doc_chars=50)
    processor = BatchPDFProcessor(extractor=extractor)

    result = await processor.process_document(pdf_source=scanned_bytes, filename="scanned_scan.pdf")

    assert result.status == PDFExtractionStatus.NEEDS_OCR
    assert result.needs_ocr is True
    assert result.document is None
    assert "ocr is required" in (result.error_message or "").lower()


# ============================================================================
# Tests: Batch Fault Tolerance & Throughput Benchmark
# ============================================================================


@pytest.mark.asyncio
async def test_batch_fault_tolerance_mixed_payload() -> None:
    """Verify that a batch with mixed valid and broken PDFs completes with isolated statuses."""
    valid_pdf = generate_sample_clinical_pdf("PATIENT_VALID")
    corrupt_pdf = b"NOT_A_REAL_PDF"
    empty_pdf = b""
    short_pdf = generate_insufficient_text_pdf()

    sources = [valid_pdf, corrupt_pdf, empty_pdf, short_pdf, valid_pdf]
    filenames = ["valid1.pdf", "corrupt.pdf", "empty.pdf", "short.pdf", "valid2.pdf"]

    batch_res = await process_documents_batch(
        pdf_sources=sources,
        filenames=filenames,
    )

    assert batch_res.total_documents == 5
    assert batch_res.successful_documents == 2
    assert batch_res.failed_documents == 3

    statuses = [r.status for r in batch_res.results]
    assert statuses[0] == PDFExtractionStatus.SUCCESS
    assert statuses[1] == PDFExtractionStatus.MALFORMED_PDF
    assert statuses[2] == PDFExtractionStatus.EMPTY_PDF
    assert statuses[3] == PDFExtractionStatus.INSUFFICIENT_TEXT
    assert statuses[4] == PDFExtractionStatus.SUCCESS


@pytest.mark.asyncio
async def test_throughput_benchmark_ten_documents() -> None:
    """Benchmark extraction throughput on 10 multi-page clinical discharge summaries."""
    batch_size = 10
    sources = [
        generate_sample_clinical_pdf(patient_name=f"BENCHMARK_{i}", num_pages=2)
        for i in range(batch_size)
    ]

    batch_result = await process_documents_batch(sources)

    assert batch_result.total_documents == 10
    assert batch_result.successful_documents == 10
    assert batch_result.total_elapsed_ms > 0
    assert batch_result.throughput_docs_per_sec > 0

    print(
        f"\n[Throughput Benchmark] 10 Multi-page PDFs (2 pages each): "
        f"Total time = {batch_result.total_elapsed_ms:.2f} ms | "
        f"Throughput = {batch_result.throughput_docs_per_sec:.2f} docs/sec"
    )
