"""Unit tests for multimodal document ingestion, auto-detection, and OCR parsing."""

import io

import pymupdf
import pytest
from PIL import Image, ImageDraw

from medical_coding.ingestion.agent import ClinicalDocumentIngestionAgent
from medical_coding.ingestion.detector import DocumentFormat, DocumentFormatDetector
from medical_coding.ingestion.parsers import (
    ImageOCRDocumentParser,
    ManualTextParser,
    PDFDocumentParser,
    TextDocumentParser,
)


@pytest.fixture
def ingestion_agent() -> ClinicalDocumentIngestionAgent:
    """Fixture providing initialized ClinicalDocumentIngestionAgent."""
    return ClinicalDocumentIngestionAgent(max_pdf_pages=10)


def test_detector_magic_bytes_and_extensions() -> None:
    """Test format detection from binary magic bytes and file extensions."""
    # PDF detection
    pdf_header = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
    fmt, mime = DocumentFormatDetector.detect_from_bytes(pdf_header, filename="test.pdf")
    assert fmt == DocumentFormat.PDF
    assert mime == "application/pdf"

    # PNG detection
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    fmt, mime = DocumentFormatDetector.detect_from_bytes(png_header)
    assert fmt == DocumentFormat.IMAGE
    assert mime == "image/png"

    # JPEG detection
    jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    fmt, mime = DocumentFormatDetector.detect_from_bytes(jpeg_header)
    assert fmt == DocumentFormat.IMAGE
    assert mime == "image/jpeg"

    # Plain text detection
    txt_bytes = b"Patient admitted with acute chest pain."
    fmt, mime = DocumentFormatDetector.detect_from_bytes(txt_bytes, filename="notes.txt")
    assert fmt == DocumentFormat.TXT
    assert mime == "text/plain"

    # Manual text string
    fmt, mime = DocumentFormatDetector.detect_format("Manual clinical note entry.\nNo history of asthma.")
    assert fmt == DocumentFormat.MANUAL_TEXT
    assert mime == "text/plain"


def test_text_document_parser() -> None:
    """Test TextDocumentParser handles UTF-8 bytes and plain text."""
    parser = TextDocumentParser()
    content = "DISCHARGE SUMMARY\nPatient has hypertension and diabetes."
    res = parser.parse(content.encode("utf-8"), filename="summary.txt")

    assert res.format == DocumentFormat.TXT
    assert res.status == "SUCCESS"
    assert res.is_ocr is False
    assert "hypertension" in res.normalized_text
    assert res.word_count > 0

    # Empty text
    empty_res = parser.parse(b"   \n\t  ", filename="empty.txt")
    assert empty_res.status == "EMPTY"


def test_manual_text_parser() -> None:
    """Test ManualTextParser cleans whitespace and parses raw text strings."""
    parser = ManualTextParser()
    raw = "   Patient   admitted with   severe dyspnea.   "
    res = parser.parse(raw)

    assert res.format == DocumentFormat.MANUAL_TEXT
    assert res.status == "SUCCESS"
    assert res.normalized_text == "Patient admitted with severe dyspnea."


def test_pdf_document_parser() -> None:
    """Test PDFDocumentParser extracts text from PyMuPDF vector PDF."""
    parser = PDFDocumentParser(max_pages=5)

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "DISCHARGE SUMMARY\nDiagnosis: Acute systolic heart failure.")
    pdf_bytes = doc.write()
    doc.close()

    res = parser.parse(pdf_bytes, filename="discharge.pdf")
    assert res.format == DocumentFormat.PDF
    assert res.status == "SUCCESS"
    assert "heart failure" in res.normalized_text
    assert res.page_count == 1
    assert res.is_ocr is False


def test_image_ocr_parser() -> None:
    """Test ImageOCRDocumentParser executes OCR on synthetic image screenshot."""
    parser = ImageOCRDocumentParser()

    # Generate synthetic screenshot with PIL
    img = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), "CLINICAL SUMMARY: Acute myocardial infarction anterior wall", fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    res = parser.parse(png_bytes, filename="screenshot.png")
    assert res.format == DocumentFormat.IMAGE
    assert res.status == "SUCCESS"
    assert res.is_ocr is True
    assert res.ocr_confidence is not None
    assert "myocardial" in res.normalized_text.lower() or "infarction" in res.normalized_text.lower()


def test_agent_auto_detects_and_routes(ingestion_agent: ClinicalDocumentIngestionAgent) -> None:
    """Test ClinicalDocumentIngestionAgent end-to-end routing across all four modalities."""
    # 1. Manual Text
    r1 = ingestion_agent.ingest("Patient presented with fever and cough.\nDiagnosed with community acquired pneumonia.")
    assert r1.format == DocumentFormat.MANUAL_TEXT
    assert r1.status == "SUCCESS"

    # 2. Text File bytes
    txt_data = b"DISCHARGE NOTE\nEssential hypertension under control."
    r2 = ingestion_agent.ingest(txt_data, filename="encounter.txt")
    assert r2.format == DocumentFormat.TXT
    assert r2.status == "SUCCESS"
    assert "hypertension" in r2.normalized_text

    # 3. PDF bytes
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "DISCHARGE SUMMARY\nDiagnosis: Acute systolic congestive heart failure.")
    pdf_bytes = doc.write()
    doc.close()
    r3 = ingestion_agent.ingest(pdf_bytes, filename="cardiology.pdf")
    assert r3.format == DocumentFormat.PDF
    assert r3.status == "SUCCESS"
    assert "heart failure" in r3.normalized_text

    # 4. Image Screenshot bytes
    img = Image.new("RGB", (600, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "DISCHARGE RECORD: Sepsis unspecified organism", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_data = buf.getvalue()
    r4 = ingestion_agent.ingest(img_data, filename="chart_screenshot.png")
    assert r4.format == DocumentFormat.IMAGE
    assert r4.status == "SUCCESS"
    assert r4.is_ocr is True
    assert "sepsis" in r4.normalized_text.lower()
