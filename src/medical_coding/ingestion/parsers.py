"""Specialized parsers for PDF documents, plain text, and clinical image screenshots."""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from medical_coding.ingestion.detector import DocumentFormat
from medical_coding.pdf.extractor import PDFExtractor
from medical_coding.pdf.models import PDFExtractionStatus
from medical_coding.pdf.normalizer import DocumentNormalizer
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class IngestionResult:
    """Standardized output container across all document and image parsers."""

    def __init__(
        self,
        format: DocumentFormat,
        status: str,  # "SUCCESS", "EMPTY", "ERROR"
        raw_text: str,
        normalized_text: str,
        char_count: int = 0,
        word_count: int = 0,
        page_count: int = 1,
        is_ocr: bool = False,
        ocr_confidence: float | None = None,
        pages: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        self.format = format
        self.status = status
        self.raw_text = raw_text
        self.normalized_text = normalized_text
        self.char_count = char_count or len(normalized_text)
        self.word_count = word_count or len(normalized_text.split())
        self.page_count = page_count
        self.is_ocr = is_ocr
        self.ocr_confidence = ocr_confidence
        self.pages = pages or []
        self.metadata = metadata or {}
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        """Serialize ingestion result to dictionary."""
        return {
            "format": self.format.value,
            "status": self.status,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "page_count": self.page_count,
            "is_ocr": self.is_ocr,
            "ocr_confidence": self.ocr_confidence,
            "pages": self.pages,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


class BaseDocumentParser(ABC):
    """Abstract base class for format-specific document parsers."""

    @abstractmethod
    def parse(self, source: bytes | str | Path, filename: str | None = None, **kwargs: Any) -> IngestionResult:
        """Parse source document into standardized IngestionResult."""
        pass


class TextDocumentParser(BaseDocumentParser):
    """Parser for plain text files (.txt, .text, .md, .csv) with automatic encoding detection."""

    def __init__(self, normalizer: DocumentNormalizer | None = None) -> None:
        self.normalizer = normalizer or DocumentNormalizer()

    def parse(self, source: bytes | str | Path, filename: str | None = None, **kwargs: Any) -> IngestionResult:
        raw_text = ""
        try:
            if isinstance(source, bytes):
                # Attempt UTF-8, then Latin-1/CP1252 fallbacks
                for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                    try:
                        raw_text = source.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
            elif isinstance(source, Path):
                raw_bytes = source.read_bytes()
                return self.parse(raw_bytes, filename=filename or source.name)
            elif isinstance(source, str):
                p = Path(source)
                if p.exists() and p.is_file():
                    return self.parse(p.read_bytes(), filename=filename or p.name)
                raw_text = source

            norm_text = self.normalizer.normalize(raw_text)
            if not norm_text.strip():
                return IngestionResult(
                    format=DocumentFormat.TXT,
                    status="EMPTY",
                    raw_text="",
                    normalized_text="",
                    error_message="Plain text file is empty or contains only whitespace.",
                )

            return IngestionResult(
                format=DocumentFormat.TXT,
                status="SUCCESS",
                raw_text=raw_text,
                normalized_text=norm_text,
                char_count=len(norm_text),
                word_count=len(norm_text.split()),
                page_count=1,
                is_ocr=False,
                pages=[{
                    "page_number": 1,
                    "normalized_text": norm_text,
                    "char_count": len(norm_text),
                    "word_count": len(norm_text.split()),
                }],
                metadata={"source_filename": filename or "document.txt", "parser": "TextDocumentParser"},
            )
        except Exception as exc:
            logger.exception("Failed to parse text document: %s", exc)
            return IngestionResult(
                format=DocumentFormat.TXT,
                status="ERROR",
                raw_text="",
                normalized_text="",
                error_message=f"Text parsing error: {exc}",
            )


class ImageOCRDocumentParser(BaseDocumentParser):
    """OCR parser for clinical screenshots and images (.png, .jpg, .jpeg, .tiff, .bmp, .webp)."""

    def __init__(self, normalizer: DocumentNormalizer | None = None) -> None:
        self.normalizer = normalizer or DocumentNormalizer()
        self._ocr_engine: Any = None

    def _get_ocr_engine(self) -> Any:
        if self._ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr_engine = RapidOCR()
            except ImportError:
                logger.warning("rapidocr_onnxruntime not installed; OCR parsing unavailable.")
                self._ocr_engine = None
        return self._ocr_engine

    @staticmethod
    def _repair_ocr_text(text: str) -> str:
        """Heal common OCR subword spacing artifacts produced by bitmap rendering."""
        repairs = [
            (r"\bse\s+psis\b", "sepsis"),
            (r"\bdisc\s+harge\b", "discharge"),
            (r"\bhyper\s+tension\b", "hypertension"),
            (r"\bdia\s+betes\b", "diabetes"),
            (r"\bmyo\s+cardial\b", "myocardial"),
            (r"\bin\s+farction\b", "infarction"),
            (r"\bpneu\s+monia\b", "pneumonia"),
            (r"\bsys\s+tolic\b", "systolic"),
            (r"\bdias\s+tolic\b", "diastolic"),
            (r"\bcon\s+gestive\b", "congestive"),
            (r"\bexacer\s+bation\b", "exacerbation"),
            (r"\bobstruc\s+tive\b", "obstructive"),
            (r"\bpulmo\s+nary\b", "pulmonary"),
            (r"\bkid\s+ney\b", "kidney"),
            (r"\bfib\s+rillation\b", "fibrillation"),
        ]
        repaired = text
        for pattern, replacement in repairs:
            repaired = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)
        return repaired

    def parse(self, source: bytes | str | Path, filename: str | None = None, **kwargs: Any) -> IngestionResult:
        """Run OCR on image bytes or path to extract clinical text."""
        img_bytes: bytes
        try:
            if isinstance(source, bytes):
                img_bytes = source
            elif isinstance(source, Path):
                img_bytes = source.read_bytes()
            elif isinstance(source, str):
                p = Path(source)
                if p.exists() and p.is_file():
                    img_bytes = p.read_bytes()
                else:
                    return IngestionResult(
                        format=DocumentFormat.IMAGE,
                        status="ERROR",
                        raw_text="",
                        normalized_text="",
                        error_message=f"Image file not found at '{source}'",
                    )
            else:
                return IngestionResult(
                    format=DocumentFormat.IMAGE,
                    status="ERROR",
                    raw_text="",
                    normalized_text="",
                    error_message=f"Unsupported image source type: {type(source)}",
                )

            ocr = self._get_ocr_engine()
            if ocr is None:
                return IngestionResult(
                    format=DocumentFormat.IMAGE,
                    status="ERROR",
                    raw_text="",
                    normalized_text="",
                    error_message="RapidOCR engine is not installed or failed to initialize.",
                )

            # Execute OCR inference
            ocr_results, _ = ocr(img_bytes)

            if not ocr_results:
                return IngestionResult(
                    format=DocumentFormat.IMAGE,
                    status="EMPTY",
                    raw_text="",
                    normalized_text="",
                    is_ocr=True,
                    error_message="No text detected in image screenshot.",
                )

            # Sort detected boxes top-to-bottom, left-to-right
            # ocr_results format: [ [box_coords, text, confidence_str], ... ]
            sorted_results = sorted(
                ocr_results,
                key=lambda item: (item[0][0][1], item[0][0][0]),  # y0 then x0
            )

            extracted_lines: list[str] = []
            confidences: list[float] = []

            for item in sorted_results:
                line_text = str(item[1]).strip()
                if line_text:
                    extracted_lines.append(line_text)
                    try:
                        confidences.append(float(item[2]))
                    except (ValueError, TypeError):
                        pass

            raw_text = "\n".join(extracted_lines)
            repaired_text = self._repair_ocr_text(raw_text)
            norm_text = self.normalizer.normalize(repaired_text)
            avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.85

            return IngestionResult(
                format=DocumentFormat.IMAGE,
                status="SUCCESS",
                raw_text=repaired_text,
                normalized_text=norm_text,
                char_count=len(norm_text),
                word_count=len(norm_text.split()),
                page_count=1,
                is_ocr=True,
                ocr_confidence=round(avg_conf, 3),
                pages=[{
                    "page_number": 1,
                    "normalized_text": norm_text,
                    "char_count": len(norm_text),
                    "word_count": len(norm_text.split()),
                    "ocr_confidence": round(avg_conf, 3),
                }],
                metadata={
                    "source_filename": filename or "screenshot.png",
                    "lines_detected": len(extracted_lines),
                    "average_confidence": round(avg_conf, 3),
                    "parser": "ImageOCRDocumentParser",
                },
            )

        except Exception as exc:
            logger.exception("OCR extraction failed on image: %s", exc)
            return IngestionResult(
                format=DocumentFormat.IMAGE,
                status="ERROR",
                raw_text="",
                normalized_text="",
                error_message=f"Image OCR failed: {exc}",
            )


class PDFDocumentParser(BaseDocumentParser):
    """Parser for clinical PDFs using PyMuPDF and pdfplumber with OCR fallback for scanned pages."""

    def __init__(
        self,
        max_pages: int = 50,
        normalizer: DocumentNormalizer | None = None,
        image_ocr_parser: ImageOCRDocumentParser | None = None,
    ) -> None:
        self.max_pages = max_pages
        self.normalizer = normalizer or DocumentNormalizer()
        self.extractor = PDFExtractor(max_pages=max_pages, normalizer=self.normalizer)
        self.image_ocr = image_ocr_parser or ImageOCRDocumentParser(normalizer=self.normalizer)

    def parse(self, source: bytes | str | Path, filename: str | None = None, **kwargs: Any) -> IngestionResult:
        """Extract text from PDF, automatically falling back to OCR if scanned."""
        extract_res = self.extractor.extract_sync(source)

        # If extraction succeeded and text volume is sufficient, return directly
        if (
            extract_res.status == PDFExtractionStatus.SUCCESS
            and len(extract_res.full_normalized_text.strip()) >= 50
        ):
            pages_list = [
                {
                    "page_number": p.page_number,
                    "normalized_text": p.normalized_text,
                    "char_count": p.char_count,
                    "word_count": p.word_count,
                    "has_images": p.has_images,
                }
                for p in extract_res.pages
            ]
            return IngestionResult(
                format=DocumentFormat.PDF,
                status="SUCCESS",
                raw_text=extract_res.full_raw_text,
                normalized_text=extract_res.full_normalized_text,
                char_count=len(extract_res.full_normalized_text),
                word_count=len(extract_res.full_normalized_text.split()),
                page_count=len(extract_res.pages),
                is_ocr=False,
                pages=pages_list,
                metadata={
                    "source_filename": filename or "document.pdf",
                    "extractor_used": extract_res.extractor_used,
                    "parser": "PDFDocumentParser",
                },
            )

        # Scanned PDF or insufficient text: Fall back to OCR on rendered page pixmaps
        if extract_res.is_scanned or extract_res.status in (
            PDFExtractionStatus.NEEDS_OCR,
            PDFExtractionStatus.INSUFFICIENT_TEXT,
        ):
            logger.info("PDF appears to be scanned or contains raster images. Invoking RapidOCR on pages...")
            try:
                import pymupdf

                pdf_doc = (
                    pymupdf.open(stream=source, filetype="pdf")
                    if isinstance(source, bytes)
                    else pymupdf.open(str(source))
                )

                ocr_pages: list[dict[str, Any]] = []
                full_ocr_parts: list[str] = []
                total_conf: list[float] = []

                for page_idx in range(min(len(pdf_doc), self.max_pages)):
                    page = pdf_doc.load_page(page_idx)
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")

                    page_res = self.image_ocr.parse(img_bytes, filename=f"page_{page_idx+1}.png")
                    if page_res.status == "SUCCESS":
                        full_ocr_parts.append(page_res.normalized_text)
                        if page_res.ocr_confidence is not None:
                            total_conf.append(page_res.ocr_confidence)
                        ocr_pages.append({
                            "page_number": page_idx + 1,
                            "normalized_text": page_res.normalized_text,
                            "char_count": page_res.char_count,
                            "word_count": page_res.word_count,
                            "has_images": True,
                            "is_ocr": True,
                        })

                pdf_doc.close()

                full_ocr_text = "\n\n".join(full_ocr_parts).strip()
                if full_ocr_text:
                    avg_c = (sum(total_conf) / len(total_conf)) if total_conf else 0.85
                    return IngestionResult(
                        format=DocumentFormat.PDF,
                        status="SUCCESS",
                        raw_text=full_ocr_text,
                        normalized_text=full_ocr_text,
                        char_count=len(full_ocr_text),
                        word_count=len(full_ocr_text.split()),
                        page_count=len(ocr_pages),
                        is_ocr=True,
                        ocr_confidence=round(avg_c, 3),
                        pages=ocr_pages,
                        metadata={
                            "source_filename": filename or "document.pdf",
                            "is_scanned": True,
                            "ocr_fallback": True,
                            "parser": "PDFDocumentParser[OCR]",
                        },
                    )
            except Exception as ocr_exc:
                logger.warning("OCR fallback for scanned PDF encountered error: %s", ocr_exc)

        # If still failed, return extraction error
        return IngestionResult(
            format=DocumentFormat.PDF,
            status="ERROR" if extract_res.status == PDFExtractionStatus.MALFORMED_PDF else "EMPTY",
            raw_text=extract_res.full_raw_text,
            normalized_text=extract_res.full_normalized_text,
            error_message=extract_res.error_message or "Failed to extract readable text from PDF.",
        )


class ManualTextParser(BaseDocumentParser):
    """Parser for direct clinical text entered via web UI or API."""

    def __init__(self, normalizer: DocumentNormalizer | None = None) -> None:
        self.normalizer = normalizer or DocumentNormalizer()

    def parse(self, source: bytes | str | Path, filename: str | None = None, **kwargs: Any) -> IngestionResult:
        text_str = source.decode("utf-8") if isinstance(source, bytes) else str(source)
        norm_text = self.normalizer.normalize(text_str)

        if not norm_text.strip():
            return IngestionResult(
                format=DocumentFormat.MANUAL_TEXT,
                status="EMPTY",
                raw_text="",
                normalized_text="",
                error_message="Manual text input is empty.",
            )

        return IngestionResult(
            format=DocumentFormat.MANUAL_TEXT,
            status="SUCCESS",
            raw_text=text_str,
            normalized_text=norm_text,
            char_count=len(norm_text),
            word_count=len(norm_text.split()),
            page_count=1,
            is_ocr=False,
            pages=[{
                "page_number": 1,
                "normalized_text": norm_text,
                "char_count": len(norm_text),
                "word_count": len(norm_text.split()),
            }],
            metadata={"source_type": "manual_entry", "parser": "ManualTextParser"},
        )
