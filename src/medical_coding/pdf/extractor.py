"""PyMuPDF primary and pdfplumber fallback text extractor with scanned PDF detection."""

import asyncio
import io
from pathlib import Path
from typing import Any

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pymupdf  # Modern PyMuPDF import
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

from medical_coding.pdf.models import ExtractedPage, PDFExtractionStatus
from medical_coding.pdf.normalizer import DocumentNormalizer
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class ExtractorResult:
    """Internal container for raw page extraction output."""

    def __init__(
        self,
        status: PDFExtractionStatus,
        pages: list[ExtractedPage],
        full_raw_text: str,
        full_normalized_text: str,
        is_scanned: bool = False,
        error_message: str | None = None,
        extractor_used: str = "pymupdf",
    ) -> None:
        self.status = status
        self.pages = pages
        self.full_raw_text = full_raw_text
        self.full_normalized_text = full_normalized_text
        self.is_scanned = is_scanned
        self.error_message = error_message
        self.extractor_used = extractor_used


class PDFExtractor:
    """Primary PyMuPDF extractor with pdfplumber fallback, page tracking, and scanned detection."""

    def __init__(
        self,
        max_pages: int = 50,
        min_doc_chars: int = 50,
        min_page_chars_for_scanned: int = 20,
        normalizer: DocumentNormalizer | None = None,
    ) -> None:
        self.max_pages = max_pages
        self.min_doc_chars = min_doc_chars
        self.min_page_chars_for_scanned = min_page_chars_for_scanned
        self.normalizer = normalizer or DocumentNormalizer()

    def extract_sync(self, source: Path | bytes | str) -> ExtractorResult:
        """Extract pages synchronously using PyMuPDF, falling back to pdfplumber if necessary."""
        # 1. Attempt primary extraction using PyMuPDF if available
        if pymupdf is not None:
            try:
                result = self._extract_with_pymupdf(source)
                # If PyMuPDF returned text, return its result directly
                if result.status == PDFExtractionStatus.SUCCESS:
                    return result
                # If PyMuPDF flagged scanned/needs OCR or empty, test pdfplumber before deciding
                if result.status in (
                    PDFExtractionStatus.NEEDS_OCR,
                    PDFExtractionStatus.INSUFFICIENT_TEXT,
                ):
                    if pdfplumber is not None:
                        logger.info("PyMuPDF yielded low text count. Attempting pdfplumber fallback...")
                        fallback_result = self._extract_with_pdfplumber(source)
                        if fallback_result.status == PDFExtractionStatus.SUCCESS:
                            logger.info("pdfplumber fallback succeeded with valid text.")
                            return fallback_result
                return result

            except Exception as exc:
                logger.warning(
                    "PyMuPDF raised exception on '%s': %s. Attempting pdfplumber fallback...",
                    source,
                    exc,
                )
                if pdfplumber is not None:
                    try:
                        return self._extract_with_pdfplumber(source)
                    except Exception as fb_exc:
                        logger.error("Both PyMuPDF and pdfplumber failed on '%s': %s", source, fb_exc)
                        return ExtractorResult(
                            status=PDFExtractionStatus.MALFORMED_PDF,
                            pages=[],
                            full_raw_text="",
                            full_normalized_text="",
                            error_message=f"Failed to parse PDF document: {exc}; Fallback error: {fb_exc}",
                            extractor_used="fallback",
                        )
                return ExtractorResult(
                    status=PDFExtractionStatus.MALFORMED_PDF,
                    pages=[],
                    full_raw_text="",
                    full_normalized_text="",
                    error_message=f"Failed to parse PDF document: {exc}",
                    extractor_used="pymupdf",
                )
        elif pdfplumber is not None:
            return self._extract_with_pdfplumber(source)
        else:
            return ExtractorResult(
                status=PDFExtractionStatus.EXTRACTION_ERROR,
                pages=[],
                full_raw_text="",
                full_normalized_text="",
                error_message="Neither PyMuPDF nor pdfplumber is available in current environment.",
                extractor_used="none",
            )

    def _extract_with_pymupdf(self, source: Path | bytes | str) -> ExtractorResult:
        """Execute extraction using PyMuPDF (fitz)."""
        doc: Any = None
        try:
            if isinstance(source, bytes):
                if not source or len(source) < 10:
                    return ExtractorResult(
                        status=PDFExtractionStatus.EMPTY_PDF,
                        pages=[],
                        full_raw_text="",
                        full_normalized_text="",
                        error_message="PDF byte stream is empty or truncated.",
                    )
                doc = pymupdf.open(stream=source, filetype="pdf")
            else:
                path = Path(source)
                if not path.exists():
                    return ExtractorResult(
                        status=PDFExtractionStatus.EXTRACTION_ERROR,
                        pages=[],
                        full_raw_text="",
                        full_normalized_text="",
                        error_message=f"PDF file not found at '{path}'",
                    )
                if path.stat().st_size == 0:
                    return ExtractorResult(
                        status=PDFExtractionStatus.EMPTY_PDF,
                        pages=[],
                        full_raw_text="",
                        full_normalized_text="",
                        error_message="PDF file is 0 bytes.",
                    )
                doc = pymupdf.open(str(path))

            total_pages = len(doc)
            if total_pages == 0:
                return ExtractorResult(
                    status=PDFExtractionStatus.EMPTY_PDF,
                    pages=[],
                    full_raw_text="",
                    full_normalized_text="",
                    error_message="PDF contains 0 pages.",
                )

            pages_to_read = min(total_pages, self.max_pages)
            extracted_pages: list[ExtractedPage] = []
            raw_parts: list[str] = []
            normalized_parts: list[str] = []
            current_offset = 0

            scanned_pages_count = 0
            image_pages_count = 0

            for page_idx in range(pages_to_read):
                page = doc.load_page(page_idx)
                raw_page_text = page.get_text("text") or ""
                norm_page_text = self.normalizer.normalize(raw_page_text)

                images = page.get_images()
                has_images = len(images) > 0
                if has_images:
                    image_pages_count += 1

                char_cnt = len(norm_page_text)
                word_cnt = len(norm_page_text.split())

                is_scanned_suspicion = has_images and (char_cnt < self.min_page_chars_for_scanned)
                if is_scanned_suspicion:
                    scanned_pages_count += 1

                end_offset = current_offset + len(norm_page_text)

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_idx + 1,
                        raw_text=raw_page_text,
                        normalized_text=norm_page_text,
                        char_count=char_cnt,
                        word_count=word_cnt,
                        has_images=has_images,
                        start_char_offset=current_offset,
                        end_char_offset=end_offset,
                    )
                )

                raw_parts.append(raw_page_text)
                normalized_parts.append(norm_page_text)
                # Account for page-joining double newline
                current_offset = end_offset + 2

            full_raw = "\n\n".join(raw_parts).strip()
            full_normalized = "\n\n".join(normalized_parts).strip()

            total_chars = len(full_normalized)

            # Detect scanned / image-only PDFs
            if (
                pages_to_read > 0
                and scanned_pages_count == pages_to_read
                and total_chars < self.min_doc_chars
            ):
                return ExtractorResult(
                    status=PDFExtractionStatus.NEEDS_OCR,
                    pages=extracted_pages,
                    full_raw_text=full_raw,
                    full_normalized_text=full_normalized,
                    is_scanned=True,
                    error_message=(
                        f"Scanned image PDF detected ({image_pages_count} image pages, {total_chars} text chars). "
                        "OCR is required to process this document; automatic text extraction cannot read raster images."
                    ),
                    extractor_used="pymupdf",
                )

            # Detect insufficient text
            if total_chars < self.min_doc_chars:
                return ExtractorResult(
                    status=PDFExtractionStatus.INSUFFICIENT_TEXT,
                    pages=extracted_pages,
                    full_raw_text=full_raw,
                    full_normalized_text=full_normalized,
                    is_scanned=image_pages_count > 0,
                    error_message=(
                        f"Extracted text volume ({total_chars} chars) is below minimum threshold ({self.min_doc_chars} chars)."
                    ),
                    extractor_used="pymupdf",
                )

            return ExtractorResult(
                status=PDFExtractionStatus.SUCCESS,
                pages=extracted_pages,
                full_raw_text=full_raw,
                full_normalized_text=full_normalized,
                is_scanned=False,
                extractor_used="pymupdf",
            )

        except Exception as exc:
            return ExtractorResult(
                status=PDFExtractionStatus.MALFORMED_PDF,
                pages=[],
                full_raw_text="",
                full_normalized_text="",
                error_message=f"PyMuPDF error: {exc}",
                extractor_used="pymupdf",
            )
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    def _extract_with_pdfplumber(self, source: Path | bytes | str) -> ExtractorResult:
        """Fallback extraction using pdfplumber."""
        if pdfplumber is None:
            logger.info("pdfplumber library is not installed; skipping fallback extraction.")
            return ExtractorResult(
                status=PDFExtractionStatus.EXTRACTION_ERROR,
                pages=[],
                full_raw_text="",
                full_normalized_text="",
                error_message="pdfplumber library is not installed.",
                extractor_used="pdfplumber",
            )

        file_obj: Any = None
        pdf: Any = None
        try:
            if isinstance(source, bytes):
                file_obj = io.BytesIO(source)
                pdf = pdfplumber.open(file_obj)
            else:
                path = Path(source)
                if not path.exists():
                    return ExtractorResult(
                        status=PDFExtractionStatus.EXTRACTION_ERROR,
                        pages=[],
                        full_raw_text="",
                        full_normalized_text="",
                        error_message=f"PDF file not found at '{path}'",
                        extractor_used="pdfplumber",
                    )
                pdf = pdfplumber.open(str(path))

            total_pages = len(pdf.pages)
            if total_pages == 0:
                return ExtractorResult(
                    status=PDFExtractionStatus.EMPTY_PDF,
                    pages=[],
                    full_raw_text="",
                    full_normalized_text="",
                    error_message="pdfplumber found 0 pages.",
                    extractor_used="pdfplumber",
                )

            pages_to_read = min(total_pages, self.max_pages)
            extracted_pages: list[ExtractedPage] = []
            raw_parts: list[str] = []
            normalized_parts: list[str] = []
            current_offset = 0

            for page_idx in range(pages_to_read):
                page = pdf.pages[page_idx]
                raw_page_text = page.extract_text() or ""
                norm_page_text = self.normalizer.normalize(raw_page_text)

                char_cnt = len(norm_page_text)
                word_cnt = len(norm_page_text.split())
                end_offset = current_offset + len(norm_page_text)

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_idx + 1,
                        raw_text=raw_page_text,
                        normalized_text=norm_page_text,
                        char_count=char_cnt,
                        word_count=word_cnt,
                        has_images=len(page.images) > 0,
                        start_char_offset=current_offset,
                        end_char_offset=end_offset,
                    )
                )

                raw_parts.append(raw_page_text)
                normalized_parts.append(norm_page_text)
                current_offset = end_offset + 2

            full_raw = "\n\n".join(raw_parts).strip()
            full_normalized = "\n\n".join(normalized_parts).strip()
            total_chars = len(full_normalized)

            if total_chars < self.min_doc_chars:
                return ExtractorResult(
                    status=PDFExtractionStatus.INSUFFICIENT_TEXT,
                    pages=extracted_pages,
                    full_raw_text=full_raw,
                    full_normalized_text=full_normalized,
                    error_message=f"pdfplumber extracted only {total_chars} characters.",
                    extractor_used="pdfplumber",
                )

            return ExtractorResult(
                status=PDFExtractionStatus.SUCCESS,
                pages=extracted_pages,
                full_raw_text=full_raw,
                full_normalized_text=full_normalized,
                extractor_used="pdfplumber",
            )

        except Exception as exc:
            return ExtractorResult(
                status=PDFExtractionStatus.MALFORMED_PDF,
                pages=[],
                full_raw_text="",
                full_normalized_text="",
                error_message=f"pdfplumber error: {exc}",
                extractor_used="pdfplumber",
            )
        finally:
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass
            if file_obj is not None:
                try:
                    file_obj.close()
                except Exception:
                    pass

    async def extract_async(self, source: Path | bytes | str) -> ExtractorResult:
        """Asynchronously extract pages by offloading CPU execution to worker thread."""
        return await asyncio.to_thread(self.extract_sync, source)
