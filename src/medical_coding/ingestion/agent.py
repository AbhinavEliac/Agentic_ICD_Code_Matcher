"""Unified Clinical Ingestion Agent coordinating auto-detection and multi-format parsing."""

from pathlib import Path

from medical_coding.ingestion.detector import DocumentFormat, DocumentFormatDetector
from medical_coding.ingestion.parsers import (
    ImageOCRDocumentParser,
    IngestionResult,
    ManualTextParser,
    PDFDocumentParser,
    TextDocumentParser,
)
from medical_coding.pdf.normalizer import DocumentNormalizer
from medical_coding.pdf.section_detector import SectionDetector
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class ClinicalDocumentIngestionAgent:
    """Intelligent agent that automatically detects document modality and routes to the appropriate parser.

    Supported Formats:
        - PDF (.pdf): Vector extraction with PyMuPDF and pdfplumber, with automatic OCR fallback if scanned.
        - Plain Text (.txt, .md, .log, .csv): Direct decoding and clinical normalization.
        - Images / Screenshots (.png, .jpg, .jpeg, .tiff, .bmp, .webp): Neural OCR text extraction.
        - Manual Text Entry: String normalization and boundary identification.
    """

    def __init__(
        self,
        max_pdf_pages: int = 50,
        normalizer: DocumentNormalizer | None = None,
    ) -> None:
        self.normalizer = normalizer or DocumentNormalizer()
        self.section_detector = SectionDetector()
        self.detector = DocumentFormatDetector()

        # Parsers
        self.text_parser = TextDocumentParser(normalizer=self.normalizer)
        self.image_parser = ImageOCRDocumentParser(normalizer=self.normalizer)
        self.pdf_parser = PDFDocumentParser(
            max_pages=max_pdf_pages,
            normalizer=self.normalizer,
            image_ocr_parser=self.image_parser,
        )
        self.manual_parser = ManualTextParser(normalizer=self.normalizer)

    def ingest(
        self,
        source: bytes | str | Path,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> IngestionResult:
        """Auto-detect format and extract clinical discharge summary text.

        Args:
            source: Raw file bytes, file Path, or string content.
            filename: Optional source filename (helps in extension heuristic).
            mime_type: Optional explicit MIME type.

        Returns:
            Standardized IngestionResult containing normalized text, format, and metadata.
        """
        fmt, detected_mime = self.detector.detect_format(
            source=source,
            filename=filename,
            mime_type=mime_type,
        )

        logger.info(
            "Ingestion Agent detected format=%s (mime=%s) for filename=%s",
            fmt.value,
            detected_mime,
            filename or "unnamed",
        )

        result: IngestionResult
        if fmt == DocumentFormat.PDF:
            result = self.pdf_parser.parse(source=source, filename=filename)
        elif fmt == DocumentFormat.IMAGE:
            result = self.image_parser.parse(source=source, filename=filename)
        elif fmt == DocumentFormat.TXT:
            result = self.text_parser.parse(source=source, filename=filename)
        elif fmt == DocumentFormat.MANUAL_TEXT:
            result = self.manual_parser.parse(source=source, filename=filename)
        else:
            # Fallback attempt: if binary, try image OCR, then text
            if isinstance(source, bytes):
                result = self.image_parser.parse(source=source, filename=filename)
                if result.status != "SUCCESS":
                    result = self.text_parser.parse(source=source, filename=filename)
            else:
                result = self.manual_parser.parse(source=source, filename=filename)

        # Attach detected MIME type to metadata
        result.metadata["detected_format"] = fmt.value
        result.metadata["detected_mime"] = detected_mime

        # Detect clinical sections if text extraction was successful
        if result.status == "SUCCESS" and result.normalized_text:
            try:
                sections = self.section_detector.detect_sections(full_text=result.normalized_text)
                result.metadata["section_count"] = len(sections)
                result.metadata["detected_sections"] = [s.section_type.value for s in sections]
            except Exception as e:
                logger.debug("Section detection optional step encountered note: %s", e)

        return result
