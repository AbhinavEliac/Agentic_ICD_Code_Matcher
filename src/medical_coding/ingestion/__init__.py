"""Multimodal clinical document ingestion and format auto-detection package."""

from medical_coding.ingestion.agent import ClinicalDocumentIngestionAgent
from medical_coding.ingestion.detector import DocumentFormat, DocumentFormatDetector
from medical_coding.ingestion.parsers import (
    BaseDocumentParser,
    ImageOCRDocumentParser,
    IngestionResult,
    ManualTextParser,
    PDFDocumentParser,
    TextDocumentParser,
)

__all__ = [
    "BaseDocumentParser",
    "ClinicalDocumentIngestionAgent",
    "DocumentFormat",
    "DocumentFormatDetector",
    "ImageOCRDocumentParser",
    "IngestionResult",
    "ManualTextParser",
    "PDFDocumentParser",
    "TextDocumentParser",
]
