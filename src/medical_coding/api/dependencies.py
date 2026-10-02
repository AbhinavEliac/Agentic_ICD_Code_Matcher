"""FastAPI dependency injection providers."""

from functools import lru_cache

from medical_coding.config.settings import get_settings
from medical_coding.orchestration.pipeline import MedicalCodingPipeline
from medical_coding.pdf.extractor import PDFExtractor
from medical_coding.pdf.processor import BatchPDFProcessor


@lru_cache(maxsize=1)
def get_pipeline() -> MedicalCodingPipeline:
    """Dependency provider for cached pipeline instance."""
    return MedicalCodingPipeline()


@lru_cache(maxsize=1)
def get_batch_processor() -> BatchPDFProcessor:
    """Dependency provider for batch PDF processor."""
    settings = get_settings()
    extractor = PDFExtractor(max_pages=settings.pdf_max_pages)
    return BatchPDFProcessor(
        extractor=extractor,
        max_concurrency=settings.max_batch_concurrency,
    )
