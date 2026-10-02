"""FastAPI routes for clinical document coding and offline status."""

import shutil
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from medical_coding.api.dependencies import get_batch_processor, get_pipeline
from medical_coding.config.settings import Settings, get_settings
from medical_coding.models.factory import ModelFactory
from medical_coding.orchestration.pipeline import MedicalCodingPipeline
from medical_coding.pdf.processor import BatchPDFProcessor
from medical_coding.schemas.response import (
    BatchJobStatus,
    CodingResult,
    TextCodingRequest,
)

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Offline health check and local configuration diagnostics."""
    llm_health = ModelFactory.check_llm_health(settings)
    return {
        "status": "healthy" if llm_health.status != "unhealthy" else "unhealthy",
        "offline_mode": settings.offline_mode,
        "environment": settings.environment,
        "llm_model": str(settings.get_resolved_model_path()),
        "llm_health": {
            "status": llm_health.status,
            "file_exists": llm_health.file_exists,
            "file_size_bytes": llm_health.file_size_bytes,
            "is_loaded": llm_health.is_loaded,
            "load_count": llm_health.load_count,
            "inference_count": llm_health.inference_count,
            "max_llm_concurrency": llm_health.max_llm_concurrency,
            "details": llm_health.details,
        },
        "embedding_model": str(settings.embedding_model_path),
        "icd_dataset": str(settings.icd_dataset_path),
        "max_batch_concurrency": settings.max_batch_concurrency,
        "max_document_concurrency": settings.max_document_concurrency,
    }


@router.post(
    "/api/v1/code/text",
    response_model=CodingResult,
    status_code=status.HTTP_200_OK,
    tags=["Coding"],
)
async def code_clinical_text(
    request: TextCodingRequest,
    pipeline: MedicalCodingPipeline = Depends(get_pipeline),
) -> CodingResult:
    """Analyze discharge summary text and return evidence-backed ICD-10-CM decisions."""
    doc_id = request.document_id or str(uuid4())
    return await pipeline.run_document(
        document_id=doc_id,
        text=request.text,
        metadata=request.metadata,
    )


@router.post(
    "/api/v1/code/pdf",
    response_model=CodingResult,
    status_code=status.HTTP_200_OK,
    tags=["Coding"],
)
async def code_pdf_document(
    file: UploadFile,
    pipeline: MedicalCodingPipeline = Depends(get_pipeline),
) -> CodingResult:
    """Extract and code an uploaded PDF clinical discharge summary."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a valid PDF document (.pdf extension required).",
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    try:
        doc_id = str(uuid4())
        return await pipeline.run_document(
            document_id=doc_id,
            pdf_path=tmp_path,
            metadata={"filename": file.filename},
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.post(
    "/api/v1/code/batch-pdf",
    response_model=BatchJobStatus,
    status_code=status.HTTP_200_OK,
    tags=["Coding"],
)
async def code_batch_pdfs(
    files: list[UploadFile],
    pipeline: MedicalCodingPipeline = Depends(get_pipeline),
    batch_processor: BatchPDFProcessor = Depends(get_batch_processor),
) -> BatchJobStatus:
    """Process an asynchronous batch of clinical PDFs (supports >= 10 files concurrently)."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one PDF file must be provided.",
        )

    temp_paths: list[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="batch_pdf_"))

    try:
        for idx, file in enumerate(files):
            stem = Path(file.filename or f"doc_{idx}").stem
            dest = temp_dir / f"{stem}_{idx}.pdf"
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_paths.append(dest)

        return await batch_processor.process_batch(
            pdf_paths=temp_paths,
            pipeline_executor=pipeline.run_document,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
