"""End-to-end integration test verifying all four input modalities with auto-detection, parsing, coding pipeline, and DB persistence."""

import asyncio
from uuid import uuid4

import pytest

from medical_coding.database.connection import init_db
from medical_coding.database.repository import MedicalCodingRepository
from medical_coding.ingestion.agent import ClinicalDocumentIngestionAgent
from medical_coding.ingestion.detector import DocumentFormat
from medical_coding.orchestration.pipeline import MedicalCodingPipeline
from medical_coding.schemas.enums import ExecutionStatus
from medical_coding.ui.sample_data import (
    CLINICAL_CASES,
    generate_sample_image,
    generate_sample_pdf,
    generate_sample_txt,
)


@pytest.fixture(scope="module")
def pipeline():
    return MedicalCodingPipeline()


@pytest.fixture(scope="module")
def repo():
    init_db()
    return MedicalCodingRepository()


@pytest.fixture(scope="module")
def ingestion_agent():
    return ClinicalDocumentIngestionAgent(max_pdf_pages=10)


def test_modality_1_pdf_end_to_end(pipeline, repo, ingestion_agent):
    case = CLINICAL_CASES["Case 1: Decompensated Heart Failure & Comorbidities"]
    pdf_bytes = generate_sample_pdf(case["title"], case["text"])
    doc_id = f"TEST-PDF-{uuid4().hex[:6].upper()}"

    # 1. Ingestion auto-detection & parsing
    ingest_res = ingestion_agent.ingest(source=pdf_bytes, filename="discharge_note.pdf")
    assert ingest_res.format == DocumentFormat.PDF
    assert ingest_res.status == "SUCCESS"
    assert "heart failure" in ingest_res.normalized_text.lower()
    assert ingest_res.page_count >= 1

    # 2. Pipeline execution
    coding_res = asyncio.run(
        pipeline.run_document(
            document_id=doc_id,
            text=ingest_res.normalized_text,
            file_bytes=pdf_bytes,
            metadata={"filename": "discharge_note.pdf", "source_type": "pdf"},
        )
    )
    assert coding_res.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS)
    all_diags = ([coding_res.primary_diagnosis] if coding_res.primary_diagnosis else []) + coding_res.secondary_diagnoses
    all_codes = [d.code for d in all_diags]
    assert "I50.21" in all_codes

    # 3. DB Persistence
    saved = repo.save_coding_result(
        result=coding_res,
        raw_text=ingest_res.normalized_text,
        filename="discharge_note.pdf",
        source_type="pdf",
        page_count=ingest_res.page_count,
        file_size_bytes=len(pdf_bytes),
    )
    assert saved["id"] > 0
    assert saved["document_id"] == doc_id
    assert saved["source_type"] == "pdf"


def test_modality_2_plain_text_end_to_end(pipeline, repo, ingestion_agent):
    case = CLINICAL_CASES["Case 4: Severe Sepsis & Acute Kidney Injury"]
    txt_bytes = generate_sample_txt(case["title"], case["text"])
    doc_id = f"TEST-TXT-{uuid4().hex[:6].upper()}"

    # 1. Ingestion auto-detection & parsing
    ingest_res = ingestion_agent.ingest(source=txt_bytes, filename="sepsis_summary.txt")
    assert ingest_res.format == DocumentFormat.TXT
    assert ingest_res.status == "SUCCESS"
    assert "sepsis" in ingest_res.normalized_text.lower()

    # 2. Pipeline execution
    coding_res = asyncio.run(
        pipeline.run_document(
            document_id=doc_id,
            text=ingest_res.normalized_text,
            file_bytes=txt_bytes,
            metadata={"filename": "sepsis_summary.txt", "source_type": "txt"},
        )
    )
    assert coding_res.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS, ExecutionStatus.ABSTAINED)
    all_diags = ([coding_res.primary_diagnosis] if coding_res.primary_diagnosis else []) + coding_res.secondary_diagnoses
    assert len(all_diags) > 0 or len(coding_res.abstentions) > 0

    # 3. DB Persistence
    saved = repo.save_coding_result(
        result=coding_res,
        raw_text=ingest_res.normalized_text,
        filename="sepsis_summary.txt",
        source_type="txt",
        page_count=1,
        file_size_bytes=len(txt_bytes),
    )
    assert saved["id"] > 0
    assert saved["document_id"] == doc_id
    assert saved["source_type"] == "txt"


def test_modality_3_image_screenshot_ocr_end_to_end(pipeline, repo, ingestion_agent):
    case = CLINICAL_CASES["Case 2: Acute COPD Exacerbation with Pneumonia"]
    img_bytes = generate_sample_image(case["title"], case["text"])
    doc_id = f"TEST-IMG-{uuid4().hex[:6].upper()}"

    # 1. Ingestion auto-detection & RapidOCR parsing
    ingest_res = ingestion_agent.ingest(source=img_bytes, filename="copd_screenshot.png")
    assert ingest_res.format == DocumentFormat.IMAGE
    assert ingest_res.status == "SUCCESS"
    assert ingest_res.is_ocr is True
    assert ingest_res.ocr_confidence is not None
    assert ingest_res.ocr_confidence > 0.60
    assert "copd" in ingest_res.normalized_text.lower() or "chronic obstructive" in ingest_res.normalized_text.lower()

    # 2. Pipeline execution
    coding_res = asyncio.run(
        pipeline.run_document(
            document_id=doc_id,
            text=ingest_res.normalized_text,
            file_bytes=img_bytes,
            metadata={"filename": "copd_screenshot.png", "source_type": "image"},
        )
    )
    assert coding_res is not None

    # 3. DB Persistence
    saved = repo.save_coding_result(
        result=coding_res,
        raw_text=ingest_res.normalized_text,
        filename="copd_screenshot.png",
        source_type="image",
        page_count=1,
        file_size_bytes=len(img_bytes),
    )
    assert saved["id"] > 0
    assert saved["document_id"] == doc_id
    assert saved["source_type"] == "image"


def test_modality_4_manual_text_end_to_end(pipeline, repo, ingestion_agent):
    clinical_notes = """HOSPITAL DISCHARGE SUMMARY
PATIENT ENCOUNTER: ENC-MANUAL-001

CHIEF COMPLAINT:
Severe shortness of breath on exertion and progressive 3-pillow orthopnea.

HISTORY OF PRESENT ILLNESS:
The patient is a 72-year-old male who presented with acute exertional dyspnea and bilateral lower extremity edema. Echocardiogram demonstrated severe systolic dysfunction with left ventricular ejection fraction 25%.

DISCHARGE DIAGNOSES:
1. Acute systolic heart failure (decompensated congestive heart failure with systolic dysfunction).
2. Essential primary hypertension.
"""
    doc_id = f"TEST-MAN-{uuid4().hex[:6].upper()}"

    # 1. Ingestion auto-detection & parsing
    ingest_res = ingestion_agent.ingest(source=clinical_notes, filename="manual_entry.txt")
    assert ingest_res.format == DocumentFormat.MANUAL_TEXT
    assert ingest_res.status == "SUCCESS"
    assert ingest_res.is_ocr is False
    assert "systolic heart failure" in ingest_res.normalized_text.lower()

    # 2. Pipeline execution
    coding_res = asyncio.run(
        pipeline.run_document(
            document_id=doc_id,
            text=ingest_res.normalized_text,
            metadata={"filename": "manual_entry.txt", "source_type": "manual_text"},
        )
    )
    assert coding_res.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS)
    all_diags = ([coding_res.primary_diagnosis] if coding_res.primary_diagnosis else []) + coding_res.secondary_diagnoses
    all_codes = [d.code for d in all_diags]
    assert "I50.21" in all_codes

    # 3. DB Persistence
    saved = repo.save_coding_result(
        result=coding_res,
        raw_text=ingest_res.normalized_text,
        filename="manual_entry.txt",
        source_type="manual_text",
        page_count=1,
        file_size_bytes=len(clinical_notes.encode("utf-8")),
    )
    assert saved["id"] > 0
    assert saved["document_id"] == doc_id
    assert saved["source_type"] == "manual_text"
