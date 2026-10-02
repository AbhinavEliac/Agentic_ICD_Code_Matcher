"""Unit tests for FastAPI endpoints."""

from fastapi.testclient import TestClient


def test_health_check_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["offline_mode"] is True
    assert "llm_model" in data


def test_code_text_endpoint(test_client: TestClient) -> None:
    payload = {
        "document_id": "test-doc-001",
        "text": "Patient was admitted with acute chest pain and diagnosed with acute systolic heart failure.",
        "metadata": {"unit": "ICU"},
    }
    response = test_client.post("/api/v1/code/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "test-doc-001"
    assert "status" in data
    assert "abstentions" in data


def test_code_pdf_rejects_non_pdf(test_client: TestClient) -> None:
    response = test_client.post(
        "/api/v1/code/pdf",
        files={"file": ("notes.txt", b"plain text data", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF document" in response.json()["detail"]


def test_code_pdf_success_endpoint(test_client: TestClient) -> None:
    import io

    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (50, 72),
        "DISCHARGE SUMMARY\nPatient admitted with acute systolic heart failure.",
        fontsize=10,
    )
    pdf_bytes = doc.write()
    doc.close()

    response = test_client.post(
        "/api/v1/code/pdf",
        files={"file": ("discharge_summary.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["status"] in ("SUCCESS", "PARTIAL_SUCCESS")
    assert data["primary_diagnosis"] is not None
    assert data["primary_diagnosis"]["code"] == "I50.21"
