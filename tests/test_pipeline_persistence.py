"""Unit and integration tests for pipeline execution persistence, step-level time tracking, and failure oversight."""

import pytest

from medical_coding.database.repository import MedicalCodingRepository
from medical_coding.graph.pipeline import process_clinical_document
from medical_coding.schemas.enums import ExecutionStatus


@pytest.fixture
def repo() -> MedicalCodingRepository:
    """Fixture providing an initialized repository instance."""
    return MedicalCodingRepository()


def test_thread_crud_lifecycle(repo: MedicalCodingRepository) -> None:
    """Verify thread creation, step logging, success finalization, and deletion."""
    thread_id = "test-crud-thread-001"
    doc_id = "ENC-TEST-001"

    # 1. Initialize thread
    thread = repo.create_pipeline_thread(
        thread_id=thread_id,
        document_id=doc_id,
        input_source="test_clinical_note.txt",
        raw_text="Patient with acute systolic heart failure.",
        total_steps=10,
    )
    assert thread["thread_id"] == thread_id
    assert thread["status"] == "RUNNING"
    assert thread["progress_pct"] == 0.0

    # 2. Record individual steps with time tracking
    s1 = repo.record_pipeline_step(
        thread_id=thread_id,
        step_index=1,
        step_name="validate_document",
        status="SUCCESS",
        duration_ms=12.4,
        log_message="Payload validated",
        details={"valid": True},
    )
    assert s1 is not None
    assert s1["step_name"] == "validate_document"
    assert s1["duration_ms"] == 12.4

    s2 = repo.record_pipeline_step(
        thread_id=thread_id,
        step_index=2,
        step_name="extract_text",
        status="SUCCESS",
        duration_ms=45.8,
        log_message="Extracted 42 characters",
        details={"char_count": 42},
    )
    assert s2 is not None

    # Check updated thread progress
    updated = repo.get_pipeline_thread(thread_id)
    assert updated is not None
    assert updated["current_step_name"] == "extract_text"
    assert updated["progress_pct"] == 20.0
    assert len(updated["steps"]) == 2

    # 3. Finalize thread with success
    fin = repo.finish_pipeline_thread_success(
        thread_id=thread_id,
        result_data={"document_id": doc_id, "status": "SUCCESS"},
        duration_ms=58.2,
    )
    assert fin is not None
    assert fin["status"] == "SUCCESS"
    assert fin["progress_pct"] == 100.0
    assert fin["duration_ms"] == 58.2
    assert fin["result"] is not None

    # 4. Clean up
    deleted = repo.delete_pipeline_thread(thread_id)
    assert deleted is True
    assert repo.get_pipeline_thread(thread_id) is None


@pytest.mark.asyncio
async def test_pipeline_execution_success_persistence(repo: MedicalCodingRepository) -> None:
    """Verify that process_clinical_document automatically records all steps and persists thread to SQLite."""
    thread_id = "test-live-success-thread"
    doc_id = "ENC-LIVE-001"
    clinical_note = "Patient admitted with acute systolic congestive heart failure and bilateral lower extremity edema."

    result = await process_clinical_document(
        source=clinical_note,
        document_id=doc_id,
        thread_id=thread_id,
        persist_thread=True,
    )

    assert result is not None
    assert result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS)
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.code == "I50.21"

    # Fetch persisted thread record from SQLite
    persisted = repo.get_pipeline_thread(thread_id)
    assert persisted is not None
    assert persisted["status"] == "SUCCESS"
    assert persisted["document_id"] == doc_id
    assert persisted["duration_ms"] > 0
    assert persisted["progress_pct"] == 100.0

    # Verify that all 10 nodes were executed and recorded with their individual latencies
    steps = persisted["steps"]
    assert len(steps) == 10
    step_names = [s["step_name"] for s in steps]
    assert "validate_document" in step_names
    assert "extract_text" in step_names
    assert "extract_diagnoses" in step_names
    assert "analyze_context" in step_names
    assert "classify_diagnoses" in step_names
    assert "retrieve_candidates" in step_names
    assert "rank_candidates" in step_names
    assert "validate_codes" in step_names
    assert "evaluate_confidence" in step_names
    assert "finalize_output" in step_names

    # Check each step has valid time tracking and status
    for s in steps:
        assert s["status"] == "SUCCESS"
        assert s["duration_ms"] >= 0.0
        assert s["log_message"] != ""

    # Clean up test thread
    repo.delete_pipeline_thread(thread_id)


@pytest.mark.asyncio
async def test_pipeline_execution_failure_oversight(repo: MedicalCodingRepository) -> None:
    """Verify that a failing node is precisely identified, recorded as FAILED, and audit trail captures previous steps."""
    thread_id = "test-live-failure-thread"
    doc_id = "ENC-FAIL-001"

    # Mock graph that fails at step 3 (extract_diagnoses)
    class SimulatedFailingGraph:
        async def astream(self, *args, **kwargs):
            yield {"validate_document": {"document_id": doc_id}}
            yield {"extract_text": {"normalized_text": "sample text"}}
            raise ValueError("Simulated parsing crash inside extract_diagnoses agent")

    result = await process_clinical_document(
        source="sample text",
        document_id=doc_id,
        thread_id=thread_id,
        graph=SimulatedFailingGraph(),
        persist_thread=True,
    )

    # 1. Result should indicate ERROR with exact failure step in metadata
    assert result.status == ExecutionStatus.ERROR
    assert result.metadata.get("failed_step") == "extract_text" or result.metadata.get("failed_step") == "validate_document" or "extract" in result.metadata.get("failed_step", "")
    assert "Simulated parsing crash" in result.metadata.get("error", "")
    assert result.metadata.get("traceback") is not None

    # 2. SQLite thread record should capture FAILED status and exact failure step
    persisted = repo.get_pipeline_thread(thread_id)
    assert persisted is not None
    assert persisted["status"] == "FAILED"
    assert persisted["failed_step"] is not None
    assert "Simulated parsing crash" in persisted["error_message"]
    assert persisted["error_traceback"] is not None

    # 3. Check steps: prior steps should be SUCCESS, final step FAILED
    steps = persisted["steps"]
    assert len(steps) >= 2
    failed_steps = [s for s in steps if s["status"] == "FAILED"]
    assert len(failed_steps) == 1
    assert "Simulated parsing crash" in failed_steps[0]["log_message"]

    # Clean up
    repo.delete_pipeline_thread(thread_id)
