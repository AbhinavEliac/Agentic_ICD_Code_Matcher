"""Unit tests for LangGraph state management and reducers."""

from medical_coding.graph.state import create_initial_state
from medical_coding.schemas.enums import PipelineStage
from medical_coding.schemas.state import (
    PipelineExecutionSnapshot,
    append_items,
    merge_dicts,
)


def test_initial_state_generation() -> None:
    state = create_initial_state(document_id="doc-test-1", raw_text="Sample clinical note.")
    assert state["document_id"] == "doc-test-1"
    assert state["raw_text"] == "Sample clinical note."
    assert state["current_stage"] == PipelineStage.INGESTION
    assert state["is_aborted"] is False
    assert state["extracted_diagnoses"] == []
    assert state["candidate_pool"] == {}


def test_append_items_reducer() -> None:
    existing = ["item1", "item2"]
    new_items = ["item3"]
    result = append_items(existing, new_items)
    assert result == ["item1", "item2", "item3"]

    # None / empty checks
    assert append_items([], ["a"]) == ["a"]
    assert append_items(["b"], []) == ["b"]


def test_merge_dicts_reducer() -> None:
    d1 = {"k1": "v1"}
    d2 = {"k2": "v2", "k1": "updated"}
    merged = merge_dicts(d1, d2)
    assert merged == {"k1": "updated", "k2": "v2"}


def test_state_snapshot() -> None:
    snap = PipelineExecutionSnapshot(
        document_id="doc-123",
        current_stage=PipelineStage.FINALIZATION,
        extracted_count=3,
        classified_count=2,
        validated_count=2,
        abstention_count=1,
        has_primary=True,
        is_terminal=True,
    )
    assert snap.extracted_count == 3
    assert snap.has_primary is True
