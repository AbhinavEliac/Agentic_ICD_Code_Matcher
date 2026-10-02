"""Unit tests for LangGraph workflow compilation and execution transitions."""

import pytest

from medical_coding.graph.state import create_initial_state
from medical_coding.graph.workflow import create_coding_workflow, get_compiled_graph
from medical_coding.schemas.enums import ExecutionStatus, PipelineStage


def test_workflow_compilation() -> None:
    """Verify that StateGraph compiles cleanly without dangling edges or schema mismatches."""
    workflow = create_coding_workflow()
    compiled = workflow.compile()
    assert compiled is not None


@pytest.mark.asyncio
async def test_graph_execution_empty_input_aborts_gracefully() -> None:
    """Verify that an empty document input routes correctly and abstains."""
    graph = get_compiled_graph()
    initial_state = create_initial_state(document_id="empty-doc", raw_text="")

    final_state = await graph.ainvoke(initial_state)

    assert final_state["current_stage"] == PipelineStage.FINALIZATION
    assert final_state["is_aborted"] is True
    assert final_state["final_result"] is not None
    assert final_state["final_result"].status == ExecutionStatus.ABSTAINED
    assert len(final_state["abstentions"]) >= 1
