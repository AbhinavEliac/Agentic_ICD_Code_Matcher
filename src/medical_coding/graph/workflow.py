"""Compilation and construction of the complete LangGraph medical coding workflow."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from medical_coding.graph.edges import (
    route_after_classification,
    route_after_document_validation,
    route_after_extraction,
    route_after_retrieval,
    route_after_text_extraction,
)
from medical_coding.graph.nodes import (
    analyze_context_node,
    classify_diagnoses_node,
    evaluate_confidence_node,
    extract_diagnoses_node,
    extract_text_node,
    finalize_output_node,
    rank_candidates_node,
    retrieve_candidates_node,
    validate_codes_node,
    validate_document_node,
)
from medical_coding.schemas.state import PipelineGraphState


def create_coding_workflow() -> StateGraph:
    """Build the state graph for medical ICD-10-CM coding.

    Target Graph Flow:
        START
          ↓
        Document Validation (validate_document)
          ↓
        Text Extraction (extract_text)
          ↓
        Clinical Extraction Agent (extract_diagnoses)
          ↓
        Context/Evidence Agent (analyze_context)
          ↓
        Primary/Secondary Classification (classify_diagnoses)
          ↓
        ICD Candidate Retrieval (retrieve_candidates)
          ↓
        ICD Candidate Ranking (rank_candidates)
          ↓
        Deterministic Validation (validate_codes)
          ↓
        Confidence / Abstention (evaluate_confidence)
          ↓
        Final JSON (finalize_output)
          ↓
        END
    """
    workflow = StateGraph(PipelineGraphState)

    # 1. Register Nodes in strict sequence
    workflow.add_node("validate_document", validate_document_node)
    workflow.add_node("extract_text", extract_text_node)
    workflow.add_node("extract_diagnoses", extract_diagnoses_node)
    workflow.add_node("analyze_context", analyze_context_node)
    workflow.add_node("classify_diagnoses", classify_diagnoses_node)
    workflow.add_node("retrieve_candidates", retrieve_candidates_node)
    workflow.add_node("rank_candidates", rank_candidates_node)
    workflow.add_node("validate_codes", validate_codes_node)
    workflow.add_node("evaluate_confidence", evaluate_confidence_node)
    workflow.add_node("finalize_output", finalize_output_node)

    # 2. Initial Transition from START to Document Validation
    workflow.add_edge(START, "validate_document")

    # 3. Conditional Transition after Document Validation
    workflow.add_conditional_edges(
        "validate_document",
        route_after_document_validation,
        {
            "extract_text": "extract_text",
            "evaluate_confidence": "evaluate_confidence",
        },
    )

    # 4. Conditional Transition after Text Extraction
    workflow.add_conditional_edges(
        "extract_text",
        route_after_text_extraction,
        {
            "extract_diagnoses": "extract_diagnoses",
            "evaluate_confidence": "evaluate_confidence",
        },
    )

    # 5. Conditional Transition after Clinical Extraction
    workflow.add_conditional_edges(
        "extract_diagnoses",
        route_after_extraction,
        {
            "analyze_context": "analyze_context",
            "evaluate_confidence": "evaluate_confidence",
        },
    )

    # 6. Linear Transition to Classification
    workflow.add_edge("analyze_context", "classify_diagnoses")

    # 7. Conditional Transition after Classification
    workflow.add_conditional_edges(
        "classify_diagnoses",
        route_after_classification,
        {
            "retrieve_candidates": "retrieve_candidates",
            "evaluate_confidence": "evaluate_confidence",
        },
    )

    # 8. Conditional Transition after Retrieval
    workflow.add_conditional_edges(
        "retrieve_candidates",
        route_after_retrieval,
        {
            "rank_candidates": "rank_candidates",
            "validate_codes": "validate_codes",
        },
    )

    # 9. Linear Sequence through Validation & Finalization
    workflow.add_edge("rank_candidates", "validate_codes")
    workflow.add_edge("validate_codes", "evaluate_confidence")
    workflow.add_edge("evaluate_confidence", "finalize_output")
    workflow.add_edge("finalize_output", END)

    return workflow


def get_compiled_graph() -> Any:
    """Compile and return the executable LangGraph app."""
    workflow = create_coding_workflow()
    return workflow.compile()


def export_graph_mermaid() -> str:
    """Export the compiled LangGraph workflow topology as a Mermaid diagram definition."""
    app = get_compiled_graph()
    try:
        return app.get_graph().draw_mermaid()
    except Exception:
        return """graph TD
    START --> validate_document
    validate_document --> extract_text
    validate_document -.-> evaluate_confidence
    extract_text --> extract_diagnoses
    extract_text -.-> evaluate_confidence
    extract_diagnoses --> analyze_context
    extract_diagnoses -.-> evaluate_confidence
    analyze_context --> classify_diagnoses
    classify_diagnoses --> retrieve_candidates
    classify_diagnoses -.-> evaluate_confidence
    retrieve_candidates --> rank_candidates
    retrieve_candidates -.-> validate_codes
    rank_candidates --> validate_codes
    validate_codes --> evaluate_confidence
    evaluate_confidence --> finalize_output
    finalize_output --> END
"""


def export_graph_ascii() -> str:
    """Export the compiled LangGraph workflow topology as an ASCII representation."""
    app = get_compiled_graph()
    try:
        return app.get_graph().draw_ascii()
    except Exception:
        return (
            "START -> validate_document -> extract_text -> extract_diagnoses -> "
            "analyze_context -> classify_diagnoses -> retrieve_candidates -> "
            "rank_candidates -> validate_codes -> evaluate_confidence -> finalize_output -> END"
        )
