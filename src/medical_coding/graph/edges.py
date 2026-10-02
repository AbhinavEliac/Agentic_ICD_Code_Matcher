"""Conditional routing functions for LangGraph state transitions."""

from typing import Literal

from medical_coding.schemas.enums import DiagnosisRole
from medical_coding.schemas.state import PipelineGraphState


def route_after_document_validation(
    state: PipelineGraphState,
) -> Literal["extract_text", "evaluate_confidence"]:
    """Route after document validation: proceed to extraction if input is valid, else abort."""
    if state.get("is_aborted", False):
        return "evaluate_confidence"
    return "extract_text"


def route_after_text_extraction(
    state: PipelineGraphState,
) -> Literal["extract_diagnoses", "evaluate_confidence"]:
    """Route after text extraction: proceed if text is available, else abort."""
    if state.get("is_aborted", False):
        return "evaluate_confidence"
    if not state.get("raw_text", "").strip():
        return "evaluate_confidence"
    return "extract_diagnoses"


def route_after_extraction(
    state: PipelineGraphState,
) -> Literal["analyze_context", "evaluate_confidence"]:
    """Route after extraction: proceed if diagnoses found, otherwise evaluate abstention."""
    if state.get("is_aborted", False):
        return "evaluate_confidence"

    diagnoses = state.get("extracted_diagnoses", []) or state.get("extracted_conditions", [])
    if not diagnoses:
        return "evaluate_confidence"

    return "analyze_context"


def route_after_classification(
    state: PipelineGraphState,
) -> Literal["retrieve_candidates", "evaluate_confidence"]:
    """Route after classification: proceed to candidate retrieval only if billable conditions exist."""
    if state.get("is_aborted", False):
        return "evaluate_confidence"

    classified = state.get("classified_diagnoses", [])
    has_billable = any(
        c.is_billable_candidate and c.role in (DiagnosisRole.PRIMARY, DiagnosisRole.SECONDARY)
        for c in classified
    )

    if not has_billable:
        return "evaluate_confidence"

    return "retrieve_candidates"


def route_after_retrieval(
    state: PipelineGraphState,
) -> Literal["rank_candidates", "validate_codes"]:
    """Route after retrieval: rank candidates if any retrieved, else proceed directly to validation."""
    pool = state.get("candidate_pool", {})
    if not pool or all(len(candidates) == 0 for candidates in pool.values()):
        return "validate_codes"

    return "rank_candidates"
