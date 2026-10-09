"""Central application state definitions for LangGraph workflows and pipeline orchestration."""

from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from medical_coding.schemas.clinical import (
    ClassifiedDiagnosis,
    ContextAssessment,
    ContextualizedDiagnosis,
    EncounterClassificationResult,
    ExtractedClinicalCondition,
    ExtractedDiagnosis,
)
from medical_coding.schemas.enums import PipelineStage
from medical_coding.schemas.evidence import (
    ClinicalDiagnosisCandidate,
    ClinicalDiagnosisState,
    ICDMappingState,
)
from medical_coding.schemas.icd import ICDCandidate, RankedSelection
from medical_coding.schemas.response import CodingResult
from medical_coding.schemas.validation import AbstentionRecord, ValidatedDiagnosis


def append_items(existing: list[Any], new_items: list[Any]) -> list[Any]:
    """LangGraph state reducer to safely append elements to lists."""
    if not existing:
        return list(new_items)
    if not new_items:
        return list(existing)
    return list(existing) + list(new_items)


def merge_dicts(existing: dict[str, Any], new_dict: dict[str, Any]) -> dict[str, Any]:
    """LangGraph state reducer to shallow-merge dictionary updates."""
    merged = dict(existing) if existing else {}
    if new_dict:
        merged.update(new_dict)
    return merged


class PipelineGraphState(TypedDict, total=False):
    """Central state passed across all LangGraph nodes in the medical coding pipeline.

    Each node inspects this state and returns a partial dictionary updating
    only the relevant fields for that stage.
    """

    # Input Document Attributes
    document_id: str
    raw_text: str
    pdf_path: str | None
    pdf_bytes: bytes | None
    file_path: str | None
    file_bytes: bytes | None
    detected_filetype: str | None
    source_type: str
    metadata: dict[str, Any]

    # Clinical Extraction & Context Analysis (Evidence-First Architecture)
    clinical_diagnosis_state: ClinicalDiagnosisState | None
    diagnosis_candidates: list[ClinicalDiagnosisCandidate]
    extracted_procedures: list[str]
    operative_findings: list[dict[str, Any]]
    icd_mapping_states: list[ICDMappingState]

    # Legacy attributes preserved for backwards compatibility
    extracted_diagnoses: list[ExtractedDiagnosis]
    extracted_conditions: list[ExtractedClinicalCondition]
    context_assessments: list[ContextAssessment]
    contextualized_diagnoses: list[ContextualizedDiagnosis]
    classified_diagnoses: list[ClassifiedDiagnosis]
    classification_result: EncounterClassificationResult

    # Local Retrieval & Ranking (Constrained by ClinicalConcept & MatchSpec)
    clinical_concepts: dict[str, Any]  # Keyed by diagnosis_id -> ClinicalConcept
    match_specs: dict[str, Any]  # Keyed by diagnosis_id -> MatchSpec
    compatibility_results: dict[str, Any]  # Keyed by diagnosis_id -> list[CompatibilityResult]
    candidate_pool: dict[str, list[ICDCandidate]]  # Keyed by diagnosis_id
    ranked_selections: list[RankedSelection]

    # Deterministic Validation & Audit
    validated_diagnoses: list[ValidatedDiagnosis]
    abstentions: Annotated[list[AbstentionRecord], append_items]
    audit_trail: Annotated[list[dict[str, Any]], append_items]

    # Pipeline Control & Errors
    current_stage: PipelineStage
    errors: Annotated[list[str], append_items]
    is_aborted: bool

    # Final Output
    final_result: CodingResult | None


class PipelineExecutionSnapshot(BaseModel):
    """Immutable Pydantic model capturing a serialized snapshot of pipeline state.

    Useful for telemetry, auditing, inspection, and regression replay.
    """

    document_id: str
    current_stage: PipelineStage
    extracted_count: int = Field(ge=0, default=0)
    classified_count: int = Field(ge=0, default=0)
    validated_count: int = Field(ge=0, default=0)
    abstention_count: int = Field(ge=0, default=0)
    error_count: int = Field(ge=0, default=0)
    has_primary: bool = False
    is_terminal: bool = False
