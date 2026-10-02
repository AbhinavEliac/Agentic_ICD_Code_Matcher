"""Final deterministic JSON output models and API request/response schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from medical_coding.schemas.enums import (
    Acuity,
    Certainty,
    DiagnosisRole,
    ExecutionStatus,
)
from medical_coding.schemas.validation import AbstentionRecord


class CodedDiagnosisResponse(BaseModel):
    """External deterministic representation of an evidence-backed ICD code decision."""

    code: str = Field(description="Authoritative ICD-10-CM code verified against catalog.")
    description: str = Field(description="Official ICD-10-CM description.")
    role: DiagnosisRole = Field(description="PRIMARY or SECONDARY.")
    acuity: Acuity = Field(description="Acuity status (e.g. ACUTE, CHRONIC).")
    certainty: Certainty = Field(description="Certainty status (e.g. CONFIRMED, SUSPECTED).")
    evidence_quote: str = Field(
        description="Exact quote from clinical record providing documentary evidence.",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Aggregate confidence score for the coding decision.",
    )
    is_terminal_billable: bool = Field(
        default=True,
        description="Confirms code is at terminal specificity required for submission.",
    )


class CodingResult(BaseModel):
    """Deterministic final output conforming to clinical coding architecture principles."""

    document_id: str = Field(description="Encounter or document identifier.")
    status: ExecutionStatus = Field(description="Overall execution status.")
    primary_diagnosis: CodedDiagnosisResponse | None = Field(
        default=None,
        description="Maximum ONE primary diagnosis; None if abstained or undetermined.",
    )
    secondary_diagnoses: list[CodedDiagnosisResponse] = Field(
        default_factory=list,
        description="Zero or more clinically justified secondary diagnoses.",
    )
    abstentions: list[AbstentionRecord] = Field(
        default_factory=list,
        description="Audit records for all conditions where coding was abstained.",
    )
    processing_time_ms: float = Field(
        ge=0.0,
        description="Total end-to-end execution latency in milliseconds.",
    )
    models_used: dict[str, str] = Field(
        default_factory=dict,
        description="Audit map of local models utilized (LLM, embeddings).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional encounter or audit metadata.",
    )

    @model_validator(mode="after")
    def verify_primary_constraint(self) -> "CodingResult":
        """Enforce strict invariant: primary diagnosis must have PRIMARY role if present."""
        if self.primary_diagnosis and self.primary_diagnosis.role != DiagnosisRole.PRIMARY:
            raise ValueError("Primary diagnosis must have role PRIMARY")
        for sec in self.secondary_diagnoses:
            if sec.role == DiagnosisRole.PRIMARY:
                raise ValueError("Secondary diagnoses cannot have role PRIMARY")
        return self


class TextCodingRequest(BaseModel):
    """Input payload for clinical text coding endpoint."""

    document_id: str | None = Field(
        default=None,
        description="Optional encounter identifier; generated if omitted.",
    )
    text: str = Field(
        min_length=10,
        description="Discharge summary or clinical encounter text to analyze.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional encounter metadata.",
    )


class BatchJobStatus(BaseModel):
    """Status record for an asynchronous batch processing job."""

    job_id: str = Field(description="Unique batch job identifier.")
    total_documents: int = Field(ge=0, description="Total count of documents in the batch.")
    completed_documents: int = Field(ge=0, description="Number of documents processed so far.")
    failed_documents: int = Field(ge=0, description="Number of documents that encountered errors.")
    status: Literal["pending", "processing", "completed", "failed"] = Field(
        default="pending",
        description="Lifecycle status of batch processing.",
    )
    results: list[CodingResult] = Field(
        default_factory=list,
        description="Collection of individual document coding results.",
    )
