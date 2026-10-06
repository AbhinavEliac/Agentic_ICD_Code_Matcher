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

    raw_term: str | None = Field(
        default=None,
        description="Original clinical term as extracted from documentation.",
    )
    normalized_diagnosis: str | None = Field(
        default=None,
        description="Canonical clinical diagnosis name authorized before coding.",
    )
    description: str = Field(description="Official clinical description matching the code or diagnosis.")
    role: DiagnosisRole = Field(description="PRIMARY or SECONDARY.")
    acuity: Acuity = Field(description="Acuity status (e.g. ACUTE, CHRONIC).")
    certainty: Certainty = Field(description="Certainty status (e.g. CONFIRMED, SUSPECTED).")
    evidence_quote: str = Field(
        description="Exact quote from clinical record providing documentary evidence.",
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of structured evidence references supporting this diagnosis.",
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
    icd10cm: str | None = Field(
        default=None,
        description="Matched ICD-10-CM code if matched, None if not.",
    )
    icdo: str | None = Field(
        default=None,
        description="Matched ICD-O oncology/morphology code if matched, None if not.",
    )
    cpt: str | None = Field(
        default=None,
        description="Matched CPT procedural code if matched, None if not.",
    )
    matching_status: Literal["MATCHED", "NO_DATABASE_MATCH"] = Field(
        default="MATCHED",
        description="Whether this diagnosis matched a database code or abstained as NO_DATABASE_MATCH.",
    )
    database_code: str | None = Field(
        default=None,
        description="Exact authoritative code from Database/ folder.",
    )
    database_description: str | None = Field(
        default=None,
        description="Exact authoritative description from Database/ folder.",
    )
    source_section: str | None = Field(
        default=None,
        description="Clinical document section where evidence was located.",
    )
    source_span: tuple[int, int] | None = Field(
        default=None,
        description="Exact character span [start, end] of clinical evidence in source text.",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_code(cls, data: Any) -> Any:
        """Allow legacy 'code' kwarg to seamlessly map to appropriate system field without being redundant in output."""
        if isinstance(data, dict):
            code_val = data.pop("code", None)
            if code_val and not data.get("icd10cm") and not data.get("icdo") and not data.get("cpt"):
                if str(code_val).startswith("M") and len(str(code_val)) >= 5:
                    data["icdo"] = code_val
                elif str(code_val).isdigit() and len(str(code_val)) in (4, 5):
                    data["cpt"] = code_val
                else:
                    data["icd10cm"] = code_val

            effective_code = code_val or data.get("icd10cm") or data.get("icdo") or data.get("cpt")
            if effective_code and not data.get("database_code"):
                data["database_code"] = str(effective_code)

            if data.get("description") and not data.get("database_description"):
                data["database_description"] = data.get("description")

            if not effective_code:
                data.setdefault("matching_status", "NO_DATABASE_MATCH")
            else:
                data.setdefault("matching_status", "MATCHED")

        return data

    @property
    def code(self) -> str:
        """Dynamic code property resolving primary matched code for backward compatibility."""
        return self.icd10cm or self.icdo or self.cpt or ""


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
    excluded_candidates: list[dict[str, str]] = Field(
        default_factory=list,
        description="Collection of non-diagnostic or rejected terms excluded by clinical gate.",
    )
    validation: dict[str, bool] = Field(
        default_factory=lambda: {
            "evidence_grounded": True,
            "database_grounded": True,
            "unsupported_specificity": False,
            "hallucinated_codes": False,
            "noise_capture": False,
            "duplicate_candidates": False,
            "primary_secondary_validated": True,
        },
        description="Deterministic architectural validation summary flags.",
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
