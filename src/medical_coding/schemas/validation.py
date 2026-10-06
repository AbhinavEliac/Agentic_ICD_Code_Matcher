"""Schemas for deterministic validation results, rule checks, and abstention auditing."""

from pydantic import BaseModel, Field

from medical_coding.schemas.clinical import EvidenceSnippet
from medical_coding.schemas.enums import AbstentionReason, DiagnosisRole, PipelineStage


class ValidationCheck(BaseModel):
    """Result of a single deterministic rule evaluation against the local ICD dataset."""

    rule_name: str = Field(
        description="Name of rule (e.g. 'CatalogExistence', 'TerminalSpecificity', 'Excludes1Constraint').",
    )
    passed: bool = Field(description="True if the code satisfies the deterministic rule.")
    details: str = Field(description="Audit message detailing check result.")


class ValidatedDiagnosis(BaseModel):
    """An ICD code that has passed deterministic validation against the local dataset."""

    diagnosis_id: str = Field(description="Entity identifier linking to extracted diagnosis.")
    raw_term: str = Field(description="Clinical term extracted from document.")
    code: str = Field(description="Validated authoritative ICD-10-CM code.")
    description: str = Field(description="Authoritative ICD-10-CM description.")
    role: DiagnosisRole = Field(description="Final role: PRIMARY or SECONDARY.")
    evidence: EvidenceSnippet = Field(description="Supporting clinical documentation evidence.")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Consolidated confidence score across extraction and ranking.",
    )
    checks: list[ValidationCheck] = Field(
        default_factory=list,
        description="Complete record of deterministic validation checks executed.",
    )
    icd10cm: str | None = Field(
        default=None,
        description="Matched ICD-10-CM code if matched, else None.",
    )
    icdo: str | None = Field(
        default=None,
        description="Matched ICD-O oncology morphology code if matched, else None.",
    )
    cpt: str | None = Field(
        default=None,
        description="Matched CPT procedural code if matched, else None.",
    )
    database_code: str | None = Field(
        default=None,
        description="Exact authoritative code from local database.",
    )
    database_description: str | None = Field(
        default=None,
        description="Exact authoritative description from local database.",
    )
    matching_status: str = Field(
        default="MATCHED",
        description="Status of database matching: MATCHED or NO_DATABASE_MATCH.",
    )
    source_section: str = Field(
        default="",
        description="Section of the document where evidence was found.",
    )
    source_span: tuple[int, int] | None = Field(
        default=None,
        description="Character offsets of evidence quote.",
    )


class AbstentionRecord(BaseModel):
    """Structured record explaining why an entity or document was abstained from coding."""

    diagnosis_id: str | None = Field(
        default=None,
        description="Diagnosis entity ID if abstention is entity-level; None if document-level.",
    )
    raw_term: str | None = Field(
        default=None,
        description="Clinical term that could not be reliably coded.",
    )
    reason: AbstentionReason = Field(
        description="Standardized machine-readable abstention category.",
    )
    detail: str = Field(
        min_length=1,
        description="Human-readable explanation of why coding was abstained.",
    )
    stage: PipelineStage = Field(
        description="Pipeline stage at which the abstention decision was triggered.",
    )
