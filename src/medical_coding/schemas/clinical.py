"""Schemas for clinical text spans, evidence grounding, extracted entities, and context."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from medical_coding.schemas.enums import (
    AbstentionReason,
    Acuity,
    Certainty,
    ClinicalEntityType,
    ConditionStatus,
    DiagnosisRole,
    Laterality,
    NegationStatus,
    Temporality,
)


class TextSpan(BaseModel):
    """Character offsets and optional page metadata within the source text."""

    start_char: int = Field(ge=0, description="Start character offset in document.")
    end_char: int = Field(ge=0, description="End character offset in document.")
    text: str = Field(min_length=1, description="Exact substring matched.")
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="1-based page index for multi-page documents/PDFs.",
    )


class EvidenceSnippet(BaseModel):
    """Verbatim clinical evidence supporting a diagnostic claim."""

    quote: str = Field(
        min_length=1,
        description="Verbatim excerpt from the clinical documentation providing ground truth.",
    )
    span: TextSpan | None = Field(
        default=None,
        description="Exact location within document text if resolvable.",
    )
    source_section: str | None = Field(
        default=None,
        description="Clinical section header (e.g. 'Discharge Diagnosis', 'History of Present Illness').",
    )


class ClinicalDocument(BaseModel):
    """Input clinical documentation container."""

    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the patient encounter or document.",
    )
    text: str = Field(min_length=1, description="Raw discharge summary or clinical note text.")
    source_type: str = Field(
        default="text",
        description="Origin format: 'text', 'pdf', 'ehr_export'.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional encounter metadata (e.g., patient age, admission date).",
    )


class EvidenceLocation(BaseModel):
    """Detailed structural evidence locator."""

    section: str = Field(
        description="Clinical section where mention occurred (e.g. 'DISCHARGE_DIAGNOSES').",
    )
    page_number: int = Field(
        default=1,
        ge=1,
        description="1-based page index where the evidence text is located.",
    )
    start_char: int | None = Field(
        default=None,
        ge=0,
        description="Start character offset of the evidence quote in document text.",
    )
    end_char: int | None = Field(
        default=None,
        ge=0,
        description="End character offset of the evidence quote in document text.",
    )


class ExtractedClinicalCondition(BaseModel):
    """Strict structured representation of an extracted clinical condition.

    CRITICAL RULES:
    1. MUST NOT contain ICD codes.
    2. MUST have verbatim evidence_text directly from document text.
    3. Cannot convert uncertain conditions into confirmed diagnoses.
    4. Cannot infer unwritten clinical facts.
    """

    condition_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique condition entity identifier.",
    )
    original_mention: str = Field(
        min_length=1,
        description="Verbatim diagnostic or clinical term extracted from text.",
    )
    normalized_description: str = Field(
        min_length=1,
        description="Canonical normalized clinical condition description.",
    )
    entity_type: ClinicalEntityType = Field(
        default=ClinicalEntityType.DIAGNOSIS,
        description="DIAGNOSIS, SYMPTOM, SIGN, or PROCEDURAL_FINDING.",
    )
    evidence_text: str = Field(
        min_length=1,
        description="Verbatim sentence or phrase from the document providing factual proof.",
    )
    evidence_location: EvidenceLocation = Field(
        description="Section and page/offset location of evidence.",
    )
    status: ConditionStatus = Field(
        default=ConditionStatus.ACTIVE,
        description="ACTIVE, HISTORICAL, RESOLVED, CHRONIC, ACUTE, or UNKNOWN.",
    )
    certainty: Certainty = Field(
        default=Certainty.CONFIRMED,
        description="CONFIRMED, SUSPECTED, POSSIBLE, RULED_OUT, or UNKNOWN.",
    )
    temporality: Temporality = Field(
        default=Temporality.CURRENT,
        description="CURRENT, HISTORICAL, RESOLVED, FAMILY_HISTORY, or UNKNOWN.",
    )
    negation: NegationStatus = Field(
        default=NegationStatus.AFFIRMATIVE,
        description="AFFIRMATIVE, NEGATED, or UNCERTAIN.",
    )
    anatomical_site: str | None = Field(
        default=None,
        description="Specific anatomical location if explicitly documented.",
    )
    laterality: Laterality = Field(
        default=Laterality.UNSPECIFIED,
        description="LEFT, RIGHT, BILATERAL, or UNSPECIFIED.",
    )
    section: str = Field(
        description="Section where mention appeared.",
    )
    treatment_evidence: str | None = Field(
        default=None,
        description="Documented treatment or management evidence if explicitly stated.",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score.",
    )


class ClinicalExtractionResult(BaseModel):
    """Aggregate output of the clinical information extraction agent."""

    document_id: str = Field(description="Encounter or document ID.")
    conditions: list[ExtractedClinicalCondition] = Field(
        default_factory=list,
        description="Collection of evidence-grounded candidate conditions.",
    )
    extraction_notes: list[str] = Field(
        default_factory=list,
        description="Audit notes regarding extraction and parsing.",
    )
    repair_applied: bool = Field(
        default=False,
        description="True if JSON or schema repair heuristics were applied.",
    )
    raw_response: str | None = Field(
        default=None,
        description="Raw output from the local LLM before parsing.",
    )


class ContextAssessment(BaseModel):
    """Evaluation of clinical relevance, current management, and coding eligibility.

    CRITICAL RULES:
    1. Past Medical History (PMH) alone does NOT qualify a condition for coding.
       Must have documented clinical evaluation, monitoring, or treatment during the stay.
    2. Ruled-out conditions cannot be confirmed or coded as active diagnoses.
    3. Uncertain conditions must retain SUSPECTED/POSSIBLE certainty; never convert to CONFIRMED.
    """

    diagnosis: str = Field(
        description="Normalized diagnostic condition term.",
    )
    condition_id: str | None = Field(
        default=None,
        description="Identifier linking to ExtractedClinicalCondition.",
    )
    current_relevance: bool = Field(
        description="True if clinically active and relevant to current hospitalization.",
    )
    coding_candidate: bool = Field(
        description="True if eligible to be coded under inpatient coding guidelines.",
    )
    status: ConditionStatus = Field(
        description="ACTIVE, HISTORICAL, RESOLVED, CHRONIC, ACUTE, UNKNOWN.",
    )
    certainty: Certainty = Field(
        description="CONFIRMED, SUSPECTED, POSSIBLE, RULED_OUT, UNKNOWN.",
    )
    temporality: Temporality = Field(
        default=Temporality.CURRENT,
        description="CURRENT, HISTORICAL, RESOLVED, UNKNOWN.",
    )
    negation: NegationStatus = Field(
        default=NegationStatus.AFFIRMATIVE,
        description="AFFIRMATIVE, NEGATED, UNCERTAIN.",
    )
    evidence: str = Field(
        description="Verbatim documented evidence quote from the clinical record.",
    )
    section: str | None = Field(
        default=None,
        description="Clinical section where mention occurred (e.g. 'DISCHARGE_DIAGNOSES').",
    )
    reason: str = Field(
        description="Explicit clinical rationale justifying the relevance and coding decision.",
    )

    # Clinical management indicators
    treated_or_managed: bool = Field(
        default=False,
        description="True if therapeutic intervention was administered.",
    )
    monitored: bool = Field(
        default=False,
        description="True if diagnostic workup, vitals, or labs monitored this condition.",
    )
    affected_clinical_management: bool = Field(
        default=False,
        description="True if condition altered hospital course or increased nursing care.",
    )
    influenced_treatment: bool = Field(
        default=False,
        description="True if condition influenced choice of therapies or medications.",
    )
    treatment_evidence: str | None = Field(
        default=None,
        description="Documented text describing treatment or monitoring if present.",
    )

    # Audit & Abstention
    abstain_recommended: bool = Field(
        default=False,
        description="True if ambiguous documentation prevents reliable classification.",
    )
    abstention_reason: AbstentionReason | None = Field(
        default=None,
        description="Explicit reason for abstention if recommended.",
    )


# Pipeline legacy models (preserved for LangGraph compatibility)
class ExtractedDiagnosis(BaseModel):
    """Legacy diagnosis entity linking verbatim evidence."""

    diagnosis_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_term: str = Field(min_length=1)
    evidence: EvidenceSnippet
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class ContextualizedDiagnosis(BaseModel):
    """Legacy contextualized condition."""

    diagnosis_id: str
    raw_term: str
    evidence: EvidenceSnippet
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    negation: NegationStatus = Field(default=NegationStatus.AFFIRMATIVE)
    temporality: Temporality = Field(default=Temporality.CURRENT)
    certainty: Certainty = Field(default=Certainty.CONFIRMED)
    acuity: Acuity = Field(default=Acuity.UNSPECIFIED)
    clinical_justification: str = Field(min_length=1)
    context_evidence: EvidenceSnippet | None = None


class ClassifiedDiagnosis(BaseModel):
    """Legacy classified condition."""

    diagnosis_id: str
    raw_term: str
    context: ContextualizedDiagnosis
    role: DiagnosisRole
    is_billable_candidate: bool
    classification_reason: str = Field(min_length=1)
    primary_justification: str | None = None


class ConditionClassification(BaseModel):
    """Structured classification of an individual condition into a billing role.

    CRITICAL RULES:
    1. Maximum ONE Primary Diagnosis per encounter.
    2. Zero or more Secondary Diagnoses.
    3. Excluded conditions (historical without care, ruled out, negated) are not billable.
    """

    diagnosis_id: str = Field(
        description="Unique entity identifier linking to extracted/assessed condition."
    )
    diagnosis: str = Field(description="Normalized diagnostic description.")
    role: DiagnosisRole = Field(description="PRIMARY, SECONDARY, or EXCLUDED.")
    is_billable_candidate: bool = Field(description="True if condition qualifies for ICD coding.")
    classification_reason: str = Field(
        description="Clinical reasoning justifying the assigned role."
    )
    primary_justification: str | None = Field(
        default=None,
        description="Explicit documentation justification if selected as Primary diagnosis.",
    )
    admitting_condition_score: float = Field(
        default=0.0,
        description="Calculated admitting significance score based on section, intervention, and acuity.",
    )
    evidence_quote: str = Field(
        default="",
        description="Documented evidence backing this condition.",
    )


class EncounterClassificationResult(BaseModel):
    """Aggregate result of the Primary / Secondary classification component for an encounter."""

    document_id: str | None = Field(default=None, description="Encounter or document identifier.")
    primary_diagnosis: ConditionClassification | None = Field(
        default=None,
        description="The unique condition chiefly responsible for occasioning admission (max 1).",
    )
    secondary_diagnoses: list[ConditionClassification] = Field(
        default_factory=list,
        description="Co-existing conditions actively managed, monitored, or treated during admission.",
    )
    excluded_conditions: list[ConditionClassification] = Field(
        default_factory=list,
        description="Historical, negated, ruled-out, or unmanaged conditions not eligible for coding.",
    )
    all_classifications: list[ConditionClassification] = Field(
        default_factory=list,
        description="Complete list of all classified conditions.",
    )
    has_unique_primary: bool = Field(
        default=False,
        description="True if exactly one primary diagnosis was established with confidence.",
    )
    is_ambiguous_primary: bool = Field(
        default=False,
        description="True if two or more conditions equally qualify as primary without documented distinction.",
    )
    abstention_recommended: bool = Field(
        default=False,
        description="True if classification could not defensibly establish roles without physician query.",
    )
    abstention_reason: AbstentionReason | None = Field(
        default=None,
        description="Reason for abstention if recommended.",
    )
    audit_notes: list[str] = Field(
        default_factory=list,
        description="Audit trail of classification decisions and validation checks.",
    )
