"""Structured clinical evidence objects, multi-dimensional scores, and decoupled diagnosis/mapping states."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from medical_coding.schemas.enums import (
    AssertionStatus,
    Certainty,
    DiagnosisRole,
    EvidenceType,
    NegationStatus,
    SectionSemantics,
    Temporality,
)


class StructuredEvidence(BaseModel):
    """Verbatim documentary evidence object extracted directly from clinical documentation."""

    text: str = Field(
        min_length=1,
        description="Verbatim sentence or phrase from the document providing factual proof.",
    )
    section: str = Field(
        default="CLINICAL_DOCUMENT",
        description="Source clinical section (e.g. 'DISCHARGE_DIAGNOSES', 'HOSPITAL_COURSE', 'RELEVANT_INVESTIGATIONS').",
    )
    section_semantics: SectionSemantics = Field(
        default=SectionSemantics.UNKNOWN,
        description="Semantic category of the source section providing evidence hierarchy weighting.",
    )
    sentence: str | None = Field(
        default=None,
        description="Complete containing sentence if available.",
    )
    polarity: NegationStatus = Field(
        default=NegationStatus.AFFIRMATIVE,
        description="AFFIRMATIVE, NEGATED, or UNCERTAIN.",
    )
    certainty: Certainty = Field(
        default=Certainty.CONFIRMED,
        description="CONFIRMED, SUPPORTED, SUSPECTED, POSSIBLE, RULED_OUT.",
    )
    temporality: Temporality = Field(
        default=Temporality.CURRENT,
        description="CURRENT, HISTORICAL, RESOLVED, FAMILY_HISTORY.",
    )
    subject: str = Field(
        default="PATIENT",
        description="Subject of clinical finding: 'PATIENT', 'FAMILY_MEMBER', 'DONOR'.",
    )
    evidence_type: EvidenceType = Field(
        default=EvidenceType.DISCHARGE_SUMMARY,
        description="Type of evidence: DISCHARGE_SUMMARY, ADMISSION_REASON, PROCEDURAL, IMAGING, LAB, HISTORY.",
    )
    clinical_relevance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Clinical relevance weight to the current inpatient encounter.",
    )

    @property
    def quote(self) -> str:
        """Alias for quote compatibility across legacy schemas."""
        return self.text

    @model_validator(mode="before")
    @classmethod
    def populate_quote_or_text(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "quote" in data and "text" not in data:
                data["text"] = data["quote"]
        return data


class MultiDimensionalScore(BaseModel):
    """Interpretable multi-dimensional scoring dimensions for clinical authorization and coding."""

    evidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Verbatim textual grounding quality."
    )
    diagnostic_certainty: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Certainty score: Confirmed=1.0, Supported=0.8, Suspected=0.5, Ruled_Out=0.0."
    )
    encounter_relevance: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Active inpatient management / therapeutic relevance."
    )
    role_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in role classification hierarchy."
    )
    semantic_match: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Clinical semantic alignment with ICD description."
    )
    specificity_match: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Laterality, acuity, and etiology alignment."
    )
    contradiction_penalty: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Penalty for conflicting, negated, or questioned documentation."
    )
    admitting_score: float = Field(
        default=0.0, description="Composite admitting hierarchy score."
    )

    def is_authorized(self, min_certainty: float = 0.5, max_contradiction: float = 0.5) -> bool:
        """Evaluate if clinical candidate passes deterministic clinical authorization."""
        return (
            self.evidence_score > 0.0
            and self.diagnostic_certainty >= min_certainty
            and self.contradiction_penalty <= max_contradiction
        )


class ClinicalDiagnosisCandidate(BaseModel):
    """Normalized clinical diagnosis candidate authorized by clinical facts before ICD retrieval."""

    diagnosis_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique diagnosis candidate identifier.",
    )
    raw_term: str = Field(
        min_length=1,
        description="Exact term as mentioned in source documentation.",
    )
    normalized_diagnosis: str = Field(
        min_length=1,
        description="Canonical medical description without ICD code assumptions.",
    )
    role: DiagnosisRole = Field(
        default=DiagnosisRole.SECONDARY,
        description="PRIMARY, SECONDARY, HISTORICAL, SYMPTOM, INCIDENTAL, RULED_OUT, UNCERTAIN.",
    )
    assertion_status: AssertionStatus | str = Field(
        default=AssertionStatus.CONFIRMED,
        description="confirmed, suspected, ruled_out, historical, family_history, hypothetical, negated, uncertain.",
    )
    certainty: Certainty = Field(
        default=Certainty.CONFIRMED,
        description="CONFIRMED, SUPPORTED, SUSPECTED, POSSIBLE, RULED_OUT, NEGATED.",
    )
    temporality: Temporality = Field(
        default=Temporality.CURRENT,
        description="CURRENT, HISTORICAL, RESOLVED, FUTURE, UNCLEAR.",
    )
    evidence: list[StructuredEvidence] = Field(
        default_factory=list,
        description="Collection of verbatim evidence objects supporting this diagnosis.",
    )
    evidence_strength: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Aggregate strength of documentary evidence."
    )
    clinical_relevance: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Inpatient clinical relevance."
    )
    encounter_relevance: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Encounter relevance."
    )
    treatment_relevance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Active medical or surgical treatment relevance."
    )
    procedure_relevance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Major operative intervention / source control relevance."
    )
    scores: MultiDimensionalScore = Field(
        default_factory=MultiDimensionalScore,
        description="Interpretable scoring vector.",
    )
    is_authorized: bool = Field(
        default=True,
        description="True if authorized by clinical documentation to enter coding.",
    )
    primary_justification: str | None = Field(
        default=None,
        description="Evidence rationale justifying selection as Primary diagnosis.",
    )
    classification_reason: str = Field(
        default="",
        description="Summary rationale of role assignment.",
    )
    clinical_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit attributes of the diagnosis (stage, subtype, grade, laterality, acuity, size).",
    )
    compound_relationship: str | None = Field(
        default=None,
        description="Semantic compound relationship: 'parent', 'associated', 'causal', 'complication', 'independent'.",
    )
    associated_conditions: list[str] = Field(
        default_factory=list,
        description="Secondary conditions or manifestations causally or etiologically linked to this candidate.",
    )
    management_evidence: list[str] = Field(
        default_factory=list,
        description="Documented inpatient therapeutic interventions, medications, or monitoring for this condition.",
    )
    admission_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Documentary weight indicating this condition occasioned the hospital admission.",
    )
    discharge_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Documentary weight indicating this condition was assessed or confirmed at discharge.",
    )
    contradiction_evidence: list[str] = Field(
        default_factory=list,
        description="Statements in documentation questioning, ruling out, or refuting this condition.",
    )
    medication_evidence: list[str] = Field(
        default_factory=list,
        description="Medications administered or prescribed relevant to this condition.",
    )
    procedure_evidence: list[str] = Field(
        default_factory=list,
        description="Diagnostic or surgical procedures performed targeting this condition.",
    )
    candidate_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in this clinical candidate being genuine and codable.",
    )

    @property
    def primary_evidence_quote(self) -> str:
        """Return the highest priority evidence text quote."""
        if not self.evidence:
            return ""
        # Prefer discharge summary, admission, procedure, or hospital course quotes
        for ev in self.evidence:
            if ev.evidence_type in (
                EvidenceType.DISCHARGE_SUMMARY,
                EvidenceType.ADMISSION_REASON,
                EvidenceType.PROCEDURE,
                EvidenceType.HOSPITAL_COURSE,
            ):
                return ev.text
        return self.evidence[0].text


class ClinicalDiagnosisState(BaseModel):
    """Encounter clinical diagnosis state established strictly prior to ICD candidate retrieval."""

    document_id: str
    primary_diagnosis: ClinicalDiagnosisCandidate | None = None
    secondary_diagnoses: list[ClinicalDiagnosisCandidate] = Field(default_factory=list)
    historical_conditions: list[ClinicalDiagnosisCandidate] = Field(default_factory=list)
    ruled_out_conditions: list[ClinicalDiagnosisCandidate] = Field(default_factory=list)
    uncertain_conditions: list[ClinicalDiagnosisCandidate] = Field(default_factory=list)
    all_candidates: list[ClinicalDiagnosisCandidate] = Field(default_factory=list)
    has_unique_primary: bool = False
    audit_notes: list[str] = Field(default_factory=list)


class ICDMappingState(BaseModel):
    """State tracking the mapping of an authorized clinical diagnosis to an ICD code."""

    diagnosis_id: str
    normalized_diagnosis: str
    role: DiagnosisRole
    selected_code: str | None = None
    selected_description: str | None = None
    selected_icd10cm: str | None = None
    selected_icdo: str | None = None
    selected_cpt: str | None = None
    is_terminal_billable: bool = False
    confidence_score: float = 0.0
    mapping_status: str = "PENDING"  # ACCEPTED, ABSTAINED, REJECTED
    abstention_reason: str | None = None
    rejection_detail: str | None = None
    validation_passed: bool = False
    validation_checks: list[dict[str, Any]] = Field(default_factory=list)
    candidate_matches: list[dict[str, Any]] = Field(default_factory=list)


# Alias for mapping candidate representation
ICDMappingCandidate = ICDMappingState


class AuditTrailEntry(BaseModel):
    """Verifiable clinical audit trail entry conforming to Section 15 specifications."""

    diagnosis: str
    role: str
    evidence: list[dict[str, Any]]
    reason_for_acceptance: str
    rejected_alternatives: list[str] = Field(default_factory=list)
    icd_candidates: list[dict[str, Any]] = Field(default_factory=list)
    selected_code: str | None = None
    validation_checks: list[dict[str, Any]] = Field(default_factory=list)


class CancerConcept(BaseModel):
    """Structured oncology concept isolating tumor biology, staging, and metastases from primary disease."""

    primary_site: str = Field(description="Primary anatomical site (e.g. 'breast', 'colon', 'lymph node').")
    laterality: str | None = Field(default=None, description="'right', 'left', 'bilateral', or None.")
    malignancy_type: str = Field(default="carcinoma", description="'carcinoma', 'lymphoma', 'sarcoma', 'melanoma', etc.")
    histology: str | None = Field(default=None, description="Histological type (e.g. 'invasive ductal', 'DLBCL').")
    grade: str | None = Field(default=None, description="Tumor grade (e.g. 'Grade III').")
    stage: str | None = Field(default=None, description="Staging (e.g. 'Stage IV').")
    metastatic_status: str = Field(default="NON_METASTATIC", description="'NON_METASTATIC', 'METASTATIC', 'UNKNOWN'.")
    metastatic_sites: list[str] = Field(default_factory=list, description="Verified metastatic anatomical sites.")
    receptor_status: dict[str, str] = Field(default_factory=dict, description="e.g. {'ER': 'negative', 'PR': 'negative', 'HER2': '0', 'PD-L1': 'positive'}.")
    molecular_attributes: list[str] = Field(default_factory=list, description="e.g. ['BRCA pathogenic', 'BRCA1 positive'].")
    treatment_history: list[str] = Field(default_factory=list, description="Past chemotherapy/radiotherapy/surgery.")
    current_treatment: list[str] = Field(default_factory=list, description="Current inpatient chemotherapy or management.")
    evidence: str = Field(default="", description="Verbatim documentary evidence.")


class OrthopedicConcept(BaseModel):
    """Structured orthopedic fracture and ligament concept separating anatomical attributes from disease entity."""

    bone: str = Field(description="Specific bone (e.g. 'lateral malleolus', 'fifth metatarsal', 'tarsal').")
    anatomical_site: str = Field(default="", description="Region (e.g. 'fibula', 'foot', 'ankle').")
    laterality: str | None = Field(default=None, description="'left', 'right', 'bilateral', or None.")
    displacement: str = Field(default="unspecified", description="'displaced', 'nondisplaced', 'unspecified'.")
    open_closed: str = Field(default="closed", description="'open', 'closed', 'unspecified'.")
    healing_status: str = Field(default="routine", description="'routine', 'delayed', 'nonunion', 'malunion', 'unspecified'.")
    encounter_phase: str = Field(default="initial", description="'initial', 'subsequent', 'sequela'.")
    associated_ligament_injury: list[str] = Field(default_factory=list, description="e.g. ['deltoid ligament sprain', 'calcaneofibular ligament sprain'].")
    evidence: str = Field(default="", description="Verbatim documentary evidence.")


class ExcludedCandidate(BaseModel):
    """Record of a rejected non-diagnostic term, medication, attribute, or metadata artifact."""

    text: str = Field(description="Raw term rejected from diagnosis coding.")
    reason: str = Field(description="Category: 'MEDICATION', 'ATTRIBUTE', 'NEGATED', 'HISTORICAL', 'METADATA', 'SYMPTOM', 'PROCEDURE', 'ANATOMY', 'UNSUPPORTED_SPECIFICITY'.")


class DiagnosisCandidate(ClinicalDiagnosisCandidate):
    """Typed diagnosis candidate strictly conforming to Section 6 specifications."""

    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_text: str = Field(default="")
    normalized_concept: str = Field(default="")
    diagnosis_family: str = Field(default="")
    evidence_text: str = Field(default="")
    source_section: str = Field(default="")
    source_span: tuple[int, int] | None = Field(default=None)
    clinical_status: str = Field(default="ACTIVE")
    current_relevance: float = Field(default=1.0)
    diagnostic_evidence: list[str] = Field(default_factory=list)
    supporting_sections: list[str] = Field(default_factory=list)
    unsupported_attributes: list[str] = Field(default_factory=list)
    specificity_supported: bool = Field(default=True)
    candidate_status: str = Field(default="ACTIVE")
    rejection_reason: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def sync_candidate_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "raw_term" in data and not data.get("raw_text"):
                data["raw_text"] = data["raw_term"]
            elif "raw_text" in data and not data.get("raw_term"):
                data["raw_term"] = data["raw_text"]
            if "normalized_diagnosis" in data and not data.get("normalized_concept"):
                data["normalized_concept"] = data["normalized_diagnosis"]
            elif "normalized_concept" in data and not data.get("normalized_diagnosis"):
                data["normalized_diagnosis"] = data["normalized_concept"]
            if "diagnosis_id" in data and not data.get("candidate_id"):
                data["candidate_id"] = data["diagnosis_id"]
            elif "candidate_id" in data and not data.get("diagnosis_id"):
                data["diagnosis_id"] = data["candidate_id"]
        return data

