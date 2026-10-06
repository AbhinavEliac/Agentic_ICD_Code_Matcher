"""Typed Schemas for Structured Clinical Concept Reasoning and Database Matching Constraints.

Conforms strictly to Master Specification Sections 5, 6, 7, and 10:
- ClinicalConcept: Normalized clinical entity with explicit supported/unknown attributes.
- MatchSpec: Deterministic retrieval constraint defining allowed code families and forbidden attributes.
- CompatibilityResult: Post-retrieval validation verifying semantic equivalence and attribute compliance.
"""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ClinicalConcept(BaseModel):
    """Generic Clinical Concept conforming to Master Directive Section 26.

    Enforces the fundamental clinical invariant:
        CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY
    Attributes not supported by explicit documentation remain in unknown_attributes.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique concept identifier linking back to diagnosis candidate.",
    )
    concept_id: str = Field(
        default="",
        description="Alias for concept id.",
    )
    raw_mentions: list[str] = Field(
        default_factory=list,
        description="Original text spans from the document.",
    )
    canonical_concept: str = Field(
        default="",
        description="Normalized clinical diagnosis name without database code assumptions.",
    )
    canonical_name: str = Field(
        default="",
        description="Alias for canonical concept.",
    )
    normalized_concept: str = Field(
        default="",
        description="Fully normalized clinical concept string.",
    )
    entity_type: str = Field(
        default="DIAGNOSIS",
        description="Category from Universal Clinical Entity Taxonomy (Section 2).",
    )
    disease_family: str = Field(
        default="",
        description="High-level clinical disease family (e.g. 'breast_malignancy', 'renal_calculus').",
    )
    anatomy: str | None = Field(
        default=None,
        description="Primary anatomical site (e.g. 'breast', 'kidney', 'knee', 'liver').",
    )
    body_site: str | None = Field(
        default=None,
        description="Alias for anatomical site.",
    )
    laterality: str | None = Field(
        default=None,
        description="Documented anatomical laterality: 'left', 'right', 'bilateral', or None.",
    )
    subsite: str | None = Field(
        default=None,
        description="Specific anatomical subsite if documented.",
    )
    etiology: str | None = Field(
        default=None,
        description="Etiological organism or causal factor (e.g. 'helicobacter_pylori', 'e_coli').",
    )
    organism: str | None = Field(
        default=None,
        description="Specific infectious pathogen if documented.",
    )
    histology: str | None = Field(
        default=None,
        description="Pathological histology (e.g. 'carcinoma', 'dlbcl', 'sarcoma', 'carcinoid').",
    )
    grade: str | None = Field(
        default=None,
        description="Pathologic or histologic grade.",
    )
    stage: str | None = Field(
        default=None,
        description="Clinical or pathological stage.",
    )
    severity: str | None = Field(
        default=None,
        description="Documented severity qualifier (e.g. 'mild', 'moderate', 'severe').",
    )
    complication: str | None = Field(
        default=None,
        description="Associated manifestation or complication.",
    )
    acuity: str | None = Field(
        default=None,
        description="Acuity profile: 'acute', 'chronic', 'acute_on_chronic', 'unspecified'.",
    )
    encounter_context: str | None = Field(
        default=None,
        description="Encounter context: 'initial', 'subsequent', 'sequela'.",
    )
    assertion_status: str = Field(
        default="CONFIRMED",
        description="CONFIRMED, SUSPECTED, UNCERTAIN, RULED_OUT, NEGATED, HISTORICAL, etc.",
    )
    temporality: str = Field(
        default="CURRENT",
        description="CURRENT, HISTORICAL, RESOLVED, FUTURE, UNCLEAR.",
    )
    temporal_status: str = Field(
        default="CURRENT",
        description="Alias for temporality.",
    )
    role: str = Field(
        default="SECONDARY",
        description="Inferred initial clinical role: PRIMARY, SECONDARY, or NON_CODABLE.",
    )
    role_hint: str = Field(
        default="SECONDARY",
        description="Alias for role.",
    )
    metastatic_status: bool = Field(
        default=False,
        description="True if cancer is documented as metastatic or secondary.",
    )
    metastatic_sites: list[str] = Field(
        default_factory=list,
        description="Explicitly documented secondary/metastatic anatomical sites.",
    )
    evidence_spans: list[str] = Field(
        default_factory=list,
        description="Verbatim textual evidence quotes supporting this clinical concept.",
    )
    source_sections: list[str] = Field(
        default_factory=list,
        description="Sections where mentions of this concept occurred.",
    )
    supporting_sections: list[str] = Field(
        default_factory=list,
        description="Alias for source sections.",
    )
    supporting_evidence: list[dict[str, Any] | str] = Field(
        default_factory=list,
        description="Detailed structured evidence objects or text strings.",
    )
    contradiction_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence statements contradicting this candidate.",
    )
    related_concepts: list[str] = Field(
        default_factory=list,
        description="Concepts related to this entity in the clinical evidence graph.",
    )
    treatment_support: list[str] = Field(
        default_factory=list,
        description="Documented treatments or active therapies supporting this condition.",
    )
    diagnostic_support: list[str] = Field(
        default_factory=list,
        description="Diagnostic investigations or pathology corroborating this condition.",
    )
    supported_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes explicitly supported by clinical documentation.",
    )
    unknown_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes relevant to the disease family that are NOT documented.",
    )
    contradicted_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes contradicted or ruled out by clinical evidence.",
    )
    contradictory_attributes: list[str] = Field(
        default_factory=list,
        description="Alias for contradicted_attributes.",
    )
    clinical_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Clinical confidence score for this concept.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Alias for clinical_confidence.",
    )
    coding_system: str = Field(
        default="ICD-10-CM",
        description="Target coding system: 'ICD-10-CM', 'ICD-O', 'CPT'.",
    )
    database_candidates: list[Any] = Field(
        default_factory=list,
        description="Candidates retrieved from the authoritative local database.",
    )
    selected_database_record: Any | None = Field(
        default=None,
        description="Final selected database record after ranking and validation.",
    )
    validation_status: str = Field(
        default="VALIDATED",
        description="Validation outcome: 'VALIDATED', 'DOWNGRADED', 'REJECTED', 'ABSTAINED'.",
    )

    def model_post_init(self, __context: Any) -> None:
        """Synchronize aliases after initialization."""
        if not self.concept_id and self.id:
            self.concept_id = self.id
        elif not self.id and self.concept_id:
            self.id = self.concept_id
        if not self.canonical_concept and self.canonical_name:
            self.canonical_concept = self.canonical_name
        elif not self.canonical_name and self.canonical_concept:
            self.canonical_name = self.canonical_concept
        if not self.normalized_concept and self.canonical_concept:
            self.normalized_concept = self.canonical_concept
        if not self.anatomy and self.body_site:
            self.anatomy = self.body_site
        elif not self.body_site and self.anatomy:
            self.body_site = self.anatomy
        if not self.temporality and self.temporal_status:
            self.temporality = self.temporal_status
        elif not self.temporal_status and self.temporality:
            self.temporal_status = self.temporality
        if not self.role and self.role_hint:
            self.role = self.role_hint
        elif not self.role_hint and self.role:
            self.role_hint = self.role
        if not self.source_sections and self.supporting_sections:
            self.source_sections = list(self.supporting_sections)
        elif not self.supporting_sections and self.source_sections:
            self.supporting_sections = list(self.source_sections)
        if not self.contradicted_attributes and self.contradictory_attributes:
            self.contradicted_attributes = list(self.contradictory_attributes)
        elif not self.contradictory_attributes and self.contradicted_attributes:
            self.contradictory_attributes = list(self.contradicted_attributes)
        if self.clinical_confidence != self.confidence:
            self.confidence = self.clinical_confidence


class MatchSpec(BaseModel):
    """Deterministic database retrieval constraint built from ClinicalConcept (Sections 16 & 26).

    Constrains database retrieval so vector similarity operates ONLY within
    clinically compatible candidate subsets.
    """

    concept_id: str = Field(
        description="ID of the governing ClinicalConcept.",
    )
    canonical_name: str = Field(
        description="Canonical diagnosis term to match.",
    )
    disease_family: str = Field(
        description="Disease family classification.",
    )
    anatomical_site: str | None = Field(
        default=None,
        description="Target anatomical site constraint.",
    )
    laterality: str | None = Field(
        default=None,
        description="Target anatomical laterality constraint.",
    )
    etiology: str | None = Field(
        default=None,
        description="Documented etiology constraint.",
    )
    histology: str | None = Field(
        default=None,
        description="Pathological histology constraint.",
    )
    severity: str | None = Field(
        default=None,
        description="Severity qualifier constraint.",
    )
    complication: str | None = Field(
        default=None,
        description="Associated complication constraint.",
    )
    temporal_status: str = Field(
        default="CURRENT",
        description="Temporal status constraint.",
    )
    encounter_context: str | None = Field(
        default=None,
        description="Encounter context constraint (initial, subsequent, sequela).",
    )
    allowed_code_families: list[str] = Field(
        default_factory=list,
        description="ICD-10-CM code category prefixes permitted for this concept (e.g. ['C50.*']).",
    )
    required_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Attributes that MUST be present or compatible in the candidate code description.",
    )
    forbidden_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes that, if present in the database description, immediately disqualify the code.",
    )
    unknown_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes not supported by evidence; codes requiring them must be rejected.",
    )
    relationship_constraints: list[str] | dict[str, Any] = Field(
        default_factory=list,
        description="Constraints on clinical relationships (e.g. ['primary_only', 'secondary_only']).",
    )
    coding_system: str = Field(
        default="ICD-10-CM",
        description="Target database coding system: 'ICD-10-CM', 'ICD-O', or 'CPT'.",
    )
    target_coding_system: str = Field(
        default="ICD-10-CM",
        description="Alias for coding_system.",
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.coding_system and self.target_coding_system:
            self.coding_system = self.target_coding_system
        elif not self.target_coding_system and self.coding_system:
            self.target_coding_system = self.coding_system


class CompatibilityResult(BaseModel):
    """Result of Compatibility Reasoner validating a database candidate against ClinicalConcept."""

    compatible: bool = Field(
        description="True if candidate represents the established concept without unsupported specificity.",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite compatibility score.",
    )
    candidate_code: str = Field(
        description="Official database code evaluated.",
    )
    candidate_description: str = Field(
        description="Official database description evaluated.",
    )
    supported_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes in the candidate description verified as documented.",
    )
    unsupported_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes in the candidate description lacking evidence in documentation.",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Contradictions (e.g. candidate is secondary neoplasm when site is primary).",
    )
    reason: str = Field(
        description="Clinical rationale justifying candidate acceptance or rejection.",
    )
