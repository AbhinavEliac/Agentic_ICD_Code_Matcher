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
    """Normalized clinical concept derived from clinical evidence prior to database retrieval.

    Enforces the fundamental clinical invariant:
        CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY
    Attributes not supported by explicit documentation remain in unknown_attributes.
    """

    concept_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique concept identifier linking back to diagnosis candidate.",
    )
    canonical_name: str = Field(
        description="Normalized clinical diagnosis name without database code assumptions.",
    )
    disease_family: str = Field(
        description="High-level clinical disease family (e.g. 'breast_malignancy', 'renal_calculus', 'gastritis').",
    )
    body_site: str | None = Field(
        default=None,
        description="Primary anatomical site (e.g. 'breast', 'kidney', 'ureter', 'stomach', 'fibula').",
    )
    laterality: str | None = Field(
        default=None,
        description="Documented anatomical laterality: 'left', 'right', 'bilateral', or None.",
    )
    etiology: str | None = Field(
        default=None,
        description="Etiological organism or causal factor (e.g. 'helicobacter_pylori', 'e_coli').",
    )
    histology: str | None = Field(
        default=None,
        description="Pathological histology (e.g. 'carcinoma', 'dlbcl', 'adenocarcinoma').",
    )
    severity: str | None = Field(
        default=None,
        description="Documented severity qualifier (e.g. 'mild', 'moderate', 'severe').",
    )
    complication: str | None = Field(
        default=None,
        description="Associated manifestation or complication (e.g. 'septic_shock', 'renal_colic').",
    )
    temporal_status: str = Field(
        default="CURRENT",
        description="CURRENT, HISTORICAL, RESOLVED, or UNCLEAR.",
    )
    assertion_status: str = Field(
        default="CONFIRMED",
        description="CONFIRMED, SUSPECTED, RULED_OUT, or NEGATED.",
    )
    role_hint: str = Field(
        default="SECONDARY",
        description="Inferred initial clinical role: PRIMARY, SECONDARY, or EXCLUDED.",
    )
    metastatic_status: bool = Field(
        default=False,
        description="True if cancer is documented as metastatic or secondary.",
    )
    metastatic_sites: list[str] = Field(
        default_factory=list,
        description="Explicitly documented secondary/metastatic anatomical sites (e.g. ['pleura']).",
    )
    evidence_spans: list[str] = Field(
        default_factory=list,
        description="Verbatim textual evidence quotes supporting this clinical concept.",
    )
    supporting_sections: list[str] = Field(
        default_factory=list,
        description="Sections where mentions of this concept occurred.",
    )
    supported_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes explicitly supported by clinical documentation.",
    )
    unknown_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes relevant to the disease family that are NOT documented.",
    )
    contradictory_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes contradicted or ruled out by clinical evidence.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Clinical confidence score for this concept.",
    )


class MatchSpec(BaseModel):
    """Deterministic database retrieval constraint built from ClinicalConcept.

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
    target_coding_system: str = Field(
        default="ICD-10-CM",
        description="Target database coding system: 'ICD-10-CM', 'ICD-O', or 'CPT'.",
    )


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
