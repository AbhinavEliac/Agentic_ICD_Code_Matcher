"""Schemas representing ICD-10-CM catalog codes, retrieved candidates, and ranking results."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from medical_coding.utils.text import format_icd_code, unformat_icd_code


class ICDCodeRecord(BaseModel):
    """Authoritative representation of a single code in the local clinical dataset."""

    code: str = Field(
        min_length=2,
        max_length=12,
        description="Formatted clinical code (e.g. 'I50.21', 'M8000.0', '1404').",
    )
    coding_system: str = Field(
        default="ICD-10-CM",
        description="Coding system of this record ('ICD-10-CM', 'ICD-O', 'CPT').",
    )
    description: str = Field(
        default="",
        description="Authoritative clinical description for this ICD-10-CM code.",
    )
    unformatted_code: str = Field(
        default="",
        description="Alphanumeric code without decimal (e.g. 'I5021').",
    )
    short_description: str = Field(
        default="",
        description="Standard short clinical description (CMS order file column).",
    )
    long_description: str = Field(
        default="",
        description="Full clinical description from CMS tabular dataset.",
    )
    is_valid_billable: bool = Field(
        default=True,
        description="True if terminal code at highest specificity (billable); False if category header.",
    )
    chapter: str | None = Field(
        default=None,
        description="ICD chapter (e.g., 'IX. Diseases of the circulatory system').",
    )
    category: str | None = Field(
        default=None,
        description="3-character category prefix (e.g. 'I50').",
    )
    inclusion_terms: list[str] = Field(
        default_factory=list,
        description="Official inclusion terms representing conditions classified under this code.",
    )
    exclusion_terms: list[str] = Field(
        default_factory=list,
        description="Exclusion terms (Excludes1 and Excludes2) representing non-codable conditions.",
    )
    synonyms: list[str] = Field(
        default_factory=list,
        description="Clinical synonyms, alternative phrasings, and common abbreviations for this code.",
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance and metadata for this record (e.g. release year, source file, version).",
    )
    excludes1: list[str] = Field(
        default_factory=list,
        description="Codes that cannot be billed together under Excludes1 rules.",
    )
    excludes2: list[str] = Field(
        default_factory=list,
        description="Codes representing not included conditions (Excludes2).",
    )
    code_first: list[str] = Field(
        default_factory=list,
        description="Underlying etiology codes required to precede this code.",
    )
    use_additional: list[str] = Field(
        default_factory=list,
        description="Secondary condition codes recommended to be coded alongside.",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_defaults_and_normalize(cls, data: Any) -> Any:
        """Normalize code format and sync description aliases before validation."""
        if not isinstance(data, dict):
            return data

        raw_code = str(data.get("code", "")).strip()
        formatted = format_icd_code(raw_code)
        unformatted = data.get("unformatted_code") or unformat_icd_code(raw_code)
        data["code"] = formatted
        data["unformatted_code"] = unformatted

        desc = data.get("description", "")
        short_desc = data.get("short_description", "")
        long_desc = data.get("long_description", "")

        if not desc:
            desc = long_desc or short_desc
        if not short_desc:
            short_desc = desc
        if not long_desc:
            long_desc = desc

        data["description"] = desc
        data["short_description"] = short_desc
        data["long_description"] = long_desc

        if not data.get("category") and formatted:
            clean = unformatted
            data["category"] = clean[:3] if len(clean) >= 3 else clean

        # Combine exclusion terms with excludes1/excludes2 if provided
        exclusions = list(data.get("exclusion_terms", []))
        if not exclusions:
            ex1 = data.get("excludes1", [])
            ex2 = data.get("excludes2", [])
            exclusions = list(ex1) + list(ex2)
        data["exclusion_terms"] = exclusions

        return data

    def get_searchable_text(self) -> str:
        """Combine all text representations for comprehensive lexical and semantic retrieval."""
        parts: list[str] = []
        if self.description:
            parts.append(self.description)
        if self.long_description and self.long_description != self.description:
            parts.append(self.long_description)
        if self.short_description and self.short_description not in (
            self.description,
            self.long_description,
        ):
            parts.append(self.short_description)
        if self.inclusion_terms:
            parts.extend(self.inclusion_terms)
        if self.synonyms:
            parts.extend(self.synonyms)
        return " | ".join(parts)


class ICDDatasetStats(BaseModel):
    """Statistical summary of loaded ICD-10-CM dataset."""

    total_records: int = Field(description="Total valid records indexed.")
    valid_billable_count: int = Field(description="Count of billable terminal codes.")
    non_billable_count: int = Field(description="Count of category headers / non-billable codes.")
    unique_categories_count: int = Field(description="Count of unique 3-character categories.")
    duplicates_dropped: int = Field(default=0, description="Duplicate code occurrences filtered.")
    malformed_dropped: int = Field(default=0, description="Malformed or invalid rows filtered.")
    total_inclusion_terms: int = Field(default=0, description="Total inclusion terms indexed.")
    total_exclusion_terms: int = Field(default=0, description="Total exclusion terms indexed.")
    total_synonyms: int = Field(default=0, description="Total synonyms indexed.")
    format_type: str = Field(default="tabular", description="Source format loaded.")


class ICDCandidate(BaseModel):
    """An authoritative candidate code retrieved from the local dataset for an extracted diagnosis."""

    code: str = Field(description="Authoritative code from local catalog.")
    description: str = Field(description="Official clinical description matching the code.")
    coding_system: str = Field(
        default="ICD-10-CM",
        description="Coding system of this candidate ('ICD-10-CM', 'ICD-O', 'CPT').",
    )
    retrieval_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized similarity/relevance score from retrieval engine.",
    )
    retrieval_method: Literal[
        "bm25",
        "faiss",
        "hybrid",
        "family_filtered_lexical",
        "fallback_token_scan",
        "vector_numpy",
    ] = Field(
        description="Algorithm that produced or consolidated this candidate.",
    )
    is_valid_billable: bool = Field(
        description="Indicates whether this candidate is terminal/billable in the catalog.",
    )
    semantic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Semantic similarity component score from FAISS vector search.",
    )
    lexical_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Lexical BM25 component score normalized to [0, 1].",
    )
    category: str | None = Field(
        default=None,
        description="Category prefix or group identifier.",
    )


class RankedSelection(BaseModel):
    """Result of evaluating and ranking retrieved candidates against clinical evidence."""

    diagnosis_id: str = Field(default="", description="Associated diagnosis entity ID.")
    raw_term: str = Field(default="", description="Original clinical term.")
    selected_code: str | None = Field(
        default=None,
        description="The chosen code strictly from the candidate list; None if abstained.",
    )
    selected_description: str | None = Field(
        default=None,
        description="Official description of the selected clinical code.",
    )
    selected_icd10cm: str | None = Field(
        default=None,
        description="Matched ICD-10-CM code if matched, else None.",
    )
    selected_icdo: str | None = Field(
        default=None,
        description="Matched ICD-O morphology code if matched, else None.",
    )
    selected_cpt: str | None = Field(
        default=None,
        description="Matched CPT code if matched, else None.",
    )
    ranking_reason: str = Field(
        default="",
        description="Detailed clinical justification linking evidence to the selected code or explaining abstention.",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim evidence quotes or snippets supporting the selection.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the ranking selection.",
    )
    abstention_reason: str | None = Field(
        default=None,
        description="Reason for abstention or rejection if no candidate was accepted.",
    )
    selected_candidate: ICDCandidate | None = Field(
        default=None,
        description="Top-ranked candidate chosen strictly from retrieved list; None if abstained.",
    )
    ranking_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the ranking selection (alias for confidence).",
    )
    selection_justification: str = Field(
        default="",
        description="Evidence-based justification linking clinical wording to chosen ICD description.",
    )
    candidate_pool: list[ICDCandidate] = Field(
        default_factory=list,
        description="All candidates considered during ranking (audit trail).",
    )
    decision: Literal["ACCEPTED", "REJECTED_LOW_CONFIDENCE", "REJECTED_MISMATCH", "ABSTAINED"] = (
        Field(
            default="ABSTAINED",
            description="Outcome of candidate evaluation.",
        )
    )
    matching_status: Literal["MATCHED", "NO_DATABASE_MATCH"] = Field(
        default="NO_DATABASE_MATCH",
        description="Whether a database code was matched or no match was found in the database.",
    )
    source_section: str | None = Field(
        default=None,
        description="Source clinical section documenting the diagnosis.",
    )
    source_span: tuple[int, int] | None = Field(
        default=None,
        description="Character span offsets in original document text.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_aliases_and_candidates(cls, data: Any) -> Any:
        """Synchronize aliases between modern and legacy ranking fields."""
        if not isinstance(data, dict):
            return data

        # Sync justification / reason
        reason = data.get("ranking_reason") or data.get("selection_justification", "")
        data["ranking_reason"] = reason
        data["selection_justification"] = reason

        # Sync score / confidence
        conf = data.get("confidence")
        score = data.get("ranking_score")
        if conf is not None:
            data["confidence"] = float(conf)
            data["ranking_score"] = float(conf)
        elif score is not None:
            data["confidence"] = float(score)
            data["ranking_score"] = float(score)

        # Sync candidate and code/description
        candidate = data.get("selected_candidate")
        if candidate and hasattr(candidate, "code"):
            data.setdefault("selected_code", candidate.code)
            data.setdefault("selected_description", candidate.description)
        elif isinstance(candidate, dict):
            data.setdefault("selected_code", candidate.get("code"))
            data.setdefault("selected_description", candidate.get("description"))

        # If selected_code is given but not candidate, lookup in pool
        if data.get("selected_code") and not data.get("selected_candidate"):
            pool = data.get("candidate_pool", [])
            for c in pool:
                c_code = c.code if hasattr(c, "code") else c.get("code")
                if c_code == data["selected_code"]:
                    data["selected_candidate"] = c
                    data.setdefault(
                        "selected_description",
                        c.description if hasattr(c, "description") else c.get("description"),
                    )
                    break

        if "matching_status" not in data or not data["matching_status"]:
            data["matching_status"] = "MATCHED" if data.get("selected_code") else "NO_DATABASE_MATCH"

        return data
