"""Final deterministic JSON output models and API request/response schemas."""

import re
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
    matching_status: Literal["MATCHED", "NEAREST_MATCH", "NO_DATABASE_MATCH"] = Field(
        default="MATCHED",
        description="Whether this diagnosis matched a database code, nearest relatable match, or abstained as NO_DATABASE_MATCH.",
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
    procedures: list[str] = Field(
        default_factory=list,
        description="Extracted procedural and surgical events segregated from diagnoses.",
    )
    operative_findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Operative and intraoperative findings evaluated under distinct clinical gate.",
    )
    oncology_context: dict[str, Any] | None = Field(
        default=None,
        description="Structured oncology biomarkers and tumor biology attributes.",
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
        """Enforce strict invariant: primary diagnosis must have PRIMARY role if present,
        and drop duplicates and repetitive evidences across diagnoses, abstentions, and procedures.
        """
        if self.primary_diagnosis and self.primary_diagnosis.role != DiagnosisRole.PRIMARY:
            raise ValueError("Primary diagnosis must have role PRIMARY")
        for sec in self.secondary_diagnoses:
            if sec.role == DiagnosisRole.PRIMARY:
                raise ValueError("Secondary diagnoses cannot have role PRIMARY")

        from medical_coding.utils.text import (
            clean_and_deduplicate_evidence_list,
            clean_and_deduplicate_evidence_quote,
            normalize_diagnosis_name,
        )

        # 1. Clean and deduplicate primary evidence
        pri_code = ""
        pri_clean_norm = ""
        if self.primary_diagnosis:
            p = self.primary_diagnosis
            p_name = p.normalized_diagnosis or p.raw_term or p.description or ""
            p.evidence_quote = clean_and_deduplicate_evidence_quote(p.evidence_quote, p_name)
            p.evidence = clean_and_deduplicate_evidence_list(getattr(p, "evidence", []) or p.evidence_quote, p_name)
            pri_code = (p.code or getattr(p, "database_code", "") or "").strip()
            pri_norm = (p_name or "").lower().strip()
            pri_clean_norm = (normalize_diagnosis_name(pri_norm) or pri_norm).lower().strip()

        # 2. Deduplicate secondary diagnoses against primary and among themselves
        deduped_secondaries: list[CodedDiagnosisResponse] = []
        seen_sec_codes: set[str] = set()
        seen_sec_names: set[str] = set()

        for s in (self.secondary_diagnoses or []):
            s_code = (s.code or getattr(s, "database_code", "") or "").strip()
            s_name = (s.normalized_diagnosis or s.raw_term or s.description or "").strip()
            s_norm = s_name.lower().strip()
            s_clean_norm = (normalize_diagnosis_name(s_norm) or s_norm).lower().strip()

            # Skip secondary if it duplicates the primary diagnosis
            if pri_code and s_code and s_code == pri_code:
                continue
            if pri_clean_norm and s_clean_norm and pri_clean_norm == s_clean_norm:
                continue

            # Check if already seen among secondaries
            existing_match: CodedDiagnosisResponse | None = None
            if s_code and s_code not in ("None", "[NO LOCAL CODE]") and s_code in seen_sec_codes:
                existing_match = next((x for x in deduped_secondaries if (x.code or "").strip() == s_code), None)
            elif s_clean_norm and s_clean_norm in seen_sec_names:
                existing_match = next(
                    (x for x in deduped_secondaries if (normalize_diagnosis_name(x.normalized_diagnosis or x.raw_term or x.description or "") or "").lower().strip() == s_clean_norm),
                    None,
                )

            if existing_match:
                # Merge evidence cleanly without repetition
                combined_ev = f"{existing_match.evidence_quote or ''} | {s.evidence_quote or ''}".strip(" |")
                merged_diag_name = existing_match.normalized_diagnosis or existing_match.description
                existing_match.evidence_quote = clean_and_deduplicate_evidence_quote(combined_ev, merged_diag_name)
                merged_ev_list = (getattr(existing_match, "evidence", []) or []) + (getattr(s, "evidence", []) or [])
                existing_match.evidence = clean_and_deduplicate_evidence_list(merged_ev_list, merged_diag_name)
                existing_match.confidence_score = max(existing_match.confidence_score, s.confidence_score)
            else:
                if s_code and s_code not in ("None", "[NO LOCAL CODE]"):
                    seen_sec_codes.add(s_code)
                if s_clean_norm:
                    seen_sec_names.add(s_clean_norm)
                # Clean evidence on this secondary
                s.evidence_quote = clean_and_deduplicate_evidence_quote(s.evidence_quote, s_name)
                s.evidence = clean_and_deduplicate_evidence_list(getattr(s, "evidence", []) or s.evidence_quote, s_name)
                deduped_secondaries.append(s)

        self.secondary_diagnoses = deduped_secondaries

        # 3. Drop conditions from abstentions that are ALREADY coded in primary or secondaries
        coded_names: set[str] = set()
        if self.primary_diagnosis:
            p_n = self.primary_diagnosis.normalized_diagnosis or self.primary_diagnosis.raw_term or self.primary_diagnosis.description or ""
            coded_names.add(p_n.lower().strip())
            coded_names.add((normalize_diagnosis_name(p_n) or p_n).lower().strip())
        for s in self.secondary_diagnoses:
            s_n = s.normalized_diagnosis or s.raw_term or s.description or ""
            coded_names.add(s_n.lower().strip())
            coded_names.add((normalize_diagnosis_name(s_n) or s_n).lower().strip())

        deduped_abstentions = []
        seen_abst_terms: set[str] = set()
        for abst in (self.abstentions or []):
            a_term = (abst.raw_term or "").strip()
            if not a_term:
                continue
            a_norm = a_term.lower().strip()
            a_clean = (normalize_diagnosis_name(a_term) or a_term).lower().strip()
            if a_norm in coded_names or a_clean in coded_names:
                continue
            if a_clean in seen_abst_terms or a_norm in seen_abst_terms:
                continue
            seen_abst_terms.add(a_clean)
            seen_abst_terms.add(a_norm)
            deduped_abstentions.append(abst)
        self.abstentions = deduped_abstentions

        # 4. Deduplicate procedures
        deduped_procs: list[str] = []
        seen_procs: set[str] = set()
        for proc in (self.procedures or []):
            p_str = proc.strip()
            if not p_str:
                continue
            clean_p = re.sub(r"\[.*?\]", "", p_str).lower().strip()
            clean_p = re.sub(r"\s+", " ", clean_p)
            if clean_p in seen_procs:
                continue
            seen_procs.add(clean_p)
            deduped_procs.append(p_str)
        self.procedures = deduped_procs

        return self

    def format_output(self) -> str:
        """Format strictly according to required clinical coding output:
        primary diagnosis: <name>, code: <code>, evidence: <evidence quote>
        secondary diagnosises:
        diagnosis: <name>, code: <code>, evidence: <evidence quote>
        """
        from medical_coding.utils.text import clean_and_deduplicate_evidence_quote

        def _get_cand_ev(cand: Any) -> str:
            raw_ev = getattr(cand, "evidence_quote", None)
            if not raw_ev:
                ev_list = getattr(cand, "evidence", []) or []
                if isinstance(ev_list, list):
                    quotes = []
                    for item in ev_list:
                        if isinstance(item, dict):
                            quotes.append(str(item.get("quote", "")))
                        elif hasattr(item, "quote"):
                            quotes.append(str(getattr(item, "quote", "")))
                        elif isinstance(item, str):
                            quotes.append(item)
                    raw_ev = " | ".join(q for q in quotes if q)
            diag_name = getattr(cand, "normalized_diagnosis", None) or getattr(cand, "raw_term", None) or getattr(cand, "description", None)
            return clean_and_deduplicate_evidence_quote(raw_ev, diag_name)

        def _get_formatted_code(cand: Any) -> str:
            base_code = getattr(cand, "icd10cm", None) or cand.code or ""
            extra_codes: list[str] = []
            if getattr(cand, "icdo", None) and cand.icdo != base_code:
                extra_codes.append(f"ICD-O: {cand.icdo}")
            if getattr(cand, "cpt", None) and cand.cpt != base_code:
                extra_codes.append(f"CPT: {cand.cpt}")
            if base_code and extra_codes:
                return f"{base_code} ({', '.join(extra_codes)})"
            elif extra_codes:
                return ", ".join(extra_codes)
            elif base_code:
                return base_code
            return "None"

        lines = []
        if self.primary_diagnosis and self.primary_diagnosis.code and self.primary_diagnosis.code != "None" and self.primary_diagnosis.matching_status != "NO_DATABASE_MATCH":
            p = self.primary_diagnosis
            p_name = p.normalized_diagnosis or p.raw_term or p.description
            p_code = _get_formatted_code(p)
            p_ev = _get_cand_ev(p)
            lines.append(f"primary diagnosis: {p_name}, code: {p_code}, evidence: {p_ev}")
        else:
            lines.append("primary diagnosis: None, code: None, evidence: None")

        # Deduplicate coded secondaries for printing
        seen_codes: set[str] = set()
        seen_names: set[str] = set()
        coded_secondaries = []
        for s in (self.secondary_diagnoses or []):
            if not s.code or s.code == "None" or s.matching_status == "NO_DATABASE_MATCH":
                continue
            code_key = s.code.strip()
            name_key = (s.normalized_diagnosis or s.raw_term or s.description or "").lower().strip()
            if code_key in seen_codes or (name_key and name_key in seen_names):
                continue
            seen_codes.add(code_key)
            if name_key:
                seen_names.add(name_key)
            coded_secondaries.append(s)

        if coded_secondaries:
            lines.append("secondary diagnosises:")
            for s in coded_secondaries:
                s_name = s.normalized_diagnosis or s.raw_term or s.description
                s_code = _get_formatted_code(s)
                s_ev = _get_cand_ev(s)
                lines.append(f"diagnosis: {s_name}, code: {s_code}, evidence: {s_ev}")
        else:
            lines.append("secondary diagnosises: None")

        return "\n".join(lines)


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
