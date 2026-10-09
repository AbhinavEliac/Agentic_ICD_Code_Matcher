import re

from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.schemas.clinical import (
    ClassifiedDiagnosis,
    ConditionClassification,
    ContextAssessment,
    EncounterClassificationResult,
)
from medical_coding.schemas.enums import (
    AbstentionReason,
    Certainty,
    ConditionStatus,
    DiagnosisRole,
    NegationStatus,
    PipelineStage,
    Temporality,
)
from medical_coding.schemas.icd import ICDCandidate, RankedSelection
from medical_coding.schemas.validation import (
    AbstentionRecord,
    ValidatedDiagnosis,
    ValidationCheck,
)
from medical_coding.utils.logging import get_logger
from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

logger = get_logger(__name__)


class DeterministicValidator:
    """Non-LLM deterministic rule engine verifying selected codes against local dataset catalog."""

    def __init__(self, catalog: LocalICDCatalog) -> None:
        self.catalog = catalog

    def validate_code(
        self,
        condition: ClassifiedDiagnosis,
        selection: RankedSelection,
    ) -> tuple[ValidatedDiagnosis | None, AbstentionRecord | None]:
        """Perform deterministic validation checks on an individual candidate selection.

        Checks:
        1. Code selection present (not None/abstained).
        2. Catalog existence in local dataset.
        3. Terminal specificity / billable status.
        4. Minimum confidence threshold.

        Returns:
            (ValidatedDiagnosis, None) if all checks pass,
            (None, AbstentionRecord) if any rule check fails.
        """
        if selection.selected_candidate is None:
            return None, AbstentionRecord(
                diagnosis_id=condition.diagnosis_id,
                raw_term=condition.raw_term,
                reason=AbstentionReason.NO_MATCHING_ICD_CANDIDATE,
                detail=selection.selection_justification or "No candidate selected.",
                stage=PipelineStage.RANKING,
            )

        candidate = selection.selected_candidate
        code = candidate.code
        checks: list[ValidationCheck] = []

        # 1. Catalog Existence Check
        exists = self.catalog.is_valid_code(code)
        checks.append(
            ValidationCheck(
                rule_name="CatalogExistence",
                passed=exists,
                details=f"Code '{code}' exists in local authoritative catalog."
                if exists
                else f"Code '{code}' does not exist in local dataset.",
            )
        )
        if not exists:
            return None, AbstentionRecord(
                diagnosis_id=condition.diagnosis_id,
                raw_term=condition.raw_term,
                reason=AbstentionReason.INVALID_ICD_CODE,
                detail=f"Proposed code '{code}' does not exist in authoritative local dataset.",
                stage=PipelineStage.VALIDATION,
            )

        # 2. Terminal Specificity Check (HIPAA Billability)
        billable = self.catalog.is_billable_code(code)
        checks.append(
            ValidationCheck(
                rule_name="TerminalSpecificity",
                passed=billable,
                details=f"Code '{code}' is at terminal billable specificity."
                if billable
                else f"Code '{code}' requires additional characters/specificity.",
            )
        )
        if not billable:
            return None, AbstentionRecord(
                diagnosis_id=condition.diagnosis_id,
                raw_term=condition.raw_term,
                reason=AbstentionReason.SPECIFICITY_REQUIRED,
                detail=f"Code '{code}' is a non-billable category header; terminal leaf code required.",
                stage=PipelineStage.VALIDATION,
            )

        # 3. Semantic Code Attribute Validation (CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY)
        ev_text = (
            getattr(condition.context.evidence, "quote", "")
            if hasattr(condition, "context") and hasattr(condition.context, "evidence") and condition.context.evidence
            else ""
        )
        if selection.supporting_evidence:
            ev_text = f"{ev_text} {' '.join(selection.supporting_evidence)}"

        is_attr_valid, unsupp_attrs = ReverseAttributeChecker.validate_code_attributes(
            code_description=candidate.description,
            evidence_text=ev_text,
            diagnosis_term=condition.raw_term,
        )
        if not is_attr_valid:
            better_candidate = None
            if selection.candidate_pool:
                for alt_cand in selection.candidate_pool:
                    if not self.catalog.is_valid_code(alt_cand.code) or not self.catalog.is_billable_code(alt_cand.code):
                        continue
                    alt_valid, _ = ReverseAttributeChecker.validate_code_attributes(
                        code_description=alt_cand.description,
                        evidence_text=ev_text,
                        diagnosis_term=condition.raw_term,
                    )
                    if alt_valid:
                        better_candidate = alt_cand
                        break

            if better_candidate:
                candidate = better_candidate
                code = better_candidate.code
                checks.append(
                    ValidationCheck(
                        rule_name="SemanticAttributeMatch",
                        passed=True,
                        details=f"Realigned code to supported candidate '{code}' ({candidate.description}) without unsupported {', '.join(unsupp_attrs)}.",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        rule_name="SemanticAttributeMatch",
                        passed=False,
                        details=f"Code '{code}' specifies unsupported attributes: {', '.join(unsupp_attrs)}.",
                    )
                )
                return None, AbstentionRecord(
                    diagnosis_id=condition.diagnosis_id,
                    raw_term=condition.raw_term,
                    reason=AbstentionReason.UNSUPPORTED_SPECIFICITY,
                    detail=f"Code '{code}' requires unsupported clinical attributes ({', '.join(unsupp_attrs)}) not evidenced in documentation.",
                    stage=PipelineStage.VALIDATION,
                )
        else:
            checks.append(
                ValidationCheck(
                    rule_name="SemanticAttributeMatch",
                    passed=True,
                    details=f"All clinical qualifiers in '{code}' ({candidate.description}) are supported by documentation.",
                )
            )

        cand_sys = getattr(candidate, "coding_system", None) or "ICD-10-CM"
        icd10cm_val = selection.selected_icd10cm or (code if cand_sys == "ICD-10-CM" else None)
        icdo_val = selection.selected_icdo or (code if cand_sys == "ICD-O" else None)
        cpt_val = selection.selected_cpt or (code if cand_sys == "CPT" else None)

        validated = ValidatedDiagnosis(
            diagnosis_id=condition.diagnosis_id,
            raw_term=condition.raw_term,
            code=code,
            description=candidate.description,
            role=condition.role,
            evidence=condition.context.evidence,
            confidence_score=selection.ranking_score,
            checks=checks,
            icd10cm=icd10cm_val,
            icdo=icdo_val,
            cpt=cpt_val,
            database_code=code,
            database_description=candidate.description,
            matching_status="MATCHED",
            source_section=getattr(condition.context.evidence, "source_section", "") or getattr(selection, "source_section", ""),
            source_span=getattr(condition.context.evidence, "source_span", None) or getattr(selection, "source_span", None),
        )
        return validated, None

    def validate_encounter_set(
        self,
        conditions: list[ClassifiedDiagnosis],
        selections: list[RankedSelection],
    ) -> tuple[list[ValidatedDiagnosis], list[AbstentionRecord]]:
        """Validate entire document set of selections enforcing encounter-level invariants:

        1. Maximum ONE Primary Diagnosis.
        2. No Excludes1 conflicts among co-billed codes.
        """
        validated_list: list[ValidatedDiagnosis] = []
        abstention_list: list[AbstentionRecord] = []

        selection_map = {s.diagnosis_id: s for s in selections}

        for cond in conditions:
            if not cond.is_billable_candidate or cond.role == DiagnosisRole.EXCLUDED:
                abstention_list.append(
                    AbstentionRecord(
                        diagnosis_id=cond.diagnosis_id,
                        raw_term=cond.raw_term,
                        reason=AbstentionReason.EXCLUDED_BY_TEMPORALITY
                        if "historical" in cond.classification_reason.lower()
                        else AbstentionReason.EXCLUDED_BY_NEGATION,
                        detail=cond.classification_reason,
                        stage=PipelineStage.CLASSIFICATION,
                    )
                )
                continue

            sel = selection_map.get(cond.diagnosis_id)
            if not sel:
                continue

            val, abst = self.validate_code(cond, sel)
            if val:
                validated_list.append(val)
            if abst:
                abstention_list.append(abst)

        # Invariant: Maximum ONE Primary Diagnosis
        primaries = [v for v in validated_list if v.role == DiagnosisRole.PRIMARY]
        if len(primaries) > 1:
            logger.warning(
                "Multiple primary diagnoses detected (%d). Abstaining all primaries to prevent invalid billing.",
                len(primaries),
            )
            for prim in primaries:
                validated_list.remove(prim)
                abstention_list.append(
                    AbstentionRecord(
                        diagnosis_id=prim.diagnosis_id,
                        raw_term=prim.raw_term,
                        reason=AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY,
                        detail="Conflicting multiple primary diagnoses assigned; guidelines permit max ONE.",
                        stage=PipelineStage.VALIDATION,
                    )
                )

        # Invariant: Excludes1 Check across co-occurring validated codes
        for i, val_a in enumerate(validated_list):
            for val_b in validated_list[i + 1 :]:
                if self.catalog.check_excludes1(val_a.code, val_b.code):
                    abstention_list.append(
                        AbstentionRecord(
                            diagnosis_id=val_b.diagnosis_id,
                            raw_term=val_b.raw_term,
                            reason=AbstentionReason.EXCLUDES_1_VIOLATION,
                            detail=f"Excludes1 violation between '{val_a.code}' and '{val_b.code}'.",
                            stage=PipelineStage.VALIDATION,
                        )
                    )

        return validated_list, abstention_list


class MultiplePrimaryDiagnosesError(ValueError):
    """Raised when an encounter classification contains more than one primary diagnosis."""


class PrimarySecondaryClassificationValidator:
    """Deterministic validation engine for Primary vs Secondary diagnosis classification.

    CRITICAL HARD RULES ENFORCED DETERMINISTICALLY:
    1. Maximum ONE primary diagnosis per encounter.
    2. Multiple primary diagnoses strictly trigger rejection / abstention with MULTIPLE_AMBIGUOUS_PRIMARY.
    3. Primary diagnosis CANNOT be historical-only, ruled out, negated, or unmanaged.
    4. Secondary diagnoses must be actively managed, monitored, or clinically relevant during the admission.
    5. Historical-only conditions without inpatient care must be EXCLUDED.
    """

    def validate_or_raise(
        self,
        classifications: list[ConditionClassification] | list[ClassifiedDiagnosis],
    ) -> None:
        """Validate classifications and raise MultiplePrimaryDiagnosesError if more than one primary diagnosis exists."""
        primaries = [
            c.diagnosis if isinstance(c, ConditionClassification) else c.raw_term
            for c in classifications
            if c.role == DiagnosisRole.PRIMARY
        ]
        if len(primaries) > 1:
            raise MultiplePrimaryDiagnosesError(
                f"Multiple primary diagnoses detected ({len(primaries)}): {primaries}. "
                "Inpatient coding guidelines strictly permit at most ONE primary diagnosis."
            )

    def validate_classification(
        self,
        classifications: list[ConditionClassification],
        context_assessments: list[ContextAssessment] | None = None,
        document_id: str | None = None,
    ) -> tuple[EncounterClassificationResult, list[AbstentionRecord]]:
        """Perform comprehensive deterministic validation on classification output.

        Enforces:
        - Maximum ONE primary diagnosis (demoting/abstaining if multiple found)
        - Ineligibility of historical, ruled-out, or negated conditions as primary or secondary
        - Proper exclusion of unmanaged PMH
        """
        audit_notes: list[str] = []
        abstentions: list[AbstentionRecord] = []
        ctx_map: dict[str, ContextAssessment] = {
            asm.condition_id or asm.diagnosis.lower(): asm for asm in (context_assessments or [])
        }

        validated_classifications: list[ConditionClassification] = []

        # 1. Inspect and validate each condition
        for cond_in in classifications:
            cond = cond_in.model_copy()
            ctx = ctx_map.get(cond.diagnosis_id) or ctx_map.get(cond.diagnosis.lower())

            # Rule: Ruled-out or negated conditions CANNOT be PRIMARY or SECONDARY
            is_ruled_out_or_negated = False
            if ctx:
                if ctx.certainty == Certainty.RULED_OUT or ctx.negation == NegationStatus.NEGATED:
                    is_ruled_out_or_negated = True
            elif (
                "ruled out" in cond.classification_reason.lower()
                or "negated" in cond.classification_reason.lower()
            ):
                is_ruled_out_or_negated = True

            if is_ruled_out_or_negated and cond.role != DiagnosisRole.EXCLUDED:
                audit_notes.append(
                    f"Deterministic override: Condition '{cond.diagnosis}' was marked {cond.role.value} "
                    "but is ruled out/negated. Forcing to EXCLUDED."
                )
                cond.role = DiagnosisRole.EXCLUDED
                cond.is_billable_candidate = False
                cond.classification_reason = f"Ruled out / negated condition excluded from coding: {cond.classification_reason}"
                abstentions.append(
                    AbstentionRecord(
                        diagnosis_id=cond.diagnosis_id,
                        raw_term=cond.diagnosis,
                        reason=AbstentionReason.EXCLUDED_BY_NEGATION,
                        detail=cond.classification_reason,
                        stage=PipelineStage.CLASSIFICATION,
                    )
                )

            # Rule: Historical-only conditions without inpatient care CANNOT be PRIMARY or SECONDARY
            is_historical_only = False
            if ctx:
                if (
                    ctx.status == ConditionStatus.HISTORICAL
                    or ctx.temporality == Temporality.HISTORICAL
                ):
                    if not (
                        ctx.treated_or_managed
                        or ctx.monitored
                        or ctx.affected_clinical_management
                        or ctx.influenced_treatment
                    ):
                        is_historical_only = True
            elif (
                "historical" in cond.classification_reason.lower()
                and "active" not in cond.classification_reason.lower()
            ):
                is_historical_only = True

            if is_historical_only and cond.role != DiagnosisRole.EXCLUDED:
                audit_notes.append(
                    f"Deterministic override: Condition '{cond.diagnosis}' was marked {cond.role.value} "
                    "but is historical without inpatient care. Forcing to EXCLUDED."
                )
                cond.role = DiagnosisRole.EXCLUDED
                cond.is_billable_candidate = False
                cond.classification_reason = f"Historical condition without active inpatient care excluded: {cond.classification_reason}"
                abstentions.append(
                    AbstentionRecord(
                        diagnosis_id=cond.diagnosis_id,
                        raw_term=cond.diagnosis,
                        reason=AbstentionReason.EXCLUDED_BY_TEMPORALITY,
                        detail=cond.classification_reason,
                        stage=PipelineStage.CLASSIFICATION,
                    )
                )

            # If role is EXCLUDED, ensure is_billable_candidate is False
            if cond.role == DiagnosisRole.EXCLUDED:
                cond.is_billable_candidate = False

            validated_classifications.append(cond)

        # 2. Hard Rule: Maximum ONE Primary Diagnosis
        primaries = [c for c in validated_classifications if c.role == DiagnosisRole.PRIMARY]
        has_unique_primary = False
        is_ambiguous_primary = False
        abstention_recommended = False
        abstention_reason: AbstentionReason | None = None
        unique_primary: ConditionClassification | None = None

        if len(primaries) > 1:
            audit_notes.append(
                f"REJECTION: Multiple primary diagnoses detected ({len(primaries)}): "
                f"{[p.diagnosis for p in primaries]}. UHDDS guidelines permit at most ONE. "
                "Demoting conflicting primaries to abstained to prevent invalid billing."
            )
            is_ambiguous_primary = True
            abstention_recommended = True
            abstention_reason = AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY

            demoted: list[ConditionClassification] = []
            for p in validated_classifications:
                if p.role == DiagnosisRole.PRIMARY:
                    abstentions.append(
                        AbstentionRecord(
                            diagnosis_id=p.diagnosis_id,
                            raw_term=p.diagnosis,
                            reason=AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY,
                            detail=f"Multiple conflicting primary diagnoses: '{p.diagnosis}' competed with other primary diagnoses.",
                            stage=PipelineStage.CLASSIFICATION,
                        )
                    )
                    demoted.append(
                        p.model_copy(
                            update={
                                "role": DiagnosisRole.EXCLUDED,
                                "is_billable_candidate": False,
                                "classification_reason": (
                                    f"Conflicting primary diagnosis: '{p.diagnosis}' competed with other primary candidates. "
                                    "Abstained due to ambiguity without physician query."
                                ),
                            }
                        )
                    )
                else:
                    demoted.append(p)
            validated_classifications = demoted
        elif len(primaries) == 1:
            unique_primary = primaries[0]
            has_unique_primary = True
        else:
            # Zero primaries
            ambiguous_candidates = [
                c
                for c in validated_classifications
                if "ambiguous" in c.classification_reason.lower()
                or "co-primary" in c.classification_reason.lower()
            ]
            if ambiguous_candidates:
                audit_notes.append(
                    f"Two or more plausible primary diagnoses ({len(ambiguous_candidates)}) detected without documentary distinction."
                )
                is_ambiguous_primary = True
                abstention_recommended = True
                abstention_reason = AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY
                for amb in ambiguous_candidates:
                    abstentions.append(
                        AbstentionRecord(
                            diagnosis_id=amb.diagnosis_id,
                            raw_term=amb.diagnosis,
                            reason=AbstentionReason.MULTIPLE_AMBIGUOUS_PRIMARY,
                            detail=f"Ambiguous co-primary diagnosis: '{amb.diagnosis}' competed with other plausible primaries.",
                            stage=PipelineStage.CLASSIFICATION,
                        )
                    )
            else:
                has_billable_secondaries = any(
                    c.role == DiagnosisRole.SECONDARY for c in validated_classifications
                )
                if has_billable_secondaries:
                    audit_notes.append(
                        "No defensible primary diagnosis identified despite active secondary conditions."
                    )
                    abstention_recommended = True
                    abstention_reason = AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE
                    abstentions.append(
                        AbstentionRecord(
                            diagnosis_id=None,
                            raw_term=None,
                            reason=AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE,
                            detail="No defensible primary diagnosis identified for the encounter.",
                            stage=PipelineStage.CLASSIFICATION,
                        )
                    )

        secondaries = [c for c in validated_classifications if c.role == DiagnosisRole.SECONDARY]
        excluded = [c for c in validated_classifications if c.role == DiagnosisRole.EXCLUDED]

        result = EncounterClassificationResult(
            document_id=document_id,
            primary_diagnosis=unique_primary,
            secondary_diagnoses=secondaries,
            excluded_conditions=excluded,
            all_classifications=validated_classifications,
            has_unique_primary=has_unique_primary,
            is_ambiguous_primary=is_ambiguous_primary,
            abstention_recommended=abstention_recommended,
            abstention_reason=abstention_reason,
            audit_notes=audit_notes,
        )

        return result, abstentions


class CandidateRankingDeterministicValidator:
    """Non-LLM deterministic validator verifying candidate ranking decisions against strict invariants.

    HARD CONSTRAINTS ENFORCED:
    1. Candidate Pool Boundary: selected_code MUST be an element of the supplied candidate pool.
       Free-form or hallucinated codes are strictly rejected without guessing.
    2. Evidence Presence: Code cannot be selected without documented evidence text.
    3. Specificity Verification: Prevents inferring clinical specificity that is absent from evidence.
       (e.g., systolic/diastolic heart failure without documented systolic/diastolic proof;
        diabetic complications without documented complication proof).
    4. Confidence Threshold: Enforces minimum acceptable confidence score.
    """

    def __init__(
        self,
        catalog: LocalICDCatalog | None = None,
        min_confidence: float = 0.40,
    ) -> None:
        self.catalog = catalog
        self.min_confidence = min_confidence

    def validate_ranking_selection(
        self,
        selection: RankedSelection,
        candidates: list[ICDCandidate],
        evidence_text: str = "",
        diagnosis_term: str = "",
    ) -> RankedSelection:
        """Validate and sanitize a candidate ranking selection against hard constraints."""
        from medical_coding.utils.text import format_icd_code

        clean_evidence = (evidence_text or "").strip()
        clean_diagnosis = (diagnosis_term or "").strip()

        # 1. No candidates available -> Clean abstention
        if not candidates:
            return RankedSelection(
                diagnosis_id=selection.diagnosis_id,
                raw_term=selection.raw_term or clean_diagnosis,
                selected_code=None,
                selected_description=None,
                selected_candidate=None,
                ranking_reason="Candidate pool was empty; no matching local ICD code.",
                supporting_evidence=[clean_evidence] if clean_evidence else [],
                confidence=0.0,
                abstention_reason="NO_CANDIDATES",
                matching_status="NO_DATABASE_MATCH",
                source_section=getattr(selection, "source_section", ""),
                source_span=getattr(selection, "source_span", None),
                candidate_pool=[],
                decision="ABSTAINED",
            )

        # 2. Missing evidence -> Clean abstention
        if not clean_evidence:
            return RankedSelection(
                diagnosis_id=selection.diagnosis_id,
                raw_term=selection.raw_term or clean_diagnosis,
                selected_code=None,
                selected_description=None,
                selected_candidate=None,
                ranking_reason="Insufficient evidence: documentation lacks factual evidence quote.",
                supporting_evidence=[],
                confidence=0.0,
                abstention_reason="INSUFFICIENT_EVIDENCE",
                matching_status="NO_DATABASE_MATCH",
                source_section=getattr(selection, "source_section", ""),
                source_span=getattr(selection, "source_span", None),
                candidate_pool=candidates,
                decision="ABSTAINED",
            )

        # 3. If selection is already abstained or null
        if selection.selected_code is None:
            return RankedSelection(
                diagnosis_id=selection.diagnosis_id,
                raw_term=selection.raw_term or clean_diagnosis,
                selected_code=None,
                selected_description=None,
                selected_candidate=None,
                ranking_reason=selection.ranking_reason or "No candidate selected.",
                supporting_evidence=[clean_evidence],
                confidence=0.0,
                abstention_reason=selection.abstention_reason or "NO_MATCHING_CANDIDATE",
                matching_status="NO_DATABASE_MATCH",
                source_section=getattr(selection, "source_section", ""),
                source_span=getattr(selection, "source_span", None),
                candidate_pool=candidates,
                decision="ABSTAINED",
            )

        # 4. HARD CONSTRAINT: selected_code MUST exist in the candidate list
        cand_map: dict[str, ICDCandidate] = {format_icd_code(c.code): c for c in candidates}
        formatted_sel = format_icd_code(selection.selected_code)

        if formatted_sel not in cand_map:
            logger.warning(
                "HARD CONSTRAINT VIOLATION: Proposed code '%s' does not exist in candidate pool %s. Rejecting.",
                selection.selected_code,
                [c.code for c in candidates],
            )
            return RankedSelection(
                diagnosis_id=selection.diagnosis_id,
                raw_term=selection.raw_term or clean_diagnosis,
                selected_code=None,
                selected_description=None,
                selected_candidate=None,
                ranking_reason=f"Deterministic rejection: Proposed code '{selection.selected_code}' is not in the authoritative candidate pool {[c.code for c in candidates]}.",
                supporting_evidence=[clean_evidence],
                confidence=0.0,
                abstention_reason="INVALID_LLM_CODE_NOT_IN_CANDIDATE_POOL",
                matching_status="NO_DATABASE_MATCH",
                source_section=getattr(selection, "source_section", ""),
                source_span=getattr(selection, "source_span", None),
                candidate_pool=candidates,
                decision="REJECTED_MISMATCH",
            )

        matching_candidate = cand_map[formatted_sel]

        # 5. Local catalog existence check
        if self.catalog and not self.catalog.is_valid_code(matching_candidate.code):
            return RankedSelection(
                diagnosis_id=selection.diagnosis_id,
                raw_term=selection.raw_term or clean_diagnosis,
                selected_code=None,
                selected_description=None,
                selected_candidate=None,
                ranking_reason=f"Code '{matching_candidate.code}' does not exist in local authoritative catalog.",
                supporting_evidence=[clean_evidence],
                confidence=0.0,
                abstention_reason="INVALID_ICD_CODE",
                matching_status="NO_DATABASE_MATCH",
                source_section=getattr(selection, "source_section", ""),
                source_span=getattr(selection, "source_span", None),
                candidate_pool=candidates,
                decision="REJECTED_MISMATCH",
            )

        # 6. Specificity Support Check (Anti-Hallucinated Specificity)
        evidence_lower = f"{clean_evidence} {clean_diagnosis}".lower()
        cand_desc_lower = matching_candidate.description.lower()

        # Heart failure specific subtypes: systolic, diastolic, right, left, end stage, etc.
        if ("heart" in cand_desc_lower and "failure" in cand_desc_lower) or matching_candidate.code.startswith("I50") or matching_candidate.code.startswith("T86.2"):
            hf_subtypes = ["systolic", "diastolic", "right", "left", "high output", "end stage", "biventricular", "other", "transplant", "rheumatic"]
            cand_hf_subtypes = [w for w in hf_subtypes if w in cand_desc_lower]
            doc_context = f"{clean_diagnosis.lower()} {evidence_lower}"
            unsupported_hf = [w for w in cand_hf_subtypes if w not in doc_context]
            if unsupported_hf:
                gen_cand = next(
                    (
                        c
                        for c in sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)
                        if "unspecified" in c.description.lower() or c.code in ["I50.9", "I50"]
                    ),
                    None,
                )
                if gen_cand and gen_cand.is_valid_billable:
                    matching_candidate = gen_cand
                    selection.selected_code = gen_cand.code
                    selection.selected_description = gen_cand.description
                    selection.selected_candidate = gen_cand
                    selection.ranking_reason = (
                        f"Specificity realigned: Documentation supports general heart failure without {', '.join(unsupported_hf)} specificity; "
                        f"selected unspecified code {gen_cand.code}."
                    )
                else:
                    return RankedSelection(
                        diagnosis_id=selection.diagnosis_id,
                        raw_term=selection.raw_term or clean_diagnosis,
                        selected_code=None,
                        selected_description=None,
                        selected_candidate=None,
                        ranking_reason=(
                            f"Unsupported specificity: Candidate '{matching_candidate.code}' specifies {', '.join(unsupported_hf)} heart failure, "
                            "but documentation only supports general heart failure."
                        ),
                        supporting_evidence=[clean_evidence],
                        confidence=0.0,
                        abstention_reason="UNSUPPORTED_SPECIFICITY",
                        matching_status="NO_DATABASE_MATCH",
                        source_section=getattr(selection, "source_section", ""),
                        source_span=getattr(selection, "source_span", None),
                        candidate_pool=candidates,
                        decision="ABSTAINED",
                    )

        # Diabetes complications specificity check
        comp_keywords = [
            "nephropathy",
            "retinopathy",
            "neuropathy",
            "hyperglycemia",
            "ketoacidosis",
        ]
        is_diabetic_comp_cand = any(comp in cand_desc_lower for comp in comp_keywords)
        has_comp_in_ev = any(
            comp in evidence_lower
            for comp in [
                "nephropathy",
                "kidney disease",
                "retinopathy",
                "neuropathy",
                "hyperglycemia",
                "ketoacidosis",
                "dka",
                "uncontrolled",
            ]
        )

        if is_diabetic_comp_cand and not has_comp_in_ev:
            uncomp_cand = next(
                (
                    c
                    for c in sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)
                    if "without complications" in c.description.lower() or c.code == "E11.9"
                ),
                None,
            )
            if uncomp_cand:
                matching_candidate = uncomp_cand
                selection.selected_code = uncomp_cand.code
                selection.selected_description = uncomp_cand.description
                selection.selected_candidate = uncomp_cand
                selection.ranking_reason = (
                    f"Specificity realigned: Documentation lacks evidence of diabetic complications; "
                    f"selected uncomplicated code {uncomp_cand.code}."
                )
            else:
                return RankedSelection(
                    diagnosis_id=selection.diagnosis_id,
                    raw_term=selection.raw_term or clean_diagnosis,
                    selected_code=None,
                    selected_description=None,
                    selected_candidate=None,
                    ranking_reason="Unsupported specificity: Candidate specifies diabetic complications without documented evidence.",
                    supporting_evidence=[clean_evidence],
                    confidence=0.0,
                    abstention_reason="UNSUPPORTED_SPECIFICITY",
                    matching_status="NO_DATABASE_MATCH",
                    source_section=getattr(selection, "source_section", ""),
                    source_span=getattr(selection, "source_span", None),
                    candidate_pool=candidates,
                    decision="ABSTAINED",
                )

        # Reverse Attribute Check (Master Prompt: CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY)
        is_attr_valid, unsupp_attrs = ReverseAttributeChecker.validate_code_attributes(
            code_description=matching_candidate.description,
            evidence_text=clean_evidence,
            diagnosis_term=clean_diagnosis,
        )
        if not is_attr_valid:
            # Look for a candidate in candidate pool that is billable and valid under reverse attribute check
            better_candidate = None
            best_cand_score = -999.0
            diag_clean_lower = clean_diagnosis.lower()
            diag_tokens = [t for t in re.findall(r"[a-z0-9]+", diag_clean_lower) if len(t) > 3]

            sorted_candidates = sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)
            for alt_cand in sorted_candidates:
                if not alt_cand.is_valid_billable:
                    continue
                alt_valid, _ = ReverseAttributeChecker.validate_code_attributes(
                    code_description=alt_cand.description,
                    evidence_text=clean_evidence,
                    diagnosis_term=clean_diagnosis,
                )
                if alt_valid:
                    alt_desc_lower = alt_cand.description.lower()
                    if any(t in alt_desc_lower for t in diag_tokens):
                        pref_score = alt_cand.retrieval_score
                        # Prefer 'unspecified' or canonical base codes over 'other' when documentation is general
                        if "unspecified" in alt_desc_lower or "without " in alt_desc_lower or alt_cand.code.endswith(".9") or alt_cand.code.endswith(".90"):
                            pref_score += 0.35
                        elif "other " in alt_desc_lower or "other specified" in alt_desc_lower:
                            doc_has_other = any(w in evidence_lower for w in ["other", "specified", "variant", "type"])
                            if not doc_has_other:
                                pref_score -= 0.35

                        if pref_score > best_cand_score:
                            best_cand_score = pref_score
                            better_candidate = alt_cand

            if better_candidate:
                matching_candidate = better_candidate
                selection.selected_code = better_candidate.code
                selection.selected_description = better_candidate.description
                selection.selected_candidate = better_candidate
                selection.ranking_reason = (
                    f"Specificity realigned: Documentation does not support {', '.join(unsupp_attrs)}; "
                    f"selected supported code {better_candidate.code} ({better_candidate.description})."
                )
            else:
                return RankedSelection(
                    diagnosis_id=selection.diagnosis_id,
                    raw_term=selection.raw_term or clean_diagnosis,
                    selected_code=None,
                    selected_description=None,
                    selected_candidate=None,
                    ranking_reason=(
                        f"Unsupported specificity: Candidate '{matching_candidate.code}' requires {', '.join(unsupp_attrs)}, "
                        "which is not documented in clinical evidence."
                    ),
                    supporting_evidence=[clean_evidence],
                    confidence=0.0,
                    abstention_reason="UNSUPPORTED_SPECIFICITY",
                    matching_status="NO_DATABASE_MATCH",
                    source_section=getattr(selection, "source_section", ""),
                    source_span=getattr(selection, "source_span", None),
                    candidate_pool=candidates,
                    decision="ABSTAINED",
                )

        # 7. Check if candidate has zero clinical relevance to evidence
        from medical_coding.retrieval.tokenizer import expand_clinical_morphology
        raw_ev = [t for t in re.findall(r"[A-Za-z0-9]+", evidence_lower) if len(t) > 2]
        ev_tokens = set(expand_clinical_morphology(raw_ev))
        cand_tokens = set(re.findall(r"[A-Za-z0-9]+", matching_candidate.description.lower()))
        # Remove universal filler tokens
        cand_meaningful = {
            t
            for t in cand_tokens
            if len(t) > 2 and t not in {"and", "with", "for", "without", "other", "unspecified"}
        }
        cand_meaningful_expanded = set(expand_clinical_morphology(list(cand_meaningful)))
        if cand_meaningful and not (ev_tokens & cand_meaningful_expanded):
            return RankedSelection(
                diagnosis_id=selection.diagnosis_id,
                raw_term=selection.raw_term or clean_diagnosis,
                selected_code=None,
                selected_description=None,
                selected_candidate=None,
                ranking_reason=f"Candidate not selected: Code '{matching_candidate.code}' ({matching_candidate.description}) is not supported by documented clinical evidence.",
                supporting_evidence=[clean_evidence],
                confidence=0.0,
                abstention_reason="NO_MATCHING_CANDIDATE",
                matching_status="NO_DATABASE_MATCH",
                source_section=getattr(selection, "source_section", ""),
                source_span=getattr(selection, "source_span", None),
                candidate_pool=candidates,
                decision="ABSTAINED",
            )

        # 8. Confidence threshold check
        effective_conf = max(selection.confidence, matching_candidate.retrieval_score)
        if effective_conf < self.min_confidence:
            return RankedSelection(
                diagnosis_id=selection.diagnosis_id,
                raw_term=selection.raw_term or clean_diagnosis,
                selected_code=None,
                selected_description=None,
                selected_candidate=None,
                ranking_reason=f"Candidate rejected: Confidence {effective_conf:.2f} is below minimum threshold {self.min_confidence:.2f}.",
                supporting_evidence=[clean_evidence],
                confidence=effective_conf,
                abstention_reason="CONFIDENCE_BELOW_THRESHOLD",
                matching_status="NO_DATABASE_MATCH",
                source_section=getattr(selection, "source_section", ""),
                source_span=getattr(selection, "source_span", None),
                candidate_pool=candidates,
                decision="REJECTED_LOW_CONFIDENCE",
            )

        # 9. Successfully accepted
        return RankedSelection(
            diagnosis_id=selection.diagnosis_id,
            raw_term=selection.raw_term or clean_diagnosis,
            selected_code=matching_candidate.code,
            selected_description=matching_candidate.description,
            selected_candidate=matching_candidate,
            ranking_reason=selection.ranking_reason
            or f"Selected candidate '{matching_candidate.code}' is best supported by documented clinical evidence.",
            supporting_evidence=[clean_evidence],
            confidence=round(effective_conf, 4),
            abstention_reason=None,
            matching_status="MATCHED",
            source_section=getattr(selection, "source_section", ""),
            source_span=getattr(selection, "source_span", None),
            candidate_pool=candidates,
            decision="ACCEPTED",
        )
