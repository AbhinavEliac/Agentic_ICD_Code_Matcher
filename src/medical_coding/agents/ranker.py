"""Candidate ranking agent selecting strictly from retrieved local ICD-10-CM codes."""

import json
import re
from typing import Any

from medical_coding.agents.base import BaseAgent
from medical_coding.models.base import BaseLocalLLM
from medical_coding.prompts.ranking import (
    RANKING_SYSTEM_PROMPT,
    RANKING_USER_TEMPLATE,
)
from medical_coding.schemas.clinical import ClassifiedDiagnosis, ConditionClassification
from medical_coding.schemas.enums import Certainty, DiagnosisRole, Temporality
from medical_coding.schemas.icd import ICDCandidate, RankedSelection
from medical_coding.utils.logging import get_logger
from medical_coding.utils.text import format_icd_code
from medical_coding.validation.deterministic import (
    CandidateRankingDeterministicValidator,
)
from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

logger = get_logger(__name__)


class CandidateRankingAgent(BaseAgent):
    """Evaluates retrieved local ICD candidates against evidence to select best match or abstain.

    CORE ARCHITECTURAL INVARIANTS:
    1. The candidate ranking agent answers:
       "Which of these retrieved ICD codes is best supported by the documented clinical evidence?"
       It does NOT answer:
       "What ICD code do I know for this diagnosis?"
    2. Selected code MUST exist in the supplied candidate list.
    3. The LLM may NEVER generate or invent a new ICD code.
    4. Unsupported clinical specificity is blocked.
    """

    def __init__(
        self,
        llm: BaseLocalLLM | None = None,
        validator: CandidateRankingDeterministicValidator | None = None,
        min_confidence: float = 0.40,
        allow_rerun: bool = False,
    ) -> None:
        super().__init__(llm)  # type: ignore[arg-type]
        self.system_prompt = RANKING_SYSTEM_PROMPT
        self.user_template = RANKING_USER_TEMPLATE
        self.validator = validator or CandidateRankingDeterministicValidator(
            min_confidence=min_confidence
        )
        self.min_confidence = min_confidence
        self.allow_rerun = allow_rerun

    def rank_candidates(
        self,
        condition: ClassifiedDiagnosis | ConditionClassification | Any,
        candidates: list[ICDCandidate],
        evidence_override: str | None = None,
        role_override: str | None = None,
        allow_rerun: bool | None = None,
    ) -> RankedSelection:
        """Evaluate candidate codes and select the best match or explicitly abstain.

        Args:
            condition: Diagnosis mention, classified condition, or dict with clinical attributes.
            candidates: Top-K retrieved ICD-10-CM candidates from the local dataset.
            evidence_override: Optional explicit evidence quote override.
            role_override: Optional explicit role override.
            allow_rerun: Override for single retry on invalid code output.

        Returns:
            RankedSelection with selected_code, selected_description, ranking_reason,
            supporting_evidence, confidence, and abstention_reason.
        """
        # 1. Normalize and extract condition details
        diag_id, raw_term, evidence_quote, role, acuity, certainty, anatomy, laterality, temporality = (
            self._extract_condition_attributes(condition, evidence_override, role_override)
        )

        # 2. Short-circuit: Empty candidate pool -> Immediate Abstention (No LLM call)
        if not candidates:
            logger.info(
                "Candidate pool is empty for '%s'; abstaining immediately without LLM.", raw_term
            )
            return self.validator.validate_ranking_selection(
                selection=RankedSelection(
                    diagnosis_id=diag_id,
                    raw_term=raw_term,
                    selected_code=None,
                    selected_description=None,
                    ranking_reason="Candidate pool was empty; no matching local ICD code.",
                    supporting_evidence=[evidence_quote] if evidence_quote else [],
                    confidence=0.0,
                    abstention_reason="NO_CANDIDATES",
                    candidate_pool=[],
                    decision="ABSTAINED",
                ),
                candidates=[],
                evidence_text=evidence_quote,
                diagnosis_term=raw_term,
            )

        # 3. Short-circuit: Insufficient / missing evidence -> Immediate Abstention
        if not evidence_quote.strip():
            logger.info("Documentation lacks evidence quote for '%s'; abstaining.", raw_term)
            return self.validator.validate_ranking_selection(
                selection=RankedSelection(
                    diagnosis_id=diag_id,
                    raw_term=raw_term,
                    selected_code=None,
                    selected_description=None,
                    ranking_reason="Insufficient evidence: documentation lacks factual evidence quote.",
                    supporting_evidence=[],
                    confidence=0.0,
                    abstention_reason="INSUFFICIENT_EVIDENCE",
                    candidate_pool=candidates,
                    decision="ABSTAINED",
                ),
                candidates=candidates,
                evidence_text="",
                diagnosis_term=raw_term,
            )

        # 4. Short-circuit: Excluded or Historical condition -> Immediate Abstention
        if role == DiagnosisRole.EXCLUDED or role == "EXCLUDED":
            logger.info("Condition '%s' is marked as EXCLUDED; abstaining from coding.", raw_term)
            return self.validator.validate_ranking_selection(
                selection=RankedSelection(
                    diagnosis_id=diag_id,
                    raw_term=raw_term,
                    selected_code=None,
                    selected_description=None,
                    ranking_reason=f"Condition '{raw_term}' is excluded from billing/coding.",
                    supporting_evidence=[evidence_quote],
                    confidence=0.0,
                    abstention_reason="CONDITION_EXCLUDED",
                    matching_status="NO_DATABASE_MATCH",
                    candidate_pool=candidates,
                    decision="ABSTAINED",
                ),
                candidates=candidates,
                evidence_text=evidence_quote,
                diagnosis_term=raw_term,
            )

        # 4b. Short-circuit: Ruled-Out condition
        if certainty in (Certainty.RULED_OUT, "RULED_OUT") or "ruled out" in evidence_quote.lower():
            logger.info("Condition '%s' is RULED OUT in clinical documentation; abstaining.", raw_term)
            return self.validator.validate_ranking_selection(
                selection=RankedSelection(
                    diagnosis_id=diag_id,
                    raw_term=raw_term,
                    selected_code=None,
                    selected_description=None,
                    ranking_reason=f"Condition '{raw_term}' was ruled out in clinical documentation.",
                    supporting_evidence=[evidence_quote],
                    confidence=0.0,
                    abstention_reason="RULED_OUT_CONDITION",
                    matching_status="NO_DATABASE_MATCH",
                    candidate_pool=candidates,
                    decision="ABSTAINED",
                ),
                candidates=candidates,
                evidence_text=evidence_quote,
                diagnosis_term=raw_term,
            )

        # 4c. Short-circuit: Historical condition not managed during current encounter
        if temporality in (Temporality.HISTORICAL, "HISTORICAL") and role in (
            DiagnosisRole.HISTORICAL,
            "HISTORICAL",
            DiagnosisRole.EXCLUDED,
            "EXCLUDED",
        ):
            logger.info("Condition '%s' is historical and unmanaged; abstaining from coding.", raw_term)
            return self.validator.validate_ranking_selection(
                selection=RankedSelection(
                    diagnosis_id=diag_id,
                    raw_term=raw_term,
                    selected_code=None,
                    selected_description=None,
                    ranking_reason=f"Condition '{raw_term}' is historical and not managed during stay.",
                    supporting_evidence=[evidence_quote],
                    confidence=0.0,
                    abstention_reason="EXCLUDED_BY_TEMPORALITY",
                    matching_status="NO_DATABASE_MATCH",
                    candidate_pool=candidates,
                    decision="ABSTAINED",
                ),
                candidates=candidates,
                evidence_text=evidence_quote,
                diagnosis_term=raw_term,
            )

        # 5. Format candidates into bounded prompt context
        candidates_formatted = self._format_candidates(candidates)
        prompt = self._build_prompt(
            diagnosis_term=raw_term,
            evidence_quote=evidence_quote,
            role=str(role),
            acuity=str(acuity),
            certainty=str(certainty),
            anatomical_site=str(anatomy or "Not documented"),
            laterality=str(laterality or "Unspecified"),
            candidates_formatted=candidates_formatted,
        )

        # 6. Execute LLM inference or deterministic evaluation
        initial_selection: RankedSelection | None = None
        if self.llm is not None:
            initial_selection = self._execute_llm_ranking(
                prompt, diag_id, raw_term, evidence_quote, candidates
            )

        # Fallback to deterministic scoring if LLM was unavailable or produced unparseable output
        if initial_selection is None:
            initial_selection = self._evaluate_candidates_deterministically(
                diag_id=diag_id,
                raw_term=raw_term,
                evidence_quote=evidence_quote,
                acuity=acuity,
                candidates=candidates,
            )

        # 7. Check Hard Constraint: Is proposed code in candidates list?
        rerun_flag = self.allow_rerun if allow_rerun is None else allow_rerun
        if (
            initial_selection.selected_code is not None
            and not any(
                format_icd_code(initial_selection.selected_code) == format_icd_code(c.code)
                for c in candidates
            )
            and rerun_flag
            and self.llm is not None
        ):
            # One retry with explicit error feedback
            logger.warning(
                "LLM proposed invalid code '%s' not in candidates. Triggering single rerun with bounded instruction.",
                initial_selection.selected_code,
            )
            rerun_prompt = (
                f"{prompt}\n\n"
                f"CORRECTION ERROR: Your previous selection '{initial_selection.selected_code}' is NOT in the candidate list.\n"
                f"You are strictly forbidden from inventing codes. Choose ONLY from: {[c.code for c in candidates]} or output null.\n"
                f"Output valid JSON only:"
            )
            initial_selection = (
                self._execute_llm_ranking(
                    rerun_prompt, diag_id, raw_term, evidence_quote, candidates
                )
                or initial_selection
            )

        # 8. Deterministic Validation immediately after LLM
        validated_selection = self.validator.validate_ranking_selection(
            selection=initial_selection,
            candidates=candidates,
            evidence_text=evidence_quote,
            diagnosis_term=raw_term,
        )

        return validated_selection

    def run(self, **kwargs: Any) -> RankedSelection:
        """Standard pipeline invocation interface."""
        condition = kwargs.get("condition")
        candidates = kwargs.get("candidates", [])
        evidence_override = kwargs.get("evidence")
        role_override = kwargs.get("role")
        return self.rank_candidates(
            condition=condition,
            candidates=candidates,
            evidence_override=evidence_override,
            role_override=role_override,
        )

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------

    def _extract_condition_attributes(
        self,
        condition: Any,
        evidence_override: str | None,
        role_override: str | None,
    ) -> tuple[str, str, str, str, str, str, str | None, str | None]:
        """Extract and normalize clinical attributes across various diagnosis representations."""
        diag_id = "diag-unknown"
        raw_term = "Unknown Diagnosis"
        evidence_quote = ""
        role = "SECONDARY"
        acuity = "UNSPECIFIED"
        certainty = "CONFIRMED"
        anatomy = None
        laterality = "UNSPECIFIED"
        temporality = "CURRENT"

        if hasattr(condition, "diagnosis_id"):
            diag_id = str(condition.diagnosis_id)
        if hasattr(condition, "raw_term"):
            raw_term = str(condition.raw_term)
        elif hasattr(condition, "diagnosis"):
            raw_term = str(condition.diagnosis)

        # Evidence resolution
        if evidence_override:
            evidence_quote = evidence_override
        elif hasattr(condition, "evidence_quote") and condition.evidence_quote:
            evidence_quote = condition.evidence_quote
        elif hasattr(condition, "evidence_text") and condition.evidence_text:
            evidence_quote = condition.evidence_text
        elif hasattr(condition, "context") and hasattr(condition.context, "evidence"):
            evidence_quote = getattr(condition.context.evidence, "quote", "")

        # Role resolution
        if role_override:
            role = role_override
        elif hasattr(condition, "role"):
            role = str(condition.role.value if hasattr(condition.role, "value") else condition.role)

        # Context details
        if hasattr(condition, "context"):
            ctx = condition.context
            if hasattr(ctx, "acuity"):
                acuity = str(ctx.acuity.value if hasattr(ctx.acuity, "value") else ctx.acuity)
            if hasattr(ctx, "certainty"):
                certainty = str(
                    ctx.certainty.value if hasattr(ctx.certainty, "value") else ctx.certainty
                )
            if hasattr(ctx, "temporality"):
                temporality = str(
                    ctx.temporality.value if hasattr(ctx.temporality, "value") else ctx.temporality
                )

        if hasattr(condition, "anatomical_site") and condition.anatomical_site:
            anatomy = str(condition.anatomical_site)
        if hasattr(condition, "laterality"):
            laterality = str(
                condition.laterality.value
                if hasattr(condition.laterality, "value")
                else condition.laterality
            )

        return diag_id, raw_term, evidence_quote, role, acuity, certainty, anatomy, laterality, temporality

    def _format_candidates(self, candidates: list[ICDCandidate]) -> str:
        """Format candidate list with indices and clinical details for the prompt."""
        lines: list[str] = []
        for i, c in enumerate(candidates, start=1):
            billable_str = "Billable" if c.is_valid_billable else "Non-Billable Header"
            lines.append(
                f"Candidate [{i}]: Code: {c.code} | Description: {c.description} | "
                f"Retrieval Score: {c.retrieval_score:.4f} | Status: {billable_str}"
            )
        return "\n".join(lines)

    def _build_prompt(
        self,
        diagnosis_term: str,
        evidence_quote: str,
        role: str,
        acuity: str,
        certainty: str,
        anatomical_site: str,
        laterality: str,
        candidates_formatted: str,
    ) -> str:
        """Construct prompt combining system instructions and formatted clinical parameters."""
        user_body = self.user_template.format(
            diagnosis_term=diagnosis_term,
            evidence_quote=evidence_quote,
            role=role,
            acuity=acuity,
            certainty=certainty,
            anatomical_site=anatomical_site,
            laterality=laterality,
            candidates_formatted=candidates_formatted,
        )
        return f"{self.system_prompt}\n\n{user_body}"

    def _execute_llm_ranking(
        self,
        prompt: str,
        diag_id: str,
        raw_term: str,
        evidence_quote: str,
        candidates: list[ICDCandidate],
    ) -> RankedSelection | None:
        """Execute LLM call and parse structured response."""
        try:
            assert self.llm is not None
            response_text = self.llm.generate(prompt=prompt, max_tokens=512, temperature=0.0)
            if not response_text or not response_text.strip():
                return None

            json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if not json_match:
                return None

            parsed = json.loads(json_match.group(1))

            raw_code = parsed.get("selected_code")
            code_str = (
                str(raw_code).strip() if raw_code and str(raw_code).lower() != "null" else None
            )
            desc_str = parsed.get("selected_description")
            reason = parsed.get("ranking_reason") or parsed.get("justification", "")
            confidence = float(parsed.get("confidence", parsed.get("confidence_score", 0.85)))
            decision = parsed.get("decision", "ACCEPTED" if code_str else "ABSTAINED")
            abstention_reason = parsed.get("abstention_reason")

            matching_cand = next((c for c in candidates if c.code == code_str), None)
            cand_system = getattr(matching_cand, "coding_system", None) if matching_cand else "ICD-10-CM"
            icd10cm_val = code_str if cand_system == "ICD-10-CM" else None
            icdo_val = code_str if cand_system == "ICD-O" else None
            cpt_val = code_str if cand_system == "CPT" else None

            for c in candidates:
                c_sys = getattr(c, "coding_system", None)
                if c_sys == "ICD-10-CM" and not icd10cm_val:
                    icd10cm_val = c.code
                elif c_sys == "ICD-O" and not icdo_val:
                    icdo_val = c.code
                elif c_sys == "CPT" and not cpt_val:
                    cpt_val = c.code

            return RankedSelection(
                diagnosis_id=diag_id,
                raw_term=raw_term,
                selected_code=code_str,
                selected_description=desc_str,
                selected_icd10cm=icd10cm_val,
                selected_icdo=icdo_val,
                selected_cpt=cpt_val,
                ranking_reason=reason,
                supporting_evidence=[evidence_quote],
                confidence=confidence,
                abstention_reason=abstention_reason,
                candidate_pool=candidates,
                decision=decision,
            )
        except Exception as e:
            logger.warning("Error executing LLM ranking: %s. Using deterministic evaluation.", e)
            return None

    def _evaluate_candidates_deterministically(
        self,
        diag_id: str,
        raw_term: str,
        evidence_quote: str,
        acuity: str,
        candidates: list[ICDCandidate],
    ) -> RankedSelection:
        """Deterministic evidence-based candidate scoring when offline or in test environments."""
        evidence_clean = (evidence_quote or "").lower()
        term_clean = (raw_term or "").lower()
        full_context = f"{term_clean} {evidence_clean}"

        from medical_coding.retrieval.tokenizer import CLINICAL_ABBREVIATIONS, CLINICAL_MORPHOLOGY

        QUALIFIERS_TO_IGNORE = {
            "acute", "chronic", "subacute", "right", "left", "bilateral", "unspecified",
            "primary", "secondary", "severe", "mild", "moderate", "history", "admitted",
            "discharge", "recurrent", "compensated", "decompensated", "status", "post",
            "with", "without", "due", "to", "and", "or", "in", "of", "for",
            "community", "acquired",
        }
        term_tokens = [t for t in re.findall(r"[a-z0-9]+", term_clean) if len(t) > 2]
        core_diag_tokens = [t for t in term_tokens if t not in QUALIFIERS_TO_IGNORE]
        if not core_diag_tokens:
            core_diag_tokens = term_tokens

        # Expand abbreviations (e.g. ACL -> anterior cruciate ligament, NET -> neuroendocrine tumor)
        expanded_core_tokens = list(core_diag_tokens)
        for ct in core_diag_tokens:
            if ct in CLINICAL_ABBREVIATIONS:
                for ab_word in CLINICAL_ABBREVIATIONS[ct].split():
                    if ab_word not in QUALIFIERS_TO_IGNORE and ab_word not in expanded_core_tokens:
                        expanded_core_tokens.append(ab_word)

        scored_candidates: list[tuple[ICDCandidate, float, str]] = []

        for cand in candidates:
            score = cand.retrieval_score
            desc_lower = cand.description.lower()
            rationale_parts: list[str] = []

            # 1. Non-billable penalty
            if not cand.is_valid_billable:
                score *= 0.2
                rationale_parts.append("non-billable category header")

            cand_tokens = [t for t in re.findall(r"[a-z0-9]+", desc_lower) if len(t) > 2]
            cand_tokens_set = set(cand_tokens)

            # 2. Core diagnostic entity matching (using expanded clinical terms)
            core_matches = 0
            for ct in expanded_core_tokens:
                variants = [ct] + CLINICAL_MORPHOLOGY.get(ct, [])
                if any(v in cand_tokens_set for v in variants):
                    core_matches += 1
            core_match_fraction = (core_matches / len(expanded_core_tokens)) if expanded_core_tokens else 1.0

            if core_match_fraction == 0.0:
                score = 0.05
                rationale_parts.append("diagnostic entity mismatch")
            elif core_match_fraction >= 0.5:
                score += 0.15 + (0.25 * core_match_fraction)
                rationale_parts.append(f"core diagnostic entity match ({core_matches}/{len(expanded_core_tokens)})")

            # 3. Direct diagnosis term overlap (differentiates diagnosis from incidental narrative mentions)
            term_overlap = [
                t for t in cand_tokens
                if t in term_tokens or any(v in term_tokens for v in CLINICAL_MORPHOLOGY.get(t, []))
            ]
            if term_overlap:
                score += 0.15 * (len(term_overlap) / len(cand_tokens))

            # 4. Symptom vs Definitive Disease entity rule (CMS Guideline I.B.4)
            # Chapter 18 (R00-R99) or symptom codes (e.g. N23) cannot supersede a definitive pathological entity.
            is_def_disease = any(
                any(ct.endswith(sfx) for sfx in ("itis", "oma", "osis"))
                or ct in ("calculus", "lithiasis", "infarction", "failure", "disease", "disorder", "syndrome", "ulcer", "stenosis", "obstruction", "effusion", "pyelonephritis")
                for ct in core_diag_tokens
            )
            is_symptom_code = (
                cand.code.startswith("R")
                or cand.code == "N23"
                or (
                    any(st in cand_tokens_set for st in ["colic", "pain", "dyspnea", "shortness", "cough", "nausea", "fever", "vomiting", "malaise", "fatigue"])
                    and not (
                        any(dt in cand_tokens_set for dt in ["disease", "disorder", "syndrome", "failure", "infarction", "calculus", "lithiasis", "ulcer"])
                        or any(t.endswith(sfx) for t in cand_tokens_set for sfx in ("itis", "oma", "osis"))
                    )
                )
            )
            if is_def_disease and is_symptom_code:
                score = 0.05
                rationale_parts.append("symptom code superseded by definitive diagnosis")

            # 5. Acuity alignment
            is_acute_doc = bool(re.search(r"\b(?:acute|exacerbation|decompensated|decompensation)\b", full_context))
            is_chronic_doc = bool(re.search(r"\b(?:chronic|longstanding)\b", full_context) or (re.search(r"\bcompensated\b", full_context) and not re.search(r"\bdecompensated\b", full_context)))

            is_acute_cand = (
                ("acute" in desc_lower and "chronic" not in desc_lower)
                or cand.code.startswith(("I21", "J20", "K35", "K85", "N10", "N17"))
                or (cand.code.startswith("K80") and "acute" in desc_lower)
                or (cand.code.startswith("K81") and "acute" in desc_lower)
            )
            is_chronic_cand = "chronic" in desc_lower and "acute" not in desc_lower
            is_acute_on_chronic_cand = "acute on chronic" in desc_lower or (
                "acute" in desc_lower and "chronic" in desc_lower
            )

            if is_acute_on_chronic_cand:
                if is_acute_doc and is_chronic_doc:
                    score += 0.25
                    rationale_parts.append("acute on chronic documentation match")
                elif not is_acute_doc or not is_chronic_doc:
                    score -= 0.30
            elif is_acute_cand:
                if is_acute_doc and not is_chronic_doc:
                    score += 0.20
                    rationale_parts.append("acute acuity match")
                elif is_chronic_doc and not is_acute_doc:
                    score -= 0.25
            elif is_chronic_cand:
                if is_chronic_doc and not is_acute_doc:
                    score += 0.20
                    rationale_parts.append("chronic acuity match")
                elif is_acute_doc and not is_chronic_doc:
                    score -= 0.25

            # 6. Reverse Attribute & Laterality Check (Master Prompt: Code Specificity <= Evidence Specificity)
            is_attr_valid, unsupp_attrs = ReverseAttributeChecker.validate_code_attributes(
                code_description=cand.description,
                evidence_text=evidence_clean,
                diagnosis_term=term_clean,
            )
            if not is_attr_valid:
                # Proportional soft specificity penalty instead of binary disqualification (Master Prompt Section 4 & 25)
                score -= min(0.35, 0.15 * len(unsupp_attrs))
                rationale_parts.append(f"unsupported attributes: {', '.join(unsupp_attrs)}")

            # 7. Laterality alignment (Master Prompt Section 5, 24: reward documented laterality, penalize unspecified laterality)
            doc_left = bool(re.search(r"\b(?:left|lt)\b", full_context))
            doc_right = bool(re.search(r"\b(?:right|rt)\b", full_context))
            cand_left = "left" in desc_lower and "right" not in desc_lower
            cand_right = "right" in desc_lower and "left" not in desc_lower
            cand_unspec_lat = "unspecified" in desc_lower and any(
                w in desc_lower for w in ("side", "foot", "knee", "breast", "arm", "leg", "extremity", "flank", "ankle", "eye", "ear")
            )

            if (doc_left and cand_left) or (doc_right and cand_right):
                score += 0.20
                rationale_parts.append("laterality documentation match")
            elif (doc_left or doc_right) and cand_unspec_lat:
                score -= 0.20
                rationale_parts.append("unspecified laterality when specific side is documented")

            # 8. Preference for Unspecified/Baseline over Other when Documentation is General (CMS Guideline I.A.6)
            if "other " in desc_lower or "other specified" in desc_lower:
                doc_has_other_detail = any(
                    w in full_context for w in ["other", "specified", "variant", "type"]
                )
                if not doc_has_other_detail:
                    score -= 0.20
                    rationale_parts.append("other specified code without documented detail")
            elif "unspecified" in desc_lower or "without " in desc_lower or cand.code.endswith(".9") or cand.code.endswith(".90"):
                missing_core_tokens = [
                    ct for ct in expanded_core_tokens
                    if ct not in cand_tokens_set and not any(v in cand_tokens_set for v in CLINICAL_MORPHOLOGY.get(ct, []))
                ]
                if missing_core_tokens and "unspecified" in desc_lower:
                    score -= 0.15
                    rationale_parts.append(f"unspecified code when specific detail ({', '.join(missing_core_tokens)}) is documented")
                elif not missing_core_tokens and not ((doc_left or doc_right) and cand_unspec_lat):
                    score += 0.10
                    rationale_parts.append("canonical base/unspecified code for general documentation")

            scored_candidates.append(
                (cand, round(score, 4), ", ".join(rationale_parts) or "evidence overlap")
            )

        # Sort descending by raw ranking score to maintain discrimination among top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_cand, top_score, top_rationale = scored_candidates[0]

        # If top candidate score is below min_confidence, or if top candidate has unsupported attributes:
        # Check if there is another candidate in candidate pool that is billable, has no unsupported attributes,
        # and has core_match_fraction > 0.0
        if top_score < self.min_confidence:
            fallback_cand = None
            for cand, s, rat in scored_candidates:
                if not cand.is_valid_billable:
                    continue
                is_valid, _ = ReverseAttributeChecker.validate_code_attributes(
                    cand.description, evidence_clean, term_clean
                )
                if is_valid and s >= 0.35:
                    fallback_cand = cand
                    top_score = s
                    top_rationale = f"specificity realigned: {rat}"
                    break

            if fallback_cand:
                top_cand = fallback_cand
            else:
                abstain_reason = (
                    "UNSUPPORTED_SPECIFICITY"
                    if any("unsupported" in r[2] for r in scored_candidates)
                    else ("NO_MATCHING_ICD_CANDIDATE" if top_score < 0.20 else "CONFIDENCE_BELOW_THRESHOLD")
                )
                clamped_top = max(0.0, min(1.0, round(top_score, 4)))
                return RankedSelection(
                    diagnosis_id=diag_id,
                    raw_term=raw_term,
                    selected_code=None,
                    selected_description=None,
                    ranking_reason=f"No candidate met the minimum confidence threshold ({top_score:.2f} < {self.min_confidence:.2f}). Reason: {top_rationale}",
                    supporting_evidence=[evidence_quote],
                    confidence=clamped_top,
                    ranking_score=clamped_top,
                    abstention_reason=abstain_reason,
                    matching_status="NO_DATABASE_MATCH",
                    candidate_pool=candidates,
                    decision="REJECTED_LOW_CONFIDENCE",
                )

        cand_system = getattr(top_cand, "coding_system", None) or "ICD-10-CM"
        icd10cm_val = top_cand.code if cand_system == "ICD-10-CM" else None
        icdo_val = top_cand.code if cand_system == "ICD-O" else None
        cpt_val = top_cand.code if cand_system == "CPT" else None

        for c in candidates:
            c_sys = getattr(c, "coding_system", None)
            if c_sys == "ICD-10-CM" and not icd10cm_val:
                icd10cm_val = c.code
            elif c_sys == "ICD-O" and not icdo_val:
                icdo_val = c.code
            elif c_sys == "CPT" and not cpt_val:
                cpt_val = c.code

        return RankedSelection(
            diagnosis_id=diag_id,
            raw_term=raw_term,
            selected_code=top_cand.code,
            selected_description=top_cand.description,
            selected_icd10cm=icd10cm_val,
            selected_icdo=icdo_val,
            selected_cpt=cpt_val,
            selected_candidate=top_cand,
            ranking_reason=f"Candidate '{top_cand.code}' ({top_cand.description}) best supported by documentation ({top_rationale}).",
            supporting_evidence=[evidence_quote],
            confidence=max(0.0, min(1.0, round(top_score, 4))),
            abstention_reason=None,
            matching_status="MATCHED",
            candidate_pool=candidates,
            decision="ACCEPTED",
        )
