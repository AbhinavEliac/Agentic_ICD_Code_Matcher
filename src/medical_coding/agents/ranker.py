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
from medical_coding.schemas.enums import DiagnosisRole
from medical_coding.schemas.icd import ICDCandidate, RankedSelection
from medical_coding.utils.logging import get_logger
from medical_coding.utils.text import format_icd_code
from medical_coding.validation.deterministic import (
    CandidateRankingDeterministicValidator,
)

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
        diag_id, raw_term, evidence_quote, role, acuity, certainty, anatomy, laterality = (
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

        # 4. Short-circuit: Excluded condition -> Immediate Abstention
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

        if hasattr(condition, "anatomical_site") and condition.anatomical_site:
            anatomy = str(condition.anatomical_site)
        if hasattr(condition, "laterality"):
            laterality = str(
                condition.laterality.value
                if hasattr(condition.laterality, "value")
                else condition.laterality
            )

        return diag_id, raw_term, evidence_quote, role, acuity, certainty, anatomy, laterality

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

        scored_candidates: list[tuple[ICDCandidate, float, str]] = []

        for cand in candidates:
            score = cand.retrieval_score
            desc_lower = cand.description.lower()
            rationale_parts: list[str] = []

            # 1. Non-billable penalty
            if not cand.is_valid_billable:
                score *= 0.5
                rationale_parts.append("non-billable category header")

            # 2. Terminology and specific descriptor matches
            cand_tokens = [t for t in re.findall(r"[a-z0-9]+", desc_lower) if len(t) > 2]
            overlap = [t for t in cand_tokens if t in full_context]
            if overlap:
                score += 0.15 * (len(overlap) / len(cand_tokens))

            # 3. Acuity alignment
            is_acute_doc = any(
                w in full_context
                for w in ["acute", "exacerbation", "decompensated", "decompensation"]
            )
            is_chronic_doc = any(
                w in full_context for w in ["chronic", "compensated", "longstanding"]
            )

            is_acute_cand = "acute" in desc_lower and "chronic" not in desc_lower
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

            # 4. Specificity penalty for unsupported subtypes and etiologies
            # Etiology support check: transplant, rheumatic, hypertensive, postprocedural, congenital, etc.
            etiologies = {
                "transplant": ["transplant", "allograft", "graft"],
                "rheumatic": ["rheumatic"],
                "hypertensive": ["hypertension", "hypertensive", "htn", "high blood pressure"],
                "postprocedural": ["postprocedural", "postoperative", "post-op", "complication of surgery"],
                "congenital": ["congenital", "birth defect", "anomaly"],
                "toxic": ["toxic", "toxicity", "poisoning"],
                "alcoholic": ["alcoholic", "alcohol", "etoh"],
            }
            has_unsupported_etiology = False
            for etio_word, doc_cues in etiologies.items():
                if etio_word in desc_lower and not any(w in full_context for w in doc_cues):
                    score = 0.05
                    rationale_parts.append(f"unsupported {etio_word} etiology")
                    has_unsupported_etiology = True
                    break

            # Heart failure: systolic / diastolic / right / left / end stage
            if not has_unsupported_etiology and ("heart" in desc_lower and "failure" in desc_lower):
                for sub in ["systolic", "diastolic", "right", "left", "high output", "end stage", "biventricular", "other"]:
                    if sub in desc_lower and not any(
                        w in full_context for w in [sub, "hfref", "hfpef", "reduced ejection", "preserved ejection"]
                    ):
                        score = 0.05
                        rationale_parts.append(f"unsupported {sub} specificity")
                        break

            # Diabetes complications
            if any(
                comp in desc_lower
                for comp in ["nephropathy", "retinopathy", "neuropathy", "hyperglycemia"]
            ):
                if not any(
                    comp in full_context
                    for comp in [
                        "nephropathy",
                        "kidney",
                        "retinopathy",
                        "neuropathy",
                        "hyperglycemia",
                        "uncontrolled",
                    ]
                ):
                    score = 0.05
                    rationale_parts.append("unsupported complication specificity")

            clamped_score = max(0.0, min(1.0, round(score, 4)))
            scored_candidates.append(
                (cand, clamped_score, ", ".join(rationale_parts) or "evidence overlap")
            )

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_cand, top_score, top_rationale = scored_candidates[0]

        if top_score < self.min_confidence:
            return RankedSelection(
                diagnosis_id=diag_id,
                raw_term=raw_term,
                selected_code=None,
                selected_description=None,
                ranking_reason=f"No candidate met the minimum confidence threshold ({top_score:.2f} < {self.min_confidence:.2f}).",
                supporting_evidence=[evidence_quote],
                confidence=top_score,
                abstention_reason="CONFIDENCE_BELOW_THRESHOLD",
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
            confidence=top_score,
            abstention_reason=None,
            candidate_pool=candidates,
            decision="ACCEPTED",
        )
