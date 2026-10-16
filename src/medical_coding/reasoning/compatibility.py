"""Compatibility Reasoner: Post-retrieval validation verifying clinical concept compatibility.

Conforms to Master Specification Section 10:
- Evaluates disease equivalence, family, anatomical site, laterality, histology,
  metastatic direction, required attributes, unsupported specificity, and contradictions.
- Prevents database candidates from inventing or altering clinical diagnoses.
- Enforces CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY.
"""

import re

from medical_coding.schemas.reasoning import ClinicalConcept, CompatibilityResult, MatchSpec
from medical_coding.utils.logging import get_logger
from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

logger = get_logger(__name__)


class CompatibilityReasoner:
    """Evaluates whether a database code candidate accurately and safely represents a ClinicalConcept."""

    @classmethod
    def evaluate_candidate(
        cls,
        candidate_code: str,
        candidate_description: str,
        concept: ClinicalConcept,
        match_spec: MatchSpec | None = None,
        evidence_text: str = "",
    ) -> CompatibilityResult:
        """Validate candidate against ClinicalConcept and MatchSpec."""
        desc_lower = candidate_description.lower()
        code_upper = candidate_code.upper().strip()
        contradictions: list[str] = []
        supported_attrs: list[str] = []
        unsupported_attrs: list[str] = []

        # 0. Concept-Level Assertion & Contradiction Evaluation (Hard Contradictions)
        if getattr(concept, "assertion_status", None) == "RULED_OUT":
            contradictions.append(f"Clinical concept '{concept.canonical_name}' is ruled out/excluded in documentation")
        for ca in getattr(concept, "contradictory_attributes", []):
            contradictions.append(f"Concept contradiction: {ca}")
        for ce in getattr(concept, "contradiction_evidence", []):
            contradictions.append(f"Contradicting evidence: {ce}")

        # 1. Family / Prefix Compatibility (Master Prompt Section 20 & 21: SOFT Prior)
        matches_family = True
        if match_spec and match_spec.allowed_code_families:
            matches_family = any(
                code_upper.startswith(prefix) or code_upper.replace(".", "").startswith(prefix.replace(".", ""))
                for prefix in match_spec.allowed_code_families
            )

        # 2. Forbidden Attributes Check (Master Prompt Section 5: HARD Contradiction)
        if match_spec and match_spec.forbidden_attributes:
            for forbidden in match_spec.forbidden_attributes:
                forbidden_lower = forbidden.lower()
                if forbidden_lower.startswith("c") and forbidden_lower in code_upper.lower():
                    contradictions.append(f"Forbidden code match: {forbidden}")
                elif re.search(rf"\b{re.escape(forbidden_lower)}\b", desc_lower):
                    contradictions.append(f"Forbidden attribute present: '{forbidden}'")

        # 3. Required / Laterality Contradiction Check (Master Prompt Section 5)
        if match_spec and match_spec.required_attributes:
            for req_key, req_val in match_spec.required_attributes.items():
                if req_key == "laterality":
                    lat_str = str(req_val).lower()
                    if lat_str == "left" and "right" in desc_lower and "left" not in desc_lower:
                        contradictions.append("Conflicting laterality: candidate requires right side for left-sided condition")
                    elif lat_str == "right" and "left" in desc_lower and "right" not in desc_lower:
                        contradictions.append("Conflicting laterality: candidate requires left side for right-sided condition")
                elif req_key in ("site", "primary_site"):
                    site_str = str(req_val).lower()
                    from medical_coding.retrieval.tokenizer import CLINICAL_MORPHOLOGY
                    variants = [site_str] + CLINICAL_MORPHOLOGY.get(site_str, [])
                    if not any(v in desc_lower for v in variants):
                        contradictions.append(f"Conflicting anatomical site: candidate lacks required site '{site_str}'")

        # 4. Reverse Attribute Entailment Check (CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY)
        combined_ev = " ".join(concept.evidence_spans) + f" {evidence_text}"
        is_attr_valid, unsupp = ReverseAttributeChecker.validate_code_attributes(
            code_description=candidate_description,
            evidence_text=combined_ev,
            diagnosis_term=concept.canonical_name,
            code=candidate_code,
        )
        if not is_attr_valid:
            unsupported_attrs.extend(unsupp)
            for u in unsupp:
                if "contradiction" in u.lower() or "conflict" in u.lower():
                    contradictions.append(u)

        # Identify supported attributes in code description
        for a in concept.supported_attributes:
            if a in desc_lower:
                supported_attrs.append(a)

        # 5. Composite Concept Match & Dimensional Compatibility Score
        from medical_coding.reasoning.concept_matcher import compute_concept_match_score

        concept_score, match_rat, is_exact = compute_concept_match_score(
            candidate_code=candidate_code,
            candidate_description=candidate_description,
            concept_name=concept.canonical_name or concept.normalized_concept or "",
            evidence_text=combined_ev,
        )

        if is_exact and not contradictions and not unsupported_attrs:
            final_score = 1.0
            is_compatible = True
            reason = f"Candidate '{candidate_code}' is an exact clinical concept match for '{concept.canonical_name}' ({match_rat})."
            return CompatibilityResult(
                compatible=is_compatible,
                score=final_score,
                candidate_code=candidate_code,
                candidate_description=candidate_description,
                supported_attributes=supported_attrs,
                unsupported_attributes=unsupported_attrs,
                contradictions=contradictions,
                reason=reason,
            )

        base_score = concept_score
        if matches_family:
            base_score = min(1.0, base_score + 0.08)
        else:
            base_score = max(0.0, base_score - 0.15)

        if supported_attrs:
            base_score = min(1.0, base_score + 0.05)

        # Specificity penalty for unevidenced qualifiers (e.g. bleeding when none documented)
        spec_penalty = min(0.35, 0.10 * len(unsupported_attrs))
        base_score -= spec_penalty

        # Zero tolerance for explicit contradictions (Master Specification Section 5 & 10)
        if contradictions:
            final_score = 0.0
            is_compatible = False
        else:
            final_score = max(0.0, min(1.0, round(base_score, 4)))
            is_compatible = final_score >= 0.35

        if is_compatible:
            reason = f"Candidate '{candidate_code}' is clinically compatible with concept '{concept.canonical_name}' (score={final_score}, {match_rat})."
        else:
            reason_parts: list[str] = [match_rat]
            if contradictions:
                reason_parts.append(f"Contradictions: {'; '.join(contradictions)}")
            if unsupported_attrs:
                reason_parts.append(f"Unsupported qualifiers: {', '.join(unsupported_attrs)}")
            reason = f"Candidate incompatible (score={final_score}). {'; '.join(reason_parts)}"

        return CompatibilityResult(
            compatible=is_compatible,
            score=final_score,
            candidate_code=candidate_code,
            candidate_description=candidate_description,
            supported_attributes=supported_attrs,
            unsupported_attributes=unsupported_attrs,
            contradictions=contradictions,
            reason=reason,
        )
