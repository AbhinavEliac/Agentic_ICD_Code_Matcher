"""Clinical concept matcher and scorer.

Enforces:
1. Exact concept matching over first "good enough" match.
2. Handling of ICD-10-CM nonessential modifiers (parenthetical terms like '(primary)', '(congestive)').
3. Deep evaluation of core clinical entities, anatomical site, laterality, acuity, and complications.
4. Bounded scoring prioritizing the best concept match across all available candidates.
"""

import re
from typing import Any

from medical_coding.schemas.icd import ICDCandidate
from medical_coding.utils.clinical_lexicon import CLINICAL_ABBREVIATIONS, CLINICAL_MORPHOLOGY
from medical_coding.validation.reverse_attributes import ReverseAttributeChecker

# Common clinical qualifiers that describe severity, status, or administrative attributes
PERIPHERAL_QUALIFIERS = {
    "mild", "moderate", "severe", "routine", "unspecified", "uncontrolled", "controlled",
    "status", "post", "s/p", "primary", "secondary", "admitted", "discharge", "history",
    "compensated", "decompensated", "recurrent", "underlying", "known", "case", "of",
    "with", "without", "and", "or", "in", "for", "to", "from", "due",
}


def strip_nonessential_modifiers(text: str) -> str:
    """Strip ICD-10-CM nonessential modifiers (parenthetical text in descriptions).

    Under official ICD-10-CM conventions, terms in parentheses are nonessential modifiers
    that do not affect the code assignment (e.g., 'Essential (primary) hypertension',
    'Acute systolic (congestive) heart failure').
    """
    if not text:
        return ""
    # Strip parentheses and their contents
    stripped = re.sub(r"\([^)]*\)", " ", text)
    stripped = re.sub(r"\[[^\]]*\]", " ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def normalize_for_concept_matching(text: str) -> str:
    """Normalize a clinical phrase or ICD description for exact/semantic concept comparison."""
    if not text:
        return ""

    cleaned = text.strip().lower()
    # Strip nonessential modifiers
    cleaned = strip_nonessential_modifiers(cleaned)
    # Strip list markers, dates, codes
    cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", cleaned)
    # Remove punctuation
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # Expand abbreviations
    tokens = cleaned.split()
    expanded_tokens: list[str] = []
    for t in tokens:
        if t in CLINICAL_ABBREVIATIONS:
            expanded_tokens.extend(CLINICAL_ABBREVIATIONS[t].split())
        else:
            expanded_tokens.append(t)

    return " ".join(expanded_tokens).strip()


def get_core_concept_tokens(text: str) -> list[str]:
    """Extract significant core clinical entity tokens (ignoring peripheral qualifiers)."""
    norm = normalize_for_concept_matching(text)
    tokens = [t for t in norm.split() if len(t) > 2]
    core = [t for t in tokens if t not in PERIPHERAL_QUALIFIERS]
    return core or tokens


def compute_concept_match_score(
    candidate_code: str,
    candidate_description: str,
    concept_name: str,
    evidence_text: str = "",
    attributes: dict[str, Any] | None = None,
) -> tuple[float, str, bool]:
    """Compute exact and dimensional concept match score between a candidate and clinical concept.

    Returns:
        (score, rationale, is_exact_match)
    """
    if not candidate_description or not concept_name:
        return 0.0, "Missing candidate description or concept name", False

    cand_desc_clean = candidate_description.strip()
    concept_clean = concept_name.strip()
    evidence_clean = (evidence_text or "").lower()
    full_context = f"{concept_clean.lower()} {evidence_clean}"

    # 1. Normalized string comparisons
    cand_norm = normalize_for_concept_matching(cand_desc_clean)
    concept_norm = normalize_for_concept_matching(concept_clean)
    cand_raw_lower = re.sub(r"[^\w\s]", "", cand_desc_clean.lower()).strip()
    concept_raw_lower = re.sub(r"[^\w\s]", "", concept_clean.lower()).strip()

    # Exact string match (ignoring nonessential parentheticals & punctuation)
    if cand_norm == concept_norm or cand_raw_lower == concept_raw_lower:
        return 1.0, "exact clinical description match", True

    # 2. Token sets
    cand_tokens = cand_norm.split()
    concept_tokens = concept_norm.split()
    cand_token_set = set(cand_tokens)
    concept_token_set = set(concept_tokens)

    # Core diagnostic entity tokens
    core_concept_tokens = get_core_concept_tokens(concept_clean)
    core_matches = 0
    for ct in core_concept_tokens:
        variants = [ct] + CLINICAL_MORPHOLOGY.get(ct, [])
        if any(v in cand_token_set for v in variants):
            core_matches += 1

    core_fraction = (core_matches / len(core_concept_tokens)) if core_concept_tokens else 0.0

    # If core entity does not match at all (e.g. kidney vs skin), candidate is incompatible
    if core_fraction == 0.0 and len(core_concept_tokens) >= 1:
        return 0.05, "core clinical entity mismatch", False

    # 3. Base score derived from core fraction and token overlap
    shared_tokens = cand_token_set & concept_token_set
    jaccard = len(shared_tokens) / len(cand_token_set | concept_token_set) if (cand_token_set | concept_token_set) else 0.0

    base_score = 0.50 + (0.35 * core_fraction) + (0.15 * jaccard)
    rationale_parts: list[str] = [f"core entity match ({core_matches}/{len(core_concept_tokens)})"]

    # If all concept tokens appear in candidate
    if concept_token_set.issubset(cand_token_set):
        base_score += 0.10
        rationale_parts.append("full concept token coverage")

    # 4. Acuity alignment
    is_acute_doc = bool(re.search(r"\b(?:acute|exacerbation|decompensated)\b", full_context))
    is_chronic_doc = bool(re.search(r"\b(?:chronic|longstanding|compensated)\b", full_context))
    cand_is_acute = "acute" in cand_norm and "chronic" not in cand_norm
    cand_is_chronic = "chronic" in cand_norm and "acute" not in cand_norm
    cand_is_acute_on_chronic = "acute on chronic" in cand_desc_clean.lower() or ("acute" in cand_norm and "chronic" in cand_norm)

    if cand_is_acute_on_chronic:
        if is_acute_doc and is_chronic_doc:
            base_score += 0.10
            rationale_parts.append("acute on chronic match")
        elif not is_acute_doc or not is_chronic_doc:
            base_score -= 0.20
    elif cand_is_acute:
        if is_acute_doc and not is_chronic_doc:
            base_score += 0.10
            rationale_parts.append("acute acuity match")
        elif is_chronic_doc and not is_acute_doc:
            base_score -= 0.20
            rationale_parts.append("unsupported acute acuity")
    elif cand_is_chronic:
        if is_chronic_doc and not is_acute_doc:
            base_score += 0.10
            rationale_parts.append("chronic acuity match")
        elif is_acute_doc and not is_chronic_doc:
            base_score -= 0.20
            rationale_parts.append("unsupported chronic acuity")

    # 5. Laterality alignment
    doc_left = bool(re.search(r"\b(?:left|lt)\b", full_context))
    doc_right = bool(re.search(r"\b(?:right|rt)\b", full_context))
    cand_left = "left" in cand_norm and "right" not in cand_norm
    cand_right = "right" in cand_norm and "left" not in cand_norm
    cand_unspec_lat = "unspecified" in cand_norm and any(
        w in cand_norm for w in ("side", "breast", "kidney", "ureter", "flank", "foot", "ankle", "knee", "extremity")
    )

    if (doc_left and cand_left) or (doc_right and cand_right):
        base_score += 0.15
        rationale_parts.append("laterality match")
    elif (doc_left and cand_right) or (doc_right and cand_left):
        return 0.0, "contradictory laterality", False
    elif (doc_left or doc_right) and cand_unspec_lat:
        base_score -= 0.15
        rationale_parts.append("unspecified laterality when specific side documented")

    # 6. Reverse attributes check
    is_attr_valid, unsupp = ReverseAttributeChecker.validate_code_attributes(
        code_description=candidate_description,
        evidence_text=evidence_clean,
        diagnosis_term=concept_clean,
        code=candidate_code,
    )
    if not is_attr_valid:
        has_hard_fail = any(
            any(k in u.lower() for k in ("demographic", "perinatal", "neonatal", "pregnancy", "obstetric", "polarity", "contradictory", "mismatch", "pathogen"))
            for u in unsupp
        )
        if has_hard_fail:
            base_score = 0.0
            rationale_parts.append(f"disqualified by clinical contradiction: {', '.join(unsupp)}")
        else:
            base_score -= min(0.40, 0.15 * len(unsupp))
            rationale_parts.append(f"unsupported attributes: {', '.join(unsupp)}")

    # 7. Bleeding / manifestation status
    has_no_bleed = "no active bleeding" in full_context or "without bleeding" in full_context or "no bleeding" in full_context
    has_bleed = "active bleeding" in full_context or "with bleeding" in full_context or "with hemorrhage" in full_context
    if has_no_bleed and "without bleeding" in cand_norm:
        base_score += 0.15
        rationale_parts.append("without bleeding documentation match")
    elif has_no_bleed and "with bleeding" in cand_norm:
        base_score -= 0.35
        rationale_parts.append("unsupported bleeding in candidate")
    elif has_bleed and "with bleeding" in cand_norm:
        base_score += 0.15
        rationale_parts.append("with bleeding documentation match")

    # 8. Unspecified vs specific alignment
    is_unspec_cand = candidate_code.endswith((".9", ".90", ".99")) or "unspecified" in cand_norm
    has_specific_detail = any(
        w in full_context
        for w in ["systolic", "diastolic", "anterior", "inferior", "lateral", "retroperitoneal", "ureteric", "ureter", "antral"]
    )
    if is_unspec_cand and has_specific_detail:
        base_score -= 0.10
    elif is_unspec_cand and not has_specific_detail:
        base_score += 0.08
        rationale_parts.append("canonical unspecified code for general condition")

    final_score = max(0.0, min(1.0, round(base_score, 4)))
    is_exact = final_score >= 0.95 and core_fraction >= 0.90
    return final_score, ", ".join(rationale_parts), is_exact


def select_best_concept_match(
    candidates: list[ICDCandidate],
    concept_name: str,
    evidence_text: str = "",
    catalog: Any = None,
    attributes: dict[str, Any] | None = None,
    min_score_threshold: float = 0.35,
) -> tuple[ICDCandidate | None, float, str]:
    """Evaluate ALL candidates in the pool and select the single BEST concept match.

    NEVER stops at the first candidate that passes a threshold.
    Ranks every candidate and selects the highest scoring match.
    """
    if not candidates or not concept_name:
        return None, 0.0, "No candidates or empty concept name"

    scored_list: list[tuple[ICDCandidate, float, str, bool]] = []

    for cand in candidates:
        # Filter non-billable headers if billable required
        if catalog is not None:
            if not catalog.is_valid_code(cand.code):
                continue

        score, rat, is_exact = compute_concept_match_score(
            candidate_code=cand.code,
            candidate_description=cand.description,
            concept_name=concept_name,
            evidence_text=evidence_text,
            attributes=attributes,
        )

        # Non-billable penalty
        if not cand.is_valid_billable:
            score *= 0.3
            rat = f"non-billable header ({rat})"

        # Incorporate original retrieval score gently (up to 15% weight)
        cand_retrieval_s = getattr(cand, "retrieval_score", 0.5) or 0.5
        composite_score = round((0.85 * score) + (0.15 * cand_retrieval_s), 4)

        scored_list.append((cand, composite_score, rat, is_exact))

    if not scored_list:
        return None, 0.0, "No valid candidates in candidate pool"

    # Sort descending by composite concept score
    scored_list.sort(key=lambda x: x[1], reverse=True)

    best_cand, best_score, best_rat, best_is_exact = scored_list[0]

    if best_score < min_score_threshold:
        return None, best_score, f"Top candidate '{best_cand.code}' below threshold ({best_score:.2f} < {min_score_threshold:.2f}): {best_rat}"

    return best_cand, best_score, best_rat
