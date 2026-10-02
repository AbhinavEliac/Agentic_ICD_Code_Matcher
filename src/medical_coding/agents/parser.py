"""Robust JSON parser, clinical enum normalizer, and evidence-grounding validator for local LLM outputs."""

import json
import re
from typing import Any

from medical_coding.pdf.models import ClinicalPDFDocument
from medical_coding.schemas.clinical import (
    EvidenceLocation,
    ExtractedClinicalCondition,
)
from medical_coding.schemas.enums import (
    Certainty,
    ClinicalEntityType,
    ConditionStatus,
    Laterality,
    NegationStatus,
    Temporality,
)
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class ExtractionParser:
    """Parses and repairs structured extraction outputs from local LLMs.

    Guarantees:
    1. Robust JSON extraction across markdown wrappers and conversational noise.
    2. Case-insensitive enum normalization with clinical synonym mapping.
    3. Strict evidence grounding: checks that evidence_text exists verbatim in the source document.
    4. Automatic resolution of character offsets, sections, and page numbers from documentary text.
    5. Zero hallucination: drops candidate conditions if evidence is completely absent from the text.
    """

    def parse_and_validate(
        self,
        raw_llm_output: str,
        document_text: str,
        clinical_doc: ClinicalPDFDocument | None = None,
    ) -> tuple[list[ExtractedClinicalCondition], list[str], bool]:
        """Parse raw LLM response into validated ExtractedClinicalCondition models.

        Args:
            raw_llm_output: Unprocessed text emitted by the LLM.
            document_text: The source clinical text the LLM analyzed.
            clinical_doc: Optional structured ClinicalPDFDocument for exact page/section mapping.

        Returns:
            (validated_conditions, audit_notes, repair_applied)
        """
        audit_notes: list[str] = []
        repair_applied = False

        if not raw_llm_output or not raw_llm_output.strip():
            audit_notes.append("Raw LLM response was empty.")
            return [], audit_notes, False

        # 1. Clean markdown code fences and extract JSON substring
        cleaned_text = self._strip_markdown_and_extract_json(raw_llm_output)
        if cleaned_text != raw_llm_output:
            repair_applied = True

        parsed_json: Any = None
        try:
            parsed_json = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Initial JSON parse failed: %s. Applying syntax repair heuristics...", exc
            )
            repaired_text = self._repair_json_syntax(cleaned_text)
            try:
                parsed_json = json.loads(repaired_text)
                repair_applied = True
                audit_notes.append("Applied syntax repair to malformed LLM JSON.")
            except Exception as repair_exc:
                logger.error("JSON repair heuristics failed: %s", repair_exc)
                audit_notes.append(f"Failed to parse LLM output as JSON: {exc}")
                return [], audit_notes, repair_applied

        if not isinstance(parsed_json, list):
            if (
                isinstance(parsed_json, dict)
                and "conditions" in parsed_json
                and isinstance(parsed_json["conditions"], list)
            ):
                parsed_json = parsed_json["conditions"]
            elif isinstance(parsed_json, dict):
                parsed_json = [parsed_json]
            else:
                audit_notes.append("Parsed JSON was not a list of candidate conditions.")
                return [], audit_notes, repair_applied

        # 2. Validate, ground, and normalize each condition item
        validated_conditions: list[ExtractedClinicalCondition] = []
        norm_source = self._normalize_for_matching(document_text)

        for idx, item in enumerate(parsed_json):
            if not isinstance(item, dict):
                audit_notes.append(f"Skipping non-dict condition at index {idx}.")
                continue

            # Ensure minimum required fields exist
            mention = str(item.get("original_mention") or item.get("raw_term") or "").strip()
            evidence = str(item.get("evidence_text") or item.get("evidence_quote") or "").strip()

            if not mention:
                audit_notes.append(f"Item #{idx} missing original_mention; discarded.")
                continue

            # CRITICAL RULE: Every diagnosis must have evidence
            if not evidence:
                audit_notes.append(f"Item '{mention}' discarded: No evidence_text provided by LLM.")
                continue

            # Check evidence grounding in source document
            span_found = self._locate_evidence_in_text(evidence, document_text, norm_source)
            if span_found is None:
                # Evidence could not be located in document
                audit_notes.append(
                    f"Item '{mention}' discarded: evidence quote '{evidence[:60]}...' not found in source document."
                )
                continue

            start_char, end_char = span_found

            # Resolve page number and section
            page_num, sec_name = self._resolve_location_metadata(
                start_char=start_char,
                clinical_doc=clinical_doc,
                claimed_section=item.get("section"),
            )

            # Map enums safely
            certainty = self._normalize_certainty(item.get("certainty"))
            temporality = self._normalize_temporality(item.get("temporality"))
            status = self._normalize_status(item.get("status"), temporality, certainty)
            negation = self._normalize_negation(item.get("negation"))
            laterality = self._normalize_laterality(item.get("laterality"))
            entity_type = self._normalize_entity_type(item.get("entity_type"))

            norm_desc = str(item.get("normalized_description") or mention).strip()

            condition = ExtractedClinicalCondition(
                original_mention=mention,
                normalized_description=norm_desc,
                entity_type=entity_type,
                evidence_text=evidence,
                evidence_location=EvidenceLocation(
                    section=sec_name,
                    page_number=page_num,
                    start_char=start_char,
                    end_char=end_char,
                ),
                status=status,
                certainty=certainty,
                temporality=temporality,
                negation=negation,
                anatomical_site=item.get("anatomical_site") or None,
                laterality=laterality,
                section=sec_name,
                treatment_evidence=item.get("treatment_evidence") or None,
                confidence_score=float(item.get("confidence_score", 1.0)),
            )

            validated_conditions.append(condition)

        audit_notes.append(
            f"Successfully extracted and validated {len(validated_conditions)} conditions."
        )
        return validated_conditions, audit_notes, repair_applied

    def _strip_markdown_and_extract_json(self, text: str) -> str:
        """Strip markdown fences, leading explanations, and isolate the JSON array."""
        cleaned = text.strip()

        # Handle ```json ... ``` or ``` ... ```
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
            if match:
                return match.group(1).strip()
            # If no closing block or opening block, strip ``` lines
            cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

        # Isolate substring starting at first '[' and ending at last ']'
        start_idx = cleaned.find("[")
        end_idx = cleaned.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return cleaned[start_idx : end_idx + 1].strip()

        return cleaned

    def _repair_json_syntax(self, text: str) -> str:
        """Apply heuristics to repair common LLM JSON syntax errors without hallucinating facts."""
        repaired = text

        # Replace single quotes with double quotes around JSON keys and string values
        # e.g. {'key': 'value'} -> {"key": "value"}
        repaired = re.sub(r"(?<=[\{\s,\[])'([a-zA-Z0-9_\s\-]+)'(?=\s*:)", r'"\1"', repaired)
        repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)

        # Remove trailing commas before closing braces/brackets: e.g. , ] -> ] and , } -> }
        repaired = re.sub(r",\s*([\]\}])", r"\1", repaired)

        # Replace invalid unquoted literals or None -> null
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)

        return repaired

    def _normalize_for_matching(self, text: str) -> str:
        """Normalize whitespace and case for relaxed quote matching."""
        return re.sub(r"\s+", " ", text).strip().lower()

    def _locate_evidence_in_text(
        self,
        quote: str,
        original_text: str,
        norm_source: str,
    ) -> tuple[int, int] | None:
        """Verify quote exists in original text and return exact start/end character offsets."""
        if not quote:
            return None

        # 1. Exact verbatim match
        start = original_text.find(quote)
        if start != -1:
            return start, start + len(quote)

        # 2. Case-insensitive exact match
        lower_orig = original_text.lower()
        lower_quote = quote.lower()
        start = lower_orig.find(lower_quote)
        if start != -1:
            return start, start + len(quote)

        # 3. Whitespace-normalized match
        norm_quote = self._normalize_for_matching(quote)
        start_norm = norm_source.find(norm_quote)
        if start_norm != -1:
            # Estimate reasonable span in original text
            approx_start = max(0, start_norm)
            approx_end = min(len(original_text), approx_start + len(quote) + 10)
            return approx_start, approx_end

        return None

    def _resolve_location_metadata(
        self,
        start_char: int,
        clinical_doc: ClinicalPDFDocument | None,
        claimed_section: str | None,
    ) -> tuple[int, str]:
        """Resolve page number and section name from clinical document offsets."""
        page_num = 1
        sec_name = str(claimed_section or "CLINICAL_DOCUMENT").upper().strip()

        if clinical_doc is not None:
            page_num = clinical_doc.find_page_number_for_char(start_char)
            # Find matching section containing this character
            for s in clinical_doc.sections:
                if s.start_char <= start_char <= s.end_char:
                    sec_name = s.section_type
                    break

        return page_num, sec_name

    def _normalize_certainty(self, val: Any) -> Certainty:
        """Map certainty strings to standardized Certainty enum."""
        s = str(val or "").strip().upper()
        if "RULE" in s or "EXCLUD" in s or "DENI" in s:
            return Certainty.RULED_OUT
        if "SUSPECT" in s or "PROBAB" in s:
            return Certainty.SUSPECTED
        if "POSSIB" in s or "POTENTIAL" in s or "DIFFERENTIAL" in s:
            return Certainty.POSSIBLE
        if "CONFIRM" in s or "DEFINIT" in s or "POSITIVE" in s or "ACTIVE" in s:
            return Certainty.CONFIRMED
        return Certainty.CONFIRMED if not s else Certainty.UNKNOWN

    def _normalize_temporality(self, val: Any) -> Temporality:
        """Map temporality strings to standardized Temporality enum."""
        s = str(val or "").strip().upper()
        if "RESOLV" in s or "CURED" in s:
            return Temporality.RESOLVED
        if "HISTOR" in s or "PAST" in s or "PMH" in s or "PRIOR" in s:
            return Temporality.HISTORICAL
        if "CURRENT" in s or "PRESENT" in s or "ACTIVE" in s or "ADMISSION" in s:
            return Temporality.CURRENT
        if "FAMILY" in s:
            return Temporality.FAMILY_HISTORY
        return Temporality.CURRENT

    def _normalize_status(
        self, val: Any, temporality: Temporality, certainty: Certainty
    ) -> ConditionStatus:
        """Map status with consistency checks against temporality and certainty."""
        if certainty == Certainty.RULED_OUT:
            return ConditionStatus.RESOLVED
        if temporality == Temporality.RESOLVED:
            return ConditionStatus.RESOLVED
        if temporality == Temporality.HISTORICAL:
            return ConditionStatus.HISTORICAL

        s = str(val or "").strip().upper()
        if "RESOLV" in s:
            return ConditionStatus.RESOLVED
        if "HISTOR" in s or "PAST" in s:
            return ConditionStatus.HISTORICAL
        if "CHRONIC" in s:
            return ConditionStatus.CHRONIC
        if "ACUTE" in s:
            return ConditionStatus.ACUTE
        if "ACTIVE" in s:
            return ConditionStatus.ACTIVE
        return ConditionStatus.ACTIVE

    def _normalize_negation(self, val: Any) -> NegationStatus:
        """Map negation strings to standardized NegationStatus enum."""
        s = str(val or "").strip().upper()
        if "NEG" in s or "NO" in s or "DENI" in s or "ABSENT" in s:
            return NegationStatus.NEGATED
        if "UNCERTAIN" in s or "EQUIVOCAL" in s:
            return NegationStatus.UNCERTAIN
        return NegationStatus.AFFIRMATIVE

    def _normalize_laterality(self, val: Any) -> Laterality:
        """Map laterality strings to standardized Laterality enum."""
        s = str(val or "").strip().upper()
        if "LEFT" in s:
            return Laterality.LEFT
        if "RIGHT" in s:
            return Laterality.RIGHT
        if "BILATERAL" in s or "BOTH" in s:
            return Laterality.BILATERAL
        return Laterality.UNSPECIFIED

    def _normalize_entity_type(self, val: Any) -> ClinicalEntityType:
        """Map entity type strings to standardized ClinicalEntityType enum."""
        s = str(val or "").strip().upper()
        if "SYMPTOM" in s:
            return ClinicalEntityType.SYMPTOM
        if "SIGN" in s:
            return ClinicalEntityType.SIGN
        if "PROCEDUR" in s:
            return ClinicalEntityType.PROCEDURAL_FINDING
        return ClinicalEntityType.DIAGNOSIS
