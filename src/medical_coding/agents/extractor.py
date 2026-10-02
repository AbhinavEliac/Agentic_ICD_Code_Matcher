"""Clinical Information Extraction Agent extracting structured conditions with verbatim evidence grounding."""

from typing import Any
from uuid import uuid4

from medical_coding.agents.base import BaseAgent
from medical_coding.agents.parser import ExtractionParser
from medical_coding.models.base import BaseLocalLLM
from medical_coding.models.langchain_llm import LocalGPT4AllLangChainLLM
from medical_coding.pdf.models import ClinicalPDFDocument
from medical_coding.prompts.extraction import (
    CLINICAL_EXTRACTION_RETRY_TEMPLATE,
    CLINICAL_EXTRACTION_SYSTEM_PROMPT,
    CLINICAL_EXTRACTION_USER_TEMPLATE,
)
from medical_coding.schemas.clinical import (
    ClinicalExtractionResult,
    EvidenceSnippet,
    ExtractedClinicalCondition,
    ExtractedDiagnosis,
    TextSpan,
)
from medical_coding.schemas.enums import (
    Certainty,
    ConditionStatus,
    NegationStatus,
    Temporality,
)
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class ClinicalExtractionAgent(BaseAgent):
    """Extracts candidate medical conditions, certainty, temporality, and evidence without assigning ICD codes.

    ARCHITECTURAL PRINCIPLES:
    1. NEVER generates or invents ICD codes.
    2. Every condition MUST have verbatim documentary evidence.
    3. Reconstructs structured context (CONFIRMED, SUSPECTED, POSSIBLE, RULED_OUT, HISTORICAL, RESOLVED, CURRENT).
    4. Handles contradictory mentions and past medical history appropriately.
    5. Robust parser with retry policy for local quantized LLMs.
    """

    def __init__(
        self,
        llm: BaseLocalLLM | LocalGPT4AllLangChainLLM,
        max_retries: int = 2,
        parser: ExtractionParser | None = None,
    ) -> None:
        super().__init__(llm)  # type: ignore[arg-type]
        self.max_retries = max_retries
        self.parser = parser or ExtractionParser()

    def _generate(self, prompt: str) -> str:
        """Call LLM synchronously across either BaseLocalLLM or LangChain LLM interface."""
        if hasattr(self.llm, "invoke"):
            return str(self.llm.invoke(prompt))
        if hasattr(self.llm, "generate"):
            return self.llm.generate(prompt)
        raise TypeError(f"Unsupported LLM instance type: {type(self.llm)}")

    async def _generate_async(self, prompt: str) -> str:
        """Call LLM asynchronously across either LangChain LLM or async lifecycle manager."""
        if hasattr(self.llm, "ainvoke"):
            res = await self.llm.ainvoke(prompt)
            return str(res)
        if hasattr(self.llm, "generate_async"):
            return await self.llm.generate_async(prompt)
        # Fallback to sync in thread
        import asyncio

        return await asyncio.to_thread(self._generate, prompt)

    def extract_clinical_conditions(
        self,
        document: ClinicalPDFDocument | str,
        doc_id: str | None = None,
    ) -> ClinicalExtractionResult:
        """Extract structured clinical conditions from a document synchronously.

        Args:
            document: ClinicalPDFDocument instance or raw clinical text.
            doc_id: Optional document identifier.

        Returns:
            ClinicalExtractionResult containing validated ExtractedClinicalCondition items.
        """
        clinical_doc: ClinicalPDFDocument | None = None
        if isinstance(document, ClinicalPDFDocument):
            clinical_doc = document
            document_id = document.document_id
            # Send targeted clinical sections rather than raw entire document
            text_to_analyze = document.get_coding_text()
            full_text = document.full_normalized_text
        else:
            document_id = doc_id or str(uuid4())
            text_to_analyze = str(document).strip()
            full_text = text_to_analyze

        if not text_to_analyze:
            return ClinicalExtractionResult(
                document_id=document_id,
                conditions=[],
                extraction_notes=["Input document text was empty."],
                repair_applied=False,
            )

        # 1. Format initial extraction prompt
        prompt = (
            f"{CLINICAL_EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"{CLINICAL_EXTRACTION_USER_TEMPLATE.format(clinical_text=text_to_analyze)}"
        )

        logger.info("Executing clinical condition extraction for doc_id=%s...", document_id)
        raw_output = ""
        conditions: list[ExtractedClinicalCondition] = []
        audit_notes: list[str] = []
        repair_applied = False

        for attempt in range(self.max_retries + 1):
            try:
                raw_output = self._generate(prompt)
                conditions, notes, repaired = self.parser.parse_and_validate(
                    raw_llm_output=raw_output,
                    document_text=full_text,
                    clinical_doc=clinical_doc,
                )
                audit_notes.extend(notes)
                if repaired:
                    repair_applied = True

                if conditions or not text_to_analyze:
                    # Successfully parsed conditions
                    break

                logger.warning(
                    "Extraction attempt %d for doc %s produced 0 validated conditions. Retrying...",
                    attempt + 1,
                    document_id,
                )
            except Exception as exc:
                logger.error("Extraction error during attempt %d: %s", attempt + 1, exc)
                audit_notes.append(f"Attempt #{attempt + 1} error: {exc}")

            # Prepare retry prompt
            prompt = (
                f"{CLINICAL_EXTRACTION_SYSTEM_PROMPT}\n\n"
                f"{CLINICAL_EXTRACTION_RETRY_TEMPLATE.format(clinical_text=text_to_analyze)}"
            )

        # 2. Resolve contradictory mentions & enforce past medical history rules
        resolved_conditions = self._reconcile_clinical_mentions(conditions)

        return ClinicalExtractionResult(
            document_id=document_id,
            conditions=resolved_conditions,
            extraction_notes=audit_notes,
            repair_applied=repair_applied,
            raw_response=raw_output,
        )

    async def extract_clinical_conditions_async(
        self,
        document: ClinicalPDFDocument | str,
        doc_id: str | None = None,
    ) -> ClinicalExtractionResult:
        """Asynchronously extract structured clinical conditions."""
        clinical_doc: ClinicalPDFDocument | None = None
        if isinstance(document, ClinicalPDFDocument):
            clinical_doc = document
            document_id = document.document_id
            text_to_analyze = document.get_coding_text()
            full_text = document.full_normalized_text
        else:
            document_id = doc_id or str(uuid4())
            text_to_analyze = str(document).strip()
            full_text = text_to_analyze

        if not text_to_analyze:
            return ClinicalExtractionResult(
                document_id=document_id,
                conditions=[],
                extraction_notes=["Input document text was empty."],
                repair_applied=False,
            )

        prompt = (
            f"{CLINICAL_EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"{CLINICAL_EXTRACTION_USER_TEMPLATE.format(clinical_text=text_to_analyze)}"
        )

        logger.info("Executing async clinical extraction for doc_id=%s...", document_id)
        raw_output = ""
        conditions: list[ExtractedClinicalCondition] = []
        audit_notes: list[str] = []
        repair_applied = False

        for attempt in range(self.max_retries + 1):
            try:
                raw_output = await self._generate_async(prompt)
                conditions, notes, repaired = self.parser.parse_and_validate(
                    raw_llm_output=raw_output,
                    document_text=full_text,
                    clinical_doc=clinical_doc,
                )
                audit_notes.extend(notes)
                if repaired:
                    repair_applied = True

                if conditions or not text_to_analyze:
                    break
            except Exception as exc:
                audit_notes.append(f"Async attempt #{attempt + 1} error: {exc}")

            prompt = (
                f"{CLINICAL_EXTRACTION_SYSTEM_PROMPT}\n\n"
                f"{CLINICAL_EXTRACTION_RETRY_TEMPLATE.format(clinical_text=text_to_analyze)}"
            )

        resolved_conditions = self._reconcile_clinical_mentions(conditions)
        return ClinicalExtractionResult(
            document_id=document_id,
            conditions=resolved_conditions,
            extraction_notes=audit_notes,
            repair_applied=repair_applied,
            raw_response=raw_output,
        )

    def _reconcile_clinical_mentions(
        self,
        conditions: list[ExtractedClinicalCondition],
    ) -> list[ExtractedClinicalCondition]:
        """Reconcile multiple or contradictory mentions of the same clinical condition.

        Rules:
        1. If a condition is suspected initially but documented as RULED_OUT later,
           the definitive RULED_OUT certainty takes precedence.
        2. If a condition is ONLY mentioned in PAST_MEDICAL_HISTORY, enforce HISTORICAL temporality.
        3. If a condition is confirmed in DISCHARGE_DIAGNOSES, enforce CURRENT temporality (unless explicitly resolved).
        """
        if not conditions:
            return []

        # Group by normalized description
        by_concept: dict[str, list[ExtractedClinicalCondition]] = {}
        for c in conditions:
            key = c.normalized_description.lower().strip()
            by_concept.setdefault(key, []).append(c)

        reconciled: list[ExtractedClinicalCondition] = []
        for _, items in by_concept.items():
            if len(items) == 1:
                item = items[0]
                # Enforce PMH rule: if only in PMH, must be HISTORICAL
                if "PAST" in item.section.upper() or "PMH" in item.section.upper():
                    item.temporality = Temporality.HISTORICAL
                    if item.status == ConditionStatus.ACUTE:
                        item.status = ConditionStatus.CHRONIC
                reconciled.append(item)
                continue

            # Multiple mentions exist (e.g. suspected in HPI, confirmed or ruled out in Discharge Diagnoses)
            has_ruled_out = any(it.certainty == Certainty.RULED_OUT for it in items)
            has_discharge_diag = any("DISCHARGE" in it.section.upper() for it in items)

            if has_ruled_out:
                # Definitive exclusion overrides suspected/possible
                ruled_out_item = next(it for it in items if it.certainty == Certainty.RULED_OUT)
                ruled_out_item.status = ConditionStatus.RESOLVED
                ruled_out_item.negation = NegationStatus.NEGATED
                reconciled.append(ruled_out_item)
            elif has_discharge_diag:
                # Discharge diagnosis is authoritative
                dd_item = next(it for it in items if "DISCHARGE" in it.section.upper())
                dd_item.temporality = Temporality.CURRENT
                reconciled.append(dd_item)
            else:
                # Keep the mention with highest certainty
                sorted_items = sorted(
                    items,
                    key=lambda x: (
                        1 if x.certainty == Certainty.CONFIRMED else 0,
                        1 if x.temporality == Temporality.CURRENT else 0,
                    ),
                    reverse=True,
                )
                reconciled.append(sorted_items[0])

        return reconciled

    # Backwards-compatibility adapter for existing LangGraph nodes
    def extract(self, clinical_text: str) -> list[ExtractedDiagnosis]:
        """Legacy interface adapting ExtractedClinicalCondition to ExtractedDiagnosis."""
        result = self.extract_clinical_conditions(clinical_text)
        diagnoses: list[ExtractedDiagnosis] = []
        for c in result.conditions:
            # Only include affirmative, non-ruled-out conditions
            if c.certainty == Certainty.RULED_OUT or c.negation == NegationStatus.NEGATED:
                continue

            span = None
            if (
                c.evidence_location.start_char is not None
                and c.evidence_location.end_char is not None
            ):
                span = TextSpan(
                    start_char=c.evidence_location.start_char,
                    end_char=c.evidence_location.end_char,
                    text=c.evidence_text,
                    page_number=c.evidence_location.page_number,
                )

            snippet = EvidenceSnippet(
                quote=c.evidence_text,
                span=span,
                source_section=c.section,
            )

            diag = ExtractedDiagnosis(
                diagnosis_id=c.condition_id,
                raw_term=c.normalized_description,
                evidence=snippet,
                extraction_confidence=c.confidence_score,
            )
            diagnoses.append(diag)

        return diagnoses

    def run(self, **kwargs: Any) -> list[ExtractedDiagnosis]:
        text = kwargs.get("clinical_text", "")
        return self.extract(text)


# Backwards compatibility alias
DiagnosisExtractionAgent = ClinicalExtractionAgent
