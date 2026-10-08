"""Context and clinical relevance agent determining inpatient coding eligibility based on documented evidence."""

import json
import re
from typing import Any
from uuid import uuid4

from medical_coding.agents.base import BaseAgent
from medical_coding.models.base import BaseLocalLLM
from medical_coding.models.langchain_llm import LocalGPT4AllLangChainLLM
from medical_coding.prompts.classification import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_TEMPLATE,
)
from medical_coding.prompts.context import (
    CONTEXT_ASSESSMENT_SYSTEM_PROMPT,
    CONTEXT_ASSESSMENT_USER_TEMPLATE,
)
from medical_coding.schemas.clinical import (
    ClassifiedDiagnosis,
    ConditionClassification,
    ContextAssessment,
    ContextualizedDiagnosis,
    EncounterClassificationResult,
    EvidenceLocation,
    EvidenceSnippet,
    ExtractedClinicalCondition,
    ExtractedDiagnosis,
)
from medical_coding.schemas.enums import (
    AbstentionReason,
    Acuity,
    Certainty,
    ConditionStatus,
    DiagnosisRole,
    NegationStatus,
    PipelineStage,
    Temporality,
)
from medical_coding.schemas.state import PipelineGraphState
from medical_coding.schemas.validation import AbstentionRecord
from medical_coding.utils.logging import get_logger
from medical_coding.validation.deterministic import (
    PrimarySecondaryClassificationValidator,
)

logger = get_logger(__name__)


class ContextAndRelevanceAgent(BaseAgent):
    """Evaluates clinical context, inpatient relevance, and coding eligibility.

    CRITICAL ARCHITECTURAL RULES:
    1. NEVER generates or invents ICD codes.
    2. Past Medical History (PMH) alone does NOT make a condition a billable secondary diagnosis.
       It must have documented clinical evaluation, therapeutic treatment, or monitoring during the stay.
    3. Ruled-out conditions cannot be coded as confirmed diagnoses.
    4. Suspected/possible conditions preserve their uncertainty; never force into confirmed.
    5. Contradictory or unresolvable documentation triggers explicit abstention.
    """

    def __init__(
        self,
        llm: BaseLocalLLM | LocalGPT4AllLangChainLLM,
        max_retries: int = 1,
    ) -> None:
        super().__init__(llm)  # type: ignore[arg-type]
        self.max_retries = max_retries

    def _generate(self, prompt: str) -> str:
        """Call LLM synchronously."""
        if hasattr(self.llm, "invoke"):
            return str(self.llm.invoke(prompt))
        if hasattr(self.llm, "generate"):
            return self.llm.generate(prompt)
        raise TypeError(f"Unsupported LLM instance type: {type(self.llm)}")

    async def _generate_async(self, prompt: str) -> str:
        """Call LLM asynchronously."""
        if hasattr(self.llm, "ainvoke"):
            res = await self.llm.ainvoke(prompt)
            return str(res)
        if hasattr(self.llm, "generate_async"):
            return await self.llm.generate_async(prompt)
        import asyncio

        return await asyncio.to_thread(self._generate, prompt)

    def run(self, **kwargs: Any) -> list[ContextAssessment]:
        """Execute context and clinical relevance assessment conforming to BaseAgent.

        Args:
            **kwargs: Accepts 'conditions' or 'diagnoses' (list of ExtractedClinicalCondition or ExtractedDiagnosis)
                      and 'clinical_text' or 'text' (str).

        Returns:
            List of evaluated ContextAssessment objects.
        """
        conditions = kwargs.get("conditions") or kwargs.get("diagnoses") or []
        clinical_text = kwargs.get("clinical_text") or kwargs.get("text") or ""

        norm_conditions: list[ExtractedClinicalCondition] = []
        for c in conditions:
            if isinstance(c, ExtractedClinicalCondition):
                norm_conditions.append(c)
            elif isinstance(c, ExtractedDiagnosis):
                norm_conditions.append(
                    ExtractedClinicalCondition(
                        condition_id=c.diagnosis_id,
                        original_mention=c.raw_term,
                        normalized_description=c.raw_term,
                        evidence_text=c.evidence.quote,
                        evidence_location=EvidenceLocation(
                            section=c.evidence.source_section or "CLINICAL_DOCUMENT",
                            page_number=1,
                        ),
                        section=c.evidence.source_section or "CLINICAL_DOCUMENT",
                    )
                )

        return self.assess_conditions(norm_conditions, clinical_text)

    def assess_conditions(
        self,
        conditions: list[ExtractedClinicalCondition],
        clinical_text: str,
    ) -> list[ContextAssessment]:
        """Assess inpatient relevance, management, and coding eligibility for each condition.

        Args:
            conditions: Collection of extracted clinical conditions.
            clinical_text: Full clinical document or targeted clinical section text.

        Returns:
            List of validated ContextAssessment instances.
        """
        if not conditions:
            return []

        conditions_summary = [
            {
                "condition_id": c.condition_id,
                "diagnosis": c.normalized_description,
                "original_mention": c.original_mention,
                "section": c.section,
                "evidence": c.evidence_text,
                "treatment_evidence": c.treatment_evidence,
                "initial_status": c.status.value,
                "initial_certainty": c.certainty.value,
                "initial_temporality": c.temporality.value,
            }
            for c in conditions
        ]

        prompt = (
            f"{CONTEXT_ASSESSMENT_SYSTEM_PROMPT}\n\n"
            f"{CONTEXT_ASSESSMENT_USER_TEMPLATE.format(clinical_text=clinical_text, conditions_json=json.dumps(conditions_summary, indent=2))}"
        )

        logger.info(
            "Executing context and clinical relevance assessment for %d conditions...",
            len(conditions),
        )

        raw_output = ""
        for attempt in range(self.max_retries + 1):
            try:
                raw_output = self._generate(prompt)
                if not raw_output or not raw_output.strip():
                    break
                parsed_assessments = self._parse_and_validate_assessments(
                    raw_llm_output=raw_output,
                    conditions=conditions,
                )
                if parsed_assessments:
                    # Apply deterministic clinical coding rules
                    return self._enforce_coding_guidelines(parsed_assessments, conditions)
            except Exception as exc:
                logger.warning("Context assessment attempt %d failed: %s", attempt + 1, exc)

        return self._fallback_rule_assessment(conditions)

    async def assess_conditions_async(
        self,
        conditions: list[ExtractedClinicalCondition],
        clinical_text: str,
    ) -> list[ContextAssessment]:
        """Asynchronously assess inpatient relevance and coding eligibility."""
        if not conditions:
            return []

        conditions_summary = [
            {
                "condition_id": c.condition_id,
                "diagnosis": c.normalized_description,
                "original_mention": c.original_mention,
                "section": c.section,
                "evidence": c.evidence_text,
                "treatment_evidence": c.treatment_evidence,
                "initial_status": c.status.value,
                "initial_certainty": c.certainty.value,
                "initial_temporality": c.temporality.value,
            }
            for c in conditions
        ]

        prompt = (
            f"{CONTEXT_ASSESSMENT_SYSTEM_PROMPT}\n\n"
            f"{CONTEXT_ASSESSMENT_USER_TEMPLATE.format(clinical_text=clinical_text, conditions_json=json.dumps(conditions_summary, indent=2))}"
        )

        logger.info("Executing async context assessment for %d conditions...", len(conditions))

        for attempt in range(self.max_retries + 1):
            try:
                raw_output = await self._generate_async(prompt)
                if not raw_output or not raw_output.strip():
                    break
                parsed_assessments = self._parse_and_validate_assessments(
                    raw_llm_output=raw_output,
                    conditions=conditions,
                )
                if parsed_assessments:
                    return self._enforce_coding_guidelines(parsed_assessments, conditions)
            except Exception as exc:
                logger.warning("Async context assessment attempt %d failed: %s", attempt + 1, exc)

        return self._fallback_rule_assessment(conditions)

    def _parse_and_validate_assessments(
        self,
        raw_llm_output: str,
        conditions: list[ExtractedClinicalCondition],
    ) -> list[ContextAssessment]:
        """Parse raw LLM JSON response and match each condition."""
        cleaned = raw_llm_output.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
            else:
                cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

        start_idx = cleaned.find("[")
        end_idx = cleaned.rfind("]")
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return []
        cleaned = cleaned[start_idx : end_idx + 1].strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                # Simple syntax repair
                repaired = re.sub(r",\s*([\]\}])", r"\1", cleaned)
                repaired = re.sub(r"(?<=[\{\s,\[])'([a-zA-Z0-9_\s\-]+)'(?=\s*:)", r'"\1"', repaired)
                repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
                data = json.loads(repaired)
            except Exception:
                return []

        if not isinstance(data, list):
            return []

        cond_by_name = {c.normalized_description.lower().strip(): c for c in conditions}
        cond_by_mention = {c.original_mention.lower().strip(): c for c in conditions}

        assessments: list[ContextAssessment] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            diag_name = str(item.get("diagnosis") or "").strip()
            if not diag_name:
                continue

            # Link back to source condition
            matched_cond = (
                cond_by_name.get(diag_name.lower())
                or cond_by_mention.get(diag_name.lower())
                or next(
                    (
                        c
                        for c in conditions
                        if diag_name.lower() in c.normalized_description.lower()
                    ),
                    None,
                )
            )

            evidence_str = str(
                item.get("evidence") or (matched_cond.evidence_text if matched_cond else "")
            ).strip()

            certainty = self._parse_certainty(item.get("certainty"))
            temporality = self._parse_temporality(item.get("temporality"))
            status = self._parse_status(item.get("status"), certainty, temporality)
            negation = self._parse_negation(item.get("negation"))

            matched_sec = matched_cond.section if matched_cond else None
            assessment = ContextAssessment(
                diagnosis=diag_name,
                condition_id=matched_cond.condition_id if matched_cond else str(uuid4()),
                section=matched_sec,
                current_relevance=bool(item.get("current_relevance", False)),
                coding_candidate=bool(item.get("coding_candidate", False)),
                status=status,
                certainty=certainty,
                temporality=temporality,
                negation=negation,
                evidence=evidence_str,
                reason=f"[{matched_sec}] {item.get('reason') or 'Evaluated against inpatient clinical relevance guidelines.'}"
                if matched_sec
                else str(
                    item.get("reason")
                    or "Evaluated against inpatient clinical relevance guidelines."
                ),
                treated_or_managed=bool(item.get("treated_or_managed", False)),
                monitored=bool(item.get("monitored", False)),
                affected_clinical_management=bool(item.get("affected_clinical_management", False)),
                influenced_treatment=bool(item.get("influenced_treatment", False)),
                treatment_evidence=item.get("treatment_evidence") or None,
                abstain_recommended=bool(item.get("abstain_recommended", False)),
                abstention_reason=None,
            )
            assessments.append(assessment)

        return assessments

    def _enforce_coding_guidelines(
        self,
        assessments: list[ContextAssessment],
        conditions: list[ExtractedClinicalCondition],
    ) -> list[ContextAssessment]:
        """Apply deterministic coding rules and guardrails to LLM assessments."""
        cond_map = {c.condition_id: c for c in conditions}
        cond_name_map = {c.normalized_description.lower().strip(): c for c in conditions}

        enforced_list: list[ContextAssessment] = []
        for asm in assessments:
            cond = cond_map.get(asm.condition_id or "") or cond_name_map.get(
                asm.diagnosis.lower().strip()
            )
            sec = cond.section.upper() if cond else ""
            evidence_lower = (asm.evidence or (cond.evidence_text if cond else "")).lower()
            reason_lower = (asm.reason or "").lower()
            treatment_ev_lower = (
                asm.treatment_evidence or (cond.treatment_evidence if cond else "") or ""
            ).lower()
            full_context = f"{evidence_lower} {reason_lower} {treatment_ev_lower}"

            # Rule 1: Ambiguous / Contradictory Mentions requiring abstention
            is_ambiguous = (
                asm.abstain_recommended
                or "contradictory" in full_context
                or "conflicting" in full_context
                or "equivocal" in full_context
                or "unclear whether" in full_context
            )
            if is_ambiguous:
                asm.abstain_recommended = True
                asm.abstention_reason = AbstentionReason.CONTRADICTORY_DOCUMENTATION
                asm.coding_candidate = False
                asm.current_relevance = False
                asm.reason = f"Contradictory or unresolvable clinical documentation: {asm.reason}"
                enforced_list.append(asm)
                continue

            # Rule 2: Ruled-Out Conditions
            if (
                asm.certainty == Certainty.RULED_OUT
                or (cond and cond.certainty == Certainty.RULED_OUT)
                or asm.negation == NegationStatus.NEGATED
                or (cond and cond.negation == NegationStatus.NEGATED)
                or "ruled out" in full_context
                or "no evidence of" in full_context
                or "negative for" in full_context
                or "excluded" in full_context
            ):
                asm.certainty = Certainty.RULED_OUT
                asm.negation = NegationStatus.NEGATED
                asm.status = ConditionStatus.RESOLVED
                asm.current_relevance = False
                asm.coding_candidate = False
                asm.reason = f"Ruled out by diagnostic evaluation: {asm.reason}"
                enforced_list.append(asm)
                continue

            # Rule 3: Past Medical History alone must NOT automatically qualify as a coding candidate
            is_pmh_only = "PAST" in sec or "PMH" in sec or asm.temporality == Temporality.HISTORICAL
            has_inpatient_care = (
                asm.treated_or_managed
                or asm.monitored
                or asm.affected_clinical_management
                or asm.influenced_treatment
                or (cond and cond.treatment_evidence is not None)
                or any(
                    term in full_context
                    for term in [
                        "treated",
                        "adjusted",
                        "insulin",
                        "sliding scale",
                        "glucose",
                        "blood sugar",
                        "continued",
                        "withheld",
                        "titrated",
                        "protocol",
                        "dose",
                        "infusion",
                        "therapy",
                        "monitored",
                        "echocardiogram",
                        "culture",
                        "stent",
                        "started on",
                        "received",
                        "medication",
                        "on medication",
                        "treatment",
                        "rx",
                    ]
                )
            )

            if is_pmh_only and not has_inpatient_care:
                asm.current_relevance = False
                asm.coding_candidate = False
                asm.temporality = Temporality.HISTORICAL
                asm.status = ConditionStatus.HISTORICAL
                asm.reason = (
                    "Documented solely in Past Medical History without active evaluation, "
                    "monitoring, or treatment during admission."
                )
                enforced_list.append(asm)
                continue

            # Rule 4: Active Treatment or Monitoring establishes clinical relevance
            if has_inpatient_care and asm.negation != NegationStatus.NEGATED:
                asm.current_relevance = True
                asm.coding_candidate = True
                asm.temporality = Temporality.CURRENT
                if asm.status == ConditionStatus.UNKNOWN:
                    asm.status = ConditionStatus.ACTIVE

            # Rule 5: Preserve uncertainty - never convert suspected/possible into confirmed
            if cond and cond.certainty in (Certainty.SUSPECTED, Certainty.POSSIBLE):
                if asm.certainty == Certainty.CONFIRMED:
                    asm.certainty = cond.certainty

            enforced_list.append(asm)

        return enforced_list

    def _fallback_rule_assessment(
        self,
        conditions: list[ExtractedClinicalCondition],
    ) -> list[ContextAssessment]:
        """Pure rule-based fallback when LLM output is unavailable or unparseable."""
        assessments: list[ContextAssessment] = []
        for c in conditions:
            evidence_lower = c.evidence_text.lower()
            sec_upper = c.section.upper()
            treatment_ev_lower = (c.treatment_evidence or "").lower()
            combined_context = f"{evidence_lower} {treatment_ev_lower}"

            # Check 1: Ambiguous / Contradictory mentions requiring abstention
            is_ambiguous = (
                "contradictory" in combined_context
                or "conflicting" in combined_context
                or "equivocal" in combined_context
                or "unclear whether" in combined_context
            )

            # Check 2: Ruled Out conditions
            is_ruled_out = (
                c.certainty == Certainty.RULED_OUT
                or c.negation == NegationStatus.NEGATED
                or "ruled out" in combined_context
                or "ruled-out" in combined_context
                or "excluded" in combined_context
            )

            # Check 3: Inpatient management indicators
            has_treatment = c.treatment_evidence is not None or any(
                re.search(rf"\b{re.escape(term)}\b", combined_context)
                for term in [
                    "treated",
                    "started on",
                    "received",
                    "adjusted",
                    "insulin",
                    "sliding scale",
                    "glucose",
                    "blood sugar",
                    "continued",
                    "withheld",
                    "titrated",
                    "protocol",
                    "antibiotic",
                    "dose",
                    "infusion",
                    "therapy",
                    "surgery",
                    "stent",
                    "replacement",
                    "medication",
                    "on medication",
                    "treatment",
                    "rx",
                ]
            )
            has_monitoring = any(
                re.search(rf"\b{re.escape(term)}\b", combined_context)
                for term in [
                    "monitored",
                    "echocardiogram",
                    "culture",
                    "ct",
                    "labs",
                    "vitals",
                    "serial",
                    "checked",
                    "workup",
                ]
            )

            # Check 4: Past Medical History
            is_pmh = (
                "PAST" in sec_upper or "PMH" in sec_upper or c.temporality == Temporality.HISTORICAL
            )

            # Check 5: Resolved
            is_resolved = (
                c.status == ConditionStatus.RESOLVED
                or c.temporality == Temporality.RESOLVED
                or "resolved" in combined_context
            )

            if is_ambiguous:
                asm = ContextAssessment(
                    diagnosis=c.normalized_description,
                    condition_id=c.condition_id,
                    section=c.section,
                    current_relevance=False,
                    coding_candidate=False,
                    status=c.status,
                    certainty=c.certainty,
                    temporality=c.temporality,
                    negation=c.negation,
                    evidence=c.evidence_text,
                    reason=f"[{c.section}] Contradictory or unresolvable clinical documentation; abstention recommended.",
                    treated_or_managed=has_treatment,
                    monitored=has_monitoring,
                    affected_clinical_management=False,
                    influenced_treatment=False,
                    treatment_evidence=c.treatment_evidence,
                    abstain_recommended=True,
                    abstention_reason=AbstentionReason.CONTRADICTORY_DOCUMENTATION,
                )
            elif is_ruled_out:
                asm = ContextAssessment(
                    diagnosis=c.normalized_description,
                    condition_id=c.condition_id,
                    section=c.section,
                    current_relevance=False,
                    coding_candidate=False,
                    status=ConditionStatus.RESOLVED,
                    certainty=Certainty.RULED_OUT,
                    temporality=Temporality.CURRENT,
                    negation=NegationStatus.NEGATED,
                    evidence=c.evidence_text,
                    reason=f"[{c.section}] Definitively ruled out during hospital stay based on diagnostic evaluation.",
                    treated_or_managed=False,
                    monitored=has_monitoring,
                    affected_clinical_management=False,
                    influenced_treatment=False,
                    treatment_evidence=c.treatment_evidence,
                )
            elif is_pmh and not (has_treatment or has_monitoring):
                asm = ContextAssessment(
                    diagnosis=c.normalized_description,
                    condition_id=c.condition_id,
                    section=c.section,
                    current_relevance=False,
                    coding_candidate=False,
                    status=ConditionStatus.HISTORICAL,
                    certainty=c.certainty,
                    temporality=Temporality.HISTORICAL,
                    negation=c.negation,
                    evidence=c.evidence_text,
                    reason=f"[{c.section}] Past Medical History alone without documented inpatient evaluation, monitoring, or therapy.",
                    treated_or_managed=False,
                    monitored=False,
                    affected_clinical_management=False,
                    influenced_treatment=False,
                    treatment_evidence=c.treatment_evidence,
                )
            elif is_pmh and (has_treatment or has_monitoring):
                asm = ContextAssessment(
                    diagnosis=c.normalized_description,
                    condition_id=c.condition_id,
                    section=c.section,
                    current_relevance=True,
                    coding_candidate=True,
                    status=ConditionStatus.CHRONIC,
                    certainty=c.certainty,
                    temporality=Temporality.CURRENT,
                    negation=c.negation,
                    evidence=c.evidence_text,
                    reason=f"[{c.section}] Pre-existing condition actively evaluated, monitored, or treated during admission.",
                    treated_or_managed=has_treatment,
                    monitored=has_monitoring,
                    affected_clinical_management=True,
                    influenced_treatment=has_treatment,
                    treatment_evidence=c.treatment_evidence,
                )
            elif is_resolved:
                asm = ContextAssessment(
                    diagnosis=c.normalized_description,
                    condition_id=c.condition_id,
                    section=c.section,
                    current_relevance=True,
                    coding_candidate=True,
                    status=ConditionStatus.RESOLVED,
                    certainty=c.certainty,
                    temporality=Temporality.CURRENT,
                    negation=c.negation,
                    evidence=c.evidence_text,
                    reason=f"[{c.section}] Condition occurred during admission and resolved following inpatient management.",
                    treated_or_managed=has_treatment,
                    monitored=has_monitoring,
                    affected_clinical_management=True,
                    influenced_treatment=has_treatment,
                    treatment_evidence=c.treatment_evidence,
                )
            else:
                # Active acute/chronic condition or suspected condition preserving uncertainty
                asm = ContextAssessment(
                    diagnosis=c.normalized_description,
                    condition_id=c.condition_id,
                    section=c.section,
                    current_relevance=True,
                    coding_candidate=True,
                    status=c.status,
                    certainty=c.certainty,  # Preserves SUSPECTED / POSSIBLE
                    temporality=Temporality.CURRENT,
                    negation=c.negation,
                    evidence=c.evidence_text,
                    reason=f"[{c.section}] Active inpatient condition evaluated, treated, or managed during current hospitalization.",
                    treated_or_managed=has_treatment,
                    monitored=has_monitoring,
                    affected_clinical_management=True,
                    influenced_treatment=has_treatment,
                    treatment_evidence=c.treatment_evidence,
                )

            assessments.append(asm)

        return assessments

    def _parse_certainty(self, val: Any) -> Certainty:
        s = str(val or "").strip().upper()
        if "RULE" in s or "EXCLUD" in s:
            return Certainty.RULED_OUT
        if "SUSPECT" in s:
            return Certainty.SUSPECTED
        if "POSSIB" in s:
            return Certainty.POSSIBLE
        if "CONFIRM" in s:
            return Certainty.CONFIRMED
        return Certainty.CONFIRMED if not s else Certainty.UNKNOWN

    def _parse_temporality(self, val: Any) -> Temporality:
        s = str(val or "").strip().upper()
        if "HISTOR" in s or "PAST" in s:
            return Temporality.HISTORICAL
        if "RESOLV" in s:
            return Temporality.RESOLVED
        if "CURRENT" in s:
            return Temporality.CURRENT
        return Temporality.CURRENT

    def _parse_status(
        self, val: Any, certainty: Certainty, temporality: Temporality
    ) -> ConditionStatus:
        if certainty == Certainty.RULED_OUT or temporality == Temporality.RESOLVED:
            return ConditionStatus.RESOLVED
        if temporality == Temporality.HISTORICAL:
            return ConditionStatus.HISTORICAL
        s = str(val or "").strip().upper()
        if "RESOLV" in s:
            return ConditionStatus.RESOLVED
        if "HISTOR" in s:
            return ConditionStatus.HISTORICAL
        if "CHRONIC" in s:
            return ConditionStatus.CHRONIC
        if "ACUTE" in s:
            return ConditionStatus.ACUTE
        return ConditionStatus.ACTIVE

    def _parse_negation(self, val: Any) -> NegationStatus:
        s = str(val or "").strip().upper()
        if "NEG" in s or "NO" in s or "DENI" in s:
            return NegationStatus.NEGATED
        return NegationStatus.AFFIRMATIVE

    # LangGraph Node execution method
    def run_node(self, state: PipelineGraphState) -> dict[str, Any]:
        """LangGraph node execution updating graph state with contextualized diagnoses."""
        raw_text = state.get("raw_text", "")
        extracted_conditions = state.get("extracted_conditions", [])
        extracted_diagnoses = state.get("extracted_diagnoses", [])

        # Prefer extracted_conditions if available, otherwise convert legacy extracted_diagnoses
        conditions: list[ExtractedClinicalCondition] = []
        if extracted_conditions:
            conditions = list(extracted_conditions)
        elif extracted_diagnoses:
            for diag in extracted_diagnoses:
                conditions.append(
                    ExtractedClinicalCondition(
                        condition_id=diag.diagnosis_id,
                        original_mention=diag.raw_term,
                        normalized_description=diag.raw_term,
                        evidence_text=diag.evidence.quote,
                        evidence_location=EvidenceLocation(
                            section=diag.evidence.source_section or "CLINICAL_DOCUMENT",
                            page_number=1,
                        ),
                        section=diag.evidence.source_section or "CLINICAL_DOCUMENT",
                    )
                )

        assessments = self.assess_conditions(conditions, raw_text)

        # Build ContextualizedDiagnosis models for downstream compatibility
        contextualized: list[ContextualizedDiagnosis] = []
        abstentions: list[AbstentionRecord] = []

        for asm in assessments:
            if asm.abstain_recommended:
                abstentions.append(
                    AbstentionRecord(
                        diagnosis_id=asm.condition_id,
                        raw_term=asm.diagnosis,
                        reason=asm.abstention_reason
                        or AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE,
                        detail=asm.reason,
                        stage=PipelineStage.CONTEXT_ANALYSIS,
                    )
                )

            contextualized.append(
                ContextualizedDiagnosis(
                    diagnosis_id=asm.condition_id or str(uuid4()),
                    raw_term=asm.diagnosis,
                    evidence=EvidenceSnippet(
                        quote=asm.evidence,
                        source_section="CONTEXT_ASSESSMENT",
                    ),
                    extraction_confidence=1.0 if not asm.abstain_recommended else 0.4,
                    negation=asm.negation,
                    temporality=asm.temporality,
                    certainty=asm.certainty,
                    acuity=Acuity.ACUTE
                    if asm.status == ConditionStatus.ACUTE
                    else Acuity.CHRONIC
                    if asm.status == ConditionStatus.CHRONIC
                    else Acuity.UNSPECIFIED,
                    clinical_justification=asm.reason,
                )
            )

        return {
            "context_assessments": assessments,
            "contextualized_diagnoses": contextualized,
            "abstentions": abstentions,
            "current_stage": PipelineStage.CONTEXT_ANALYSIS,
        }


class PrimarySecondaryClassifier(BaseAgent):
    """Primary vs Secondary diagnosis classifier adhering to official UHDDS inpatient rules.

    INPUT CONSTRAINT:
    Operates strictly on clinically extracted and context-assessed conditions.
    Does NOT classify directly from raw document text unless necessary for resolving an evidence ambiguity.

    CRITICAL HARD RULES:
    1. Maximum ONE Primary diagnosis per encounter.
    2. Zero or more Secondary diagnoses (must be active, clinically relevant, managed, or monitored).
    3. Excluded conditions (historical without care, ruled out, negated) are not billable.
    4. Never select the first diagnosis mentioned merely because of document order.
    5. If two or more conditions equally qualify as primary without documented distinction, do NOT arbitrarily choose one; trigger abstention.
    6. If no defensible primary condition exists, do NOT force one.
    7. All classifications are deterministically verified by PrimarySecondaryClassificationValidator.
    """

    def __init__(
        self,
        llm: BaseLocalLLM | LocalGPT4AllLangChainLLM,
        max_retries: int = 1,
    ) -> None:
        super().__init__(llm)  # type: ignore[arg-type]
        self.max_retries = max_retries
        self.validator = PrimarySecondaryClassificationValidator()

    def _generate(self, prompt: str) -> str:
        """Call LLM synchronously."""
        if hasattr(self.llm, "invoke"):
            return str(self.llm.invoke(prompt))
        if hasattr(self.llm, "generate"):
            return self.llm.generate(prompt)
        raise TypeError(f"Unsupported LLM instance type: {type(self.llm)}")

    async def _generate_async(self, prompt: str) -> str:
        """Call LLM asynchronously."""
        if hasattr(self.llm, "ainvoke"):
            res = await self.llm.ainvoke(prompt)
            return str(res)
        if hasattr(self.llm, "generate_async"):
            return await self.llm.generate_async(prompt)
        import asyncio

        return await asyncio.to_thread(self._generate, prompt)

    def classify_conditions(
        self,
        assessments: list[ContextAssessment],
        clinical_text: str = "",
        document_id: str | None = None,
    ) -> tuple[EncounterClassificationResult, list[AbstentionRecord]]:
        """Classify context-assessed conditions into Primary, Secondary, and Excluded roles.

        Args:
            assessments: Clinically assessed conditions with relevance and management flags.
            clinical_text: Clinical text used only for evidence verification / ambiguity resolution.
            document_id: Optional encounter identifier.

        Returns:
            Tuple of (EncounterClassificationResult, list of AbstentionRecords).
        """
        if not assessments:
            res = EncounterClassificationResult(
                document_id=document_id,
                has_unique_primary=False,
                abstention_recommended=True,
                abstention_reason=AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE,
                audit_notes=["No conditions provided for classification."],
            )
            return res, []

        # Step 1: Attempt LLM classification if LLM is available and valid
        llm_classifications: list[ConditionClassification] | None = None
        prompt = (
            f"{CLASSIFICATION_SYSTEM_PROMPT}\n\n"
            f"{CLASSIFICATION_USER_TEMPLATE.format(conditions_json=json.dumps([a.model_dump() for a in assessments], indent=2), clinical_text=clinical_text)}"
        )

        for attempt in range(self.max_retries + 1):
            try:
                raw_output = self._generate(prompt)
                if not raw_output or not raw_output.strip():
                    break
                parsed = self._parse_llm_classification(raw_output, assessments)
                if parsed:
                    llm_classifications = parsed
                    break
            except Exception as exc:
                logger.warning("LLM classification attempt %d failed: %s", attempt + 1, exc)

        # Step 2: Fall back to deterministic rule scoring if LLM output is missing
        if not llm_classifications:
            llm_classifications = self._score_and_classify_rules(assessments, clinical_text=clinical_text)

        # Step 3: Enforce deterministic validation guardrails (rejects multiple primaries, etc.)
        validated_result, abstentions = self.validator.validate_classification(
            classifications=llm_classifications,
            context_assessments=assessments,
            document_id=document_id,
        )

        return validated_result, abstentions

    async def classify_conditions_async(
        self,
        assessments: list[ContextAssessment],
        clinical_text: str = "",
        document_id: str | None = None,
    ) -> tuple[EncounterClassificationResult, list[AbstentionRecord]]:
        """Asynchronously classify conditions into Primary, Secondary, and Excluded roles."""
        if not assessments:
            res = EncounterClassificationResult(
                document_id=document_id,
                has_unique_primary=False,
                abstention_recommended=True,
                abstention_reason=AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE,
                audit_notes=["No conditions provided for classification."],
            )
            return res, []

        llm_classifications: list[ConditionClassification] | None = None
        prompt = (
            f"{CLASSIFICATION_SYSTEM_PROMPT}\n\n"
            f"{CLASSIFICATION_USER_TEMPLATE.format(conditions_json=json.dumps([a.model_dump() for a in assessments], indent=2), clinical_text=clinical_text)}"
        )

        for attempt in range(self.max_retries + 1):
            try:
                raw_output = await self._generate_async(prompt)
                parsed = self._parse_llm_classification(raw_output, assessments)
                if parsed:
                    llm_classifications = parsed
                    break
            except Exception as exc:
                logger.warning("Async LLM classification attempt %d failed: %s", attempt + 1, exc)

        if not llm_classifications:
            llm_classifications = self._score_and_classify_rules(assessments, clinical_text=clinical_text)

        return self.validator.validate_classification(
            classifications=llm_classifications,
            context_assessments=assessments,
            document_id=document_id,
        )

    def _score_and_classify_rules(
        self,
        assessments: list[ContextAssessment],
        clinical_text: str = "",
    ) -> list[ConditionClassification]:
        """Deterministic rule-based admitting score calculation and role classification."""
        scored_candidates: list[tuple[ContextAssessment, float, list[str]]] = []
        classifications: list[ConditionClassification] = []
        doc_lower = (clinical_text or "").lower()

        from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

        for asm in assessments:
            diag_lower = asm.diagnosis.lower()
            ev_lower = (asm.evidence or "").lower()
            treat_lower = (asm.treatment_evidence or "").lower()
            reason_lower = (asm.reason or "").lower()
            sec_lower = (getattr(asm, "section", "") or "").lower().replace("_", " ")
            full_context = f"{diag_lower} {ev_lower} {treat_lower} {reason_lower} {sec_lower}"

            # Gate: Hard clinical candidate check (reject absence statements, instructions, medications)
            is_valid_diag, gate_reason = HardClinicalCandidateGate.evaluate_candidate(asm.diagnosis, ev_lower)

            # Check eligibility
            is_excluded = (
                not is_valid_diag
                or not asm.coding_candidate
                or asm.certainty == Certainty.RULED_OUT
                or asm.negation == NegationStatus.NEGATED
                or asm.temporality == Temporality.HISTORICAL
                or asm.status == ConditionStatus.HISTORICAL
            )

            if is_excluded:
                excl_reason = f"Clinical gate rejection: {gate_reason}" if not is_valid_diag else f"Excluded from coding: {asm.reason}"
                classifications.append(
                    ConditionClassification(
                        diagnosis_id=asm.condition_id or str(uuid4()),
                        diagnosis=asm.diagnosis,
                        role=DiagnosisRole.EXCLUDED,
                        is_billable_candidate=False,
                        classification_reason=excl_reason,
                        admitting_condition_score=-100.0,
                        evidence_quote=asm.evidence,
                    )
                )
                continue

            # Calculate Admitting Significance Score for active candidates
            score = 0.0
            reasons: list[str] = []

            # 1. Section / Reason for Admission Context (UHDDS Authority Hierarchy)
            # Under UHDDS, condition established after study chiefly responsible for occasioning admission
            # or underlying condition occasioning procedural/medical inpatient therapy
            # Function to test whether diag_lower is the direct primary diagnosis mention in a clause
            def _check_primary_match(text: str, target_diag: str) -> bool:
                for match in re.finditer(rf"\b(?:principal|primary)(?:\s+diagnos[ei]s)?\s*[:\-–—]\s*([^\n.;]+)", text, re.IGNORECASE):
                    clause = match.group(1).lower().strip()
                    # Strip subsequent conditions attached by "with", "complicated by", "secondary to", ","
                    for delim in [" with ", " complicated by ", " secondary to ", ","]:
                        if delim in clause:
                            clause = clause[:clause.find(delim)].strip()
                    if re.search(rf"\b{re.escape(target_diag)}\b", clause):
                        return True
                return False

            def _check_secondary_match(text: str, target_diag: str) -> bool:
                for match in re.finditer(rf"\bsecondary(?:\s+diagnos[ei]s)?\s*[:\-–—]\s*([^\n.;]+)", text, re.IGNORECASE):
                    clause = match.group(1).lower().strip()
                    if re.search(rf"\b{re.escape(target_diag)}\b", clause):
                        return True
                return False

            is_explicit_primary = bool(
                _check_primary_match(full_context, diag_lower)
                or (doc_lower and _check_primary_match(doc_lower, diag_lower))
                or sec_lower in ("principal diagnosis", "primary diagnosis", "principal diagnoses", "primary diagnoses")
                or getattr(asm, "role", None) in (DiagnosisRole.PRIMARY, "PRIMARY")
                or getattr(asm, "is_primary", False)
            )
            is_explicit_secondary = bool(
                _check_secondary_match(full_context, diag_lower)
                or (doc_lower and _check_secondary_match(doc_lower, diag_lower))
                or sec_lower in ("secondary diagnosis", "secondary diagnoses", "additional diagnoses")
            )
            is_admission_driver = False
            adm_regex = rf"\b(?:admitted\s+(?:for|with|to|in)|reason\s+for\s+admission|admitting\s+diagnosis|principal\s+diagnosis)\b[^\n\.\;]*?(?:due\s+to\s+|for\s+|of\s+|in\s+)?{re.escape(diag_lower)}"
            if re.search(adm_regex, full_context) or (doc_lower and re.search(adm_regex, doc_lower)):
                is_admission_driver = True
            else:
                # Generalized clause matching: admission occasioning phrase matching clinical tokens or acronym
                adm_clause_match = re.search(
                    r"\b(?:admitted\s+(?:for|with|to|in)|reason\s+for\s+admission|admitting\s+diagnosis|presenting\s+complaint)\b\s*([^\n.;]+)",
                    full_context if "admitted" in full_context else (doc_lower or ""),
                )
                if adm_clause_match:
                    adm_clause = adm_clause_match.group(0).lower()
                    diag_words = [w for w in re.findall(r"\b[a-z]{3,}\b", diag_lower) if w not in ("type", "with", "acute", "chronic", "left", "right", "bilateral", "unspecified", "stage", "organism")]
                    from medical_coding.retrieval.tokenizer import CLINICAL_MORPHOLOGY
                    expanded_dwords = set(diag_words)
                    for dw in diag_words:
                        expanded_dwords.update(CLINICAL_MORPHOLOGY.get(dw, []))
                    if (
                        re.search(rf"\b{re.escape(diag_lower)}\b", adm_clause)
                        or any(re.search(rf"\b{re.escape(edw)}\b", adm_clause) for edw in expanded_dwords)
                    ):
                        is_admission_driver = True

            if is_explicit_primary:
                score += 10.0
                reasons.append("Explicitly documented as PRIMARY/PRINCIPAL DIAGNOSIS by provider (+10.0)")
            elif is_admission_driver:
                score += 6.0
                reasons.append("Identified as principal condition occasioning admission (+6.0)")

            if is_explicit_secondary and not is_explicit_primary:
                score -= 5.0
                reasons.append("Explicitly documented as SECONDARY DIAGNOSIS by provider (-5.0)")

            in_final_sec = any(
                term in sec_lower
                for term in [
                    "principal diagnosis",
                    "primary diagnosis",
                    "discharge diagnosis",
                    "discharge diagnoses",
                    "final diagnosis",
                    "final diagnoses",
                    "final coding summary",
                    "coding summary",
                ]
            ) or getattr(asm, "section", "") in ("PRINCIPAL_DIAGNOSIS", "PRIMARY_DIAGNOSIS", "DISCHARGE_DIAGNOSES", "FINAL_DIAGNOSES")

            if in_final_sec:
                score += 5.0
                reasons.append("Explicitly documented under discharge / final diagnoses (+5.0)")
            elif any(
                term in sec_lower or term in full_context
                for term in [
                    "chief complaint",
                    "reason for admission",
                    "admitted for",
                    "admitted with",
                    "presenting complaint",
                    "presented with acute",
                    "admitting diagnosis",
                ]
            ):
                score += 3.5
                reasons.append(
                    "Documented as chief complaint or direct reason for admission (+3.5)"
                )

            if any(term in full_context for term in ["history of present illness", "hpi"]):
                score += 2.0
                reasons.append("Detailed in history of present illness (+2.0)")

            # Secondary / Historical Section Hierarchy:
            if sec_lower in ("past medical history", "personal history", "pmh", "medical history"):
                score -= 10.0
                reasons.append("Documented under past/personal history (-10.0)")

            # Complication vs Primary Etiology distinction (e.g. Acute pyelonephritis complicated by septic shock):
            is_complication = bool(
                re.search(rf"\b(?:complicated\s+by)\b[^\n.;]*?{re.escape(diag_lower)}", full_context)
                or (doc_lower and re.search(rf"\b(?:complicated\s+by)\b[^\n.;]*?{re.escape(diag_lower)}", doc_lower))
                or "complicated by" in (asm.reason or "").lower()
            )
            is_leading_primary = bool(
                is_explicit_primary and not is_complication and (
                    re.search(rf"\b{re.escape(diag_lower)}\s+(?:with|complicated\s+by)\b", ev_lower)
                    or (doc_lower and re.search(rf"\b{re.escape(diag_lower)}\s+(?:with|complicated\s+by)\b", doc_lower))
                )
            )
            if is_complication:
                score -= 2.0
                reasons.append("Documented as secondary complication rather than underlying primary disease (-2.0)")
            elif is_leading_primary:
                score += 2.0
                reasons.append("Underlying occasioning primary condition prior to complications (+2.0)")

            # Injury coding precedence: skeletal fractures outrank co-occurring ligament sprains/strains (CMS Guideline 19.b)
            if any(sp in diag_lower for sp in ["sprain", "strain"]) and any("fracture" in c.diagnosis.lower() for c in assessments):
                score -= 2.0
                reasons.append("Ligament sprain secondary to co-occurring skeletal fracture (-2.0)")

            # 2. Major Interventions & Procedures
            if any(
                term in full_context
                for term in [
                    "stent",
                    "catheterization",
                    "appendectomy",
                    "cholecystectomy",
                    "surgery",
                    "operative",
                    "resection",
                    "intubation",
                    "cardioversion",
                    "thrombectomy",
                    "angioplasty",
                    "bypass",
                    "procedure",
                    "intervention",
                ]
            ):
                score += 3.5
                reasons.append("Primary procedural or operative intervention performed (+3.5)")
            elif any(
                term in full_context
                for term in [
                    "iv ",
                    "infusion",
                    "ceftriaxone",
                    "furosemide",
                    "nebulized",
                    "antibiotic",
                ]
            ):
                score += 2.0
                reasons.append(
                    "Active intravenous / intensive therapeutic management administered (+2.0)"
                )
            elif asm.treated_or_managed:
                score += 1.5
                reasons.append("Actively managed and treated during hospitalization (+1.5)")

            # 3. Acuity & Certainty
            is_acute_cond = (
                asm.status == ConditionStatus.ACUTE
                or getattr(asm, "acuity", None) in (Acuity.ACUTE, "ACUTE")
                or any(w in diag_lower for w in ["acute", "sepsis", "septic", "infarction", "stemi", "nstemi"])
            )
            if is_acute_cond:
                score += 2.0
                reasons.append("Acute presentation (+2.0)")
            if asm.certainty == Certainty.CONFIRMED:
                score += 1.0
                reasons.append("Confirmed diagnostic certainty (+1.0)")
            elif asm.certainty in (Certainty.SUSPECTED, Certainty.POSSIBLE):
                score += 0.5
                reasons.append("Suspected condition evaluated at discharge (+0.5)")

            # Signs/symptoms or manifestations without definitive etiologic status (CMS Guideline I.B.4)
            is_symptom = any(
                sym in diag_lower
                for sym in ["symptom", "pain", "fatigue", "edema", "overload", "dyspnea", "shortness of breath", "nausea", "vomiting", "weakness", "fever", "cough", "dyspepsia", "discomfort", "wheezing"]
            )
            has_definitive_pathology = (
                any(diag_lower.endswith(sfx) or f"{sfx} " in diag_lower for sfx in ("itis", "oma", "osis"))
                or any(dx in diag_lower for dx in ["syndrome", "failure", "infarction", "disease", "disorder", "calculus", "lithiasis", "ulcer"])
            )
            if is_symptom and not has_definitive_pathology:
                score -= 2.0
                reasons.append("Symptom / manifestation accompanying presentation (-2.0)")

            # Infectious organism supplementary to underlying organ pathology (CMS Guideline I.C.1)
            # Sepsis is a systemic life-threatening syndrome, not a supplementary B95-B97 organism code
            is_organism = (
                "sepsis" not in diag_lower
                and any(
                    org in diag_lower
                    for org in ["helicobacter", "h. pylori", "h pylori", "infectious agent", "bacterium", "bacteria", "streptococcus", "staphylococcus", "bacillus"]
                )
            )
            if is_organism:
                score -= 2.0
                reasons.append("Supplementary etiologic organism code secondary to primary organ pathology (-2.0)")

            # Chronic baseline conditions without acute exacerbation
            is_chronic = (
                asm.status == ConditionStatus.CHRONIC
                or "chronic" in diag_lower
                or any(m in full_context for m in ["home medication", "home regimen", "continued on", "baseline", "longstanding", "routine control", "past history", "history of"])
            )
            if is_chronic and "acute" not in full_context and not is_explicit_primary and not is_admission_driver:
                score -= 1.5
                reasons.append("Chronic background condition without acute exacerbation (-1.5)")

            scored_candidates.append((asm, score, reasons))

        # Determine Primary vs Secondary
        if not scored_candidates:
            return classifications

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Check for ambiguous tie between two high-acuity admission drivers (Scenario 4)
        has_ambiguous_tie = False
        if len(scored_candidates) >= 2:
            s0 = scored_candidates[0][1]
            s1 = scored_candidates[1][1]
            asm0 = scored_candidates[0][0]
            asm1 = scored_candidates[1][0]
            if (
                s0 >= 4.0
                and s1 >= 4.0
                and abs(s0 - s1) < 0.01
                and asm0.status == ConditionStatus.ACUTE
                and asm1.status == ConditionStatus.ACUTE
            ):
                has_ambiguous_tie = True

        if has_ambiguous_tie:
            # Emit both as PRIMARY so deterministic validator can enforce UHDDS ambiguity demotion
            top_asm, top_score, top_reasons = scored_candidates[0]
            sec_asm, sec_score, sec_reasons = scored_candidates[1]
            classifications.append(
                ConditionClassification(
                    diagnosis_id=top_asm.condition_id or str(uuid4()),
                    diagnosis=top_asm.diagnosis,
                    role=DiagnosisRole.PRIMARY,
                    is_billable_candidate=True,
                    classification_reason=f"Ambiguous co-primary diagnosis: '{top_asm.diagnosis}' competed with other primary candidates (score={top_score:.1f})",
                    primary_justification=f"Condition occasioning admission: {top_asm.evidence}",
                    admitting_condition_score=top_score,
                    evidence_quote=top_asm.evidence,
                )
            )
            classifications.append(
                ConditionClassification(
                    diagnosis_id=sec_asm.condition_id or str(uuid4()),
                    diagnosis=sec_asm.diagnosis,
                    role=DiagnosisRole.PRIMARY,
                    is_billable_candidate=True,
                    classification_reason=f"Ambiguous co-primary diagnosis: '{sec_asm.diagnosis}' competed with other primary candidates (score={sec_score:.1f})",
                    primary_justification=f"Condition occasioning admission: {sec_asm.evidence}",
                    admitting_condition_score=sec_score,
                    evidence_quote=sec_asm.evidence,
                )
            )
            for asm, score, r_list in scored_candidates[2:]:
                classifications.append(
                    ConditionClassification(
                        diagnosis_id=asm.condition_id or str(uuid4()),
                        diagnosis=asm.diagnosis,
                        role=DiagnosisRole.SECONDARY,
                        is_billable_candidate=True,
                        classification_reason=f"Co-existing condition managed during admission (score={score:.1f}): {'; '.join(r_list)}",
                        admitting_condition_score=score,
                        evidence_quote=asm.evidence,
                    )
                )
            return classifications

        # Select the top candidate that meets Section 8 primary criteria
        def _is_pure_symptom(cand_asm: ContextAssessment) -> bool:
            d_lower = cand_asm.diagnosis.lower()
            is_sym = any(
                sym in d_lower
                for sym in [
                    "symptom", "pain", "fatigue", "edema", "overload", "dyspnea",
                    "shortness of breath", "nausea", "vomiting", "weakness", "fever",
                    "cough", "dyspepsia", "discomfort", "wheezing", "respiratory distress",
                ]
            )
            has_def = (
                any(d_lower.endswith(sfx) or f"{sfx} " in d_lower for sfx in ("itis", "oma", "osis"))
                or any(dx in d_lower for dx in ["syndrome", "failure", "infarction", "disease", "disorder", "calculus", "lithiasis", "ulcer", "pneumonia"])
            )
            return is_sym and not has_def

        has_definitive_candidates = any(
            not _is_pure_symptom(c[0]) for c in scored_candidates if c[1] > 1.5
        )

        chosen_primary_idx = None
        for i, (asm, s, _r_list) in enumerate(scored_candidates):
            if (
                s > 1.5
                and asm.certainty in (Certainty.CONFIRMED, Certainty.SUSPECTED, Certainty.POSSIBLE, "CONFIRMED", "SUSPECTED", "POSSIBLE")
                and asm.temporality in (Temporality.CURRENT, "CURRENT")
                and bool(asm.evidence and asm.evidence.strip())
            ):
                if has_definitive_candidates and _is_pure_symptom(asm):
                    continue
                is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(asm.diagnosis, asm.evidence or "")
                if is_valid:
                    chosen_primary_idx = i
                    break

        # Fallback if no candidate exceeded s > 1.5: pick highest-scoring valid active candidate
        if chosen_primary_idx is None:
            for i, (asm, _s, _r_list) in enumerate(scored_candidates):
                if (
                    asm.temporality in (Temporality.CURRENT, "CURRENT")
                    and bool(asm.evidence and asm.evidence.strip())
                ):
                    is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(asm.diagnosis, asm.evidence or "")
                    if is_valid:
                        chosen_primary_idx = i
                        break

        if chosen_primary_idx is not None:
            top_asm, top_score, top_reasons = scored_candidates[chosen_primary_idx]
            remaining_candidates = [
                item for j, item in enumerate(scored_candidates) if j != chosen_primary_idx
            ]
        else:
            top_asm, top_score, top_reasons = scored_candidates[0]
            remaining_candidates = scored_candidates[1:]

        if chosen_primary_idx is not None:
            # Primary selected (either unique or prioritized under UHDDS Section II.C)
            top_ev_quote = (top_asm.evidence or "").strip()
            pri_reason = (
                f"Designated as Primary Diagnosis under UHDDS Guidelines (chief condition established after study to be responsible for occasioning admission). "
                f"Documented evidence: \"{top_ev_quote}\". "
                f"Clinical criteria: {'; '.join(top_reasons)}."
            )
            pri_just = (
                f"Chief condition occasioning admission and inpatient care: \"{top_ev_quote}\"."
            )
            classifications.append(
                ConditionClassification(
                    diagnosis_id=top_asm.condition_id or str(uuid4()),
                    diagnosis=top_asm.diagnosis,
                    role=DiagnosisRole.PRIMARY,
                    is_billable_candidate=True,
                    classification_reason=pri_reason,
                    primary_justification=pri_just,
                    admitting_condition_score=top_score,
                    evidence_quote=top_asm.evidence,
                )
            )

            # Remaining active candidates become SECONDARY
            # CMS Guideline I.B.4: Signs and symptoms that are integral to a definitive primary disease process
            # should not be assigned as separate secondary codes.
            for asm, score, r_list in remaining_candidates:
                sec_lower = asm.diagnosis.lower()
                is_integral = any(
                    sym in sec_lower
                    for sym in [
                        "pain", "colic", "fever", "cough", "dyspnea", "shortness of breath",
                        "nausea", "vomiting", "dyspepsia", "indigestion", "heartburn",
                        "discomfort", "wheezing", "fatigue", "malaise", "respiratory distress"
                    ]
                ) and not (
                    any(sec_lower.endswith(sfx) or f"{sfx} " in sec_lower for sfx in ("itis", "oma", "osis"))
                    or any(dx in sec_lower for dx in ["syndrome", "failure", "infarction", "disease", "disorder", "calculus", "lithiasis", "ulcer"])
                )

                if is_integral:
                    classifications.append(
                        ConditionClassification(
                            diagnosis_id=asm.condition_id or str(uuid4()),
                            diagnosis=asm.diagnosis,
                            role=DiagnosisRole.EXCLUDED,
                            is_billable_candidate=False,
                            classification_reason=f"Symptom integral to the primary diagnosis '{top_asm.diagnosis}'; excluded under CMS Guideline I.B.4: \"{(asm.evidence or '').strip()}\".",
                            admitting_condition_score=score,
                            evidence_quote=asm.evidence,
                        )
                    )
                    continue

                is_valid_sec, sec_gate_reason = HardClinicalCandidateGate.evaluate_candidate(asm.diagnosis, asm.evidence or "")
                if not is_valid_sec:
                    classifications.append(
                        ConditionClassification(
                            diagnosis_id=asm.condition_id or str(uuid4()),
                            diagnosis=asm.diagnosis,
                            role=DiagnosisRole.EXCLUDED,
                            is_billable_candidate=False,
                            classification_reason=f"Clinical gate rejection: {sec_gate_reason}",
                            admitting_condition_score=-100.0,
                            evidence_quote=asm.evidence,
                        )
                    )
                    continue

                sec_ev_quote = (asm.evidence or "").strip()
                sec_reason = (
                    f"Designated as Secondary (Comorbid) Diagnosis under UHDDS Guidelines. "
                    f"Documented co-existing condition active during stay: \"{sec_ev_quote}\". "
                    f"Clinical management: {'; '.join(r_list)}."
                )
                classifications.append(
                    ConditionClassification(
                        diagnosis_id=asm.condition_id or str(uuid4()),
                        diagnosis=asm.diagnosis,
                        role=DiagnosisRole.SECONDARY,
                        is_billable_candidate=True,
                        classification_reason=sec_reason,
                        admitting_condition_score=score,
                        evidence_quote=asm.evidence,
                    )
                )
        else:
            # No defensible primary (all scores too low or non-qualifying)
            for asm, score, r_list in scored_candidates:
                classifications.append(
                    ConditionClassification(
                        diagnosis_id=asm.condition_id or str(uuid4()),
                        diagnosis=asm.diagnosis,
                        role=DiagnosisRole.SECONDARY,
                        is_billable_candidate=True,
                        classification_reason=f"Active condition managed during stay, but does not meet criteria for primary diagnosis: {'; '.join(r_list)}",
                        admitting_condition_score=score,
                        evidence_quote=asm.evidence,
                    )
                )

        return classifications

    def _parse_llm_classification(
        self,
        raw_output: str,
        assessments: list[ContextAssessment],
    ) -> list[ConditionClassification] | None:
        """Parse raw LLM output into ConditionClassification objects."""
        cleaned = raw_output.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
            else:
                cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            return None

        items: list[dict[str, Any]] = []
        if isinstance(data, dict):
            items = data.get("classifications", [])
        elif isinstance(data, list):
            items = data

        if not items:
            return None

        asm_map = {a.diagnosis.lower().strip(): a for a in assessments}
        asm_id_map = {a.condition_id: a for a in assessments if a.condition_id}

        classifications: list[ConditionClassification] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            diag_name = str(it.get("diagnosis") or "").strip()
            diag_id = str(it.get("diagnosis_id") or "").strip()

            matched_asm = (
                asm_id_map.get(diag_id)
                or asm_map.get(diag_name.lower())
                or next((a for a in assessments if diag_name.lower() in a.diagnosis.lower()), None)
            )

            role_str = str(it.get("role") or "").strip().upper()
            role = DiagnosisRole.SECONDARY
            if "PRIM" in role_str:
                role = DiagnosisRole.PRIMARY
            elif "EXCL" in role_str or "NOT" in role_str:
                role = DiagnosisRole.EXCLUDED

            classifications.append(
                ConditionClassification(
                    diagnosis_id=matched_asm.condition_id
                    if matched_asm
                    else (diag_id or str(uuid4())),
                    diagnosis=matched_asm.diagnosis if matched_asm else diag_name,
                    role=role,
                    is_billable_candidate=bool(
                        it.get("is_billable_candidate", role != DiagnosisRole.EXCLUDED)
                    ),
                    classification_reason=str(
                        it.get("classification_reason") or "LLM classification assessment."
                    ),
                    primary_justification=it.get("primary_justification"),
                    evidence_quote=matched_asm.evidence if matched_asm else "",
                )
            )

        return classifications if len(classifications) > 0 else None

    # LangGraph Node execution
    def run_node(self, state: PipelineGraphState) -> dict[str, Any]:
        """LangGraph node execution updating graph state with classified diagnoses."""
        raw_text = state.get("raw_text", "")
        doc_id = state.get("document_id")
        context_assessments = state.get("context_assessments", [])

        # If context_assessments missing, adapt from contextualized_diagnoses
        if not context_assessments:
            contextualized = state.get("contextualized_diagnoses", [])
            for c in contextualized:
                is_excl = (
                    c.negation == NegationStatus.NEGATED
                    or c.certainty == Certainty.RULED_OUT
                    or c.temporality == Temporality.HISTORICAL
                )
                sec = getattr(c.evidence, "source_section", None) if c.evidence else None
                context_assessments.append(
                    ContextAssessment(
                        condition_id=c.diagnosis_id,
                        diagnosis=c.raw_term,
                        section=sec,
                        current_relevance=not is_excl,
                        coding_candidate=not is_excl,
                        status=ConditionStatus.HISTORICAL
                        if c.temporality == Temporality.HISTORICAL
                        else ConditionStatus.ACTIVE,
                        certainty=c.certainty,
                        temporality=c.temporality,
                        negation=c.negation,
                        evidence=c.evidence.quote,
                        reason=f"[{sec}] {c.clinical_justification}" if sec else c.clinical_justification,
                        treated_or_managed=not is_excl,
                    )
                )

        enc_result, abstentions = self.classify_conditions(
            assessments=context_assessments,
            clinical_text=raw_text,
            document_id=doc_id,
        )

        # Build ClassifiedDiagnosis models for backward compatibility
        classified_diagnoses: list[ClassifiedDiagnosis] = []
        for c in enc_result.all_classifications:
            dummy_ctx = ContextualizedDiagnosis(
                diagnosis_id=c.diagnosis_id,
                raw_term=c.diagnosis,
                evidence=EvidenceSnippet(
                    quote=c.evidence_quote or "Classified diagnosis evidence",
                    source_section="CLASSIFICATION",
                ),
                extraction_confidence=1.0,
                negation=NegationStatus.AFFIRMATIVE,
                temporality=Temporality.CURRENT
                if c.role != DiagnosisRole.EXCLUDED
                else Temporality.HISTORICAL,
                certainty=Certainty.CONFIRMED,
                acuity=Acuity.ACUTE if c.role == DiagnosisRole.PRIMARY else Acuity.UNSPECIFIED,
                clinical_justification=c.classification_reason,
            )
            classified_diagnoses.append(
                ClassifiedDiagnosis(
                    diagnosis_id=c.diagnosis_id,
                    raw_term=c.diagnosis,
                    context=dummy_ctx,
                    role=c.role,
                    is_billable_candidate=c.is_billable_candidate,
                    classification_reason=c.classification_reason,
                    primary_justification=c.primary_justification,
                )
            )

        return {
            "classified_diagnoses": classified_diagnoses,
            "classification_result": enc_result,
            "abstentions": abstentions,
            "current_stage": PipelineStage.CLASSIFICATION,
        }

    def run(self, **kwargs: Any) -> EncounterClassificationResult:
        """Execute classification conforming to BaseAgent."""
        assessments = kwargs.get("assessments") or kwargs.get("conditions") or []
        clinical_text = kwargs.get("clinical_text") or kwargs.get("text") or ""
        doc_id = kwargs.get("document_id")
        result, _ = self.classify_conditions(assessments, clinical_text, doc_id)
        return result


class ClassificationAgent(PrimarySecondaryClassifier):
    """Backwards-compatible agent supporting legacy analyze_context and classify_roles signatures."""

    def __init__(self, llm: BaseLocalLLM) -> None:
        super().__init__(llm)
        self.context_system_prompt = CONTEXT_ASSESSMENT_SYSTEM_PROMPT
        self.context_user_template = CONTEXT_ASSESSMENT_USER_TEMPLATE
        self.class_system_prompt = CLASSIFICATION_SYSTEM_PROMPT
        self.class_user_template = CLASSIFICATION_USER_TEMPLATE
        self._context_agent = ContextAndRelevanceAgent(llm)

    def analyze_context(
        self,
        clinical_text: str,
        diagnoses: list[ExtractedDiagnosis],
    ) -> list[ContextualizedDiagnosis]:
        state: PipelineGraphState = {
            "raw_text": clinical_text,
            "extracted_diagnoses": diagnoses,
        }
        res = self._context_agent.run_node(state)
        return res.get("contextualized_diagnoses", [])

    def classify_roles(
        self,
        clinical_text: str,
        conditions: list[ContextualizedDiagnosis],
    ) -> list[ClassifiedDiagnosis]:
        """Classify contextualized conditions into at most ONE primary and secondary roles."""
        assessments: list[ContextAssessment] = []
        for c in conditions:
            is_excl = (
                c.negation == NegationStatus.NEGATED
                or c.certainty == Certainty.RULED_OUT
                or c.temporality == Temporality.HISTORICAL
            )
            sec = getattr(c.evidence, "source_section", None) if c.evidence else None
            assessments.append(
                ContextAssessment(
                    condition_id=c.diagnosis_id,
                    diagnosis=c.raw_term,
                    section=sec,
                    current_relevance=not is_excl,
                    coding_candidate=not is_excl,
                    status=ConditionStatus.HISTORICAL
                    if c.temporality == Temporality.HISTORICAL
                    else ConditionStatus.ACTIVE,
                    certainty=c.certainty,
                    temporality=c.temporality,
                    negation=c.negation,
                    evidence=c.evidence.quote,
                    reason=f"[{sec}] {c.clinical_justification}" if sec else c.clinical_justification,
                    treated_or_managed=not is_excl,
                )
            )

        enc_result, _ = self.classify_conditions(assessments, clinical_text)
        classified: list[ClassifiedDiagnosis] = []
        cond_map = {c.diagnosis_id: c for c in conditions}

        for cl in enc_result.all_classifications:
            ctx = cond_map.get(cl.diagnosis_id) or ContextualizedDiagnosis(
                diagnosis_id=cl.diagnosis_id,
                raw_term=cl.diagnosis,
                evidence=EvidenceSnippet(
                    quote=cl.evidence_quote or "Evidence",
                    source_section="CLASSIFICATION",
                ),
                extraction_confidence=1.0,
                negation=NegationStatus.AFFIRMATIVE,
                temporality=Temporality.CURRENT
                if cl.role != DiagnosisRole.EXCLUDED
                else Temporality.HISTORICAL,
                certainty=Certainty.CONFIRMED,
                acuity=Acuity.ACUTE if cl.role == DiagnosisRole.PRIMARY else Acuity.UNSPECIFIED,
                clinical_justification=cl.classification_reason,
            )
            classified.append(
                ClassifiedDiagnosis(
                    diagnosis_id=cl.diagnosis_id,
                    raw_term=cl.diagnosis,
                    context=ctx,
                    role=cl.role,
                    is_billable_candidate=cl.is_billable_candidate,
                    classification_reason=cl.classification_reason,
                    primary_justification=cl.primary_justification,
                )
            )
        return classified

    def run(self, **kwargs: Any) -> list[ClassifiedDiagnosis]:
        text = kwargs.get("clinical_text") or kwargs.get("text") or ""
        diagnoses = kwargs.get("diagnoses") or kwargs.get("conditions") or []
        if diagnoses and isinstance(diagnoses[0], ContextualizedDiagnosis):
            return self.classify_roles(text, diagnoses)  # type: ignore[arg-type]
        contextualized = self.analyze_context(text, diagnoses)  # type: ignore[arg-type]
        return self.classify_roles(text, contextualized)
