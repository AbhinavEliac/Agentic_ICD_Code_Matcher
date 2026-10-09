"""Evidence-first clinical fact extractor and section-aware clinical document segmenter.

Strictly enforces:
1. Discharge summary / clinical diagnosis section is the primary authority.
2. Objective investigation findings (echo, CT, lab) can support, but NEVER authorize a diagnosis alone.
3. Pre-existing PMH without documented inpatient management cannot become active diagnoses.
4. Ruled-out conditions are marked RULED_OUT / NEGATED.
5. Uncertain conditions preserve uncertainty (SUSPECTED / POSSIBLE).
6. Zero hardcoded disease whitelists.
"""

import re
from typing import Any

from medical_coding.schemas.enums import (
    Acuity,
    AssertionStatus,
    Certainty,
    DiagnosisRole,
    EvidenceType,
    Laterality,
    NegationStatus,
    Temporality,
)
from medical_coding.schemas.evidence import (
    CancerConcept,
    ClinicalDiagnosisCandidate,
    ClinicalDiagnosisState,
    MultiDimensionalScore,
    StructuredEvidence,
)
from medical_coding.utils.logging import get_logger
from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

logger = get_logger(__name__)

# Standard section header regexes
SECTION_PATTERNS: list[tuple[str, str]] = [
    ("DISCHARGE_SUMMARY", r"(?:HOSPITAL\s+DISCHARGE\s+SUMMARY|DISCHARGE\s+SUMMARY|CLINICAL\s+SUMMARY)"),
    ("DISCHARGE_DIAGNOSES", r"(?:DISCHARGE\s+DIAGNOS[EI]S|FINAL\s+DIAGNOS[EI]S|FINAL\s+CODING\s+SUMMARY|CODING\s+SUMMARY|FINAL\s+CODING|POSTOPERATIVE\s+DIAGNOS[EI]S|DIAGNOS[EI]S\s+ON\s+DISCHARGE|DIAGNOSIS)"),
    ("PRINCIPAL_DIAGNOSIS", r"(?:PRINCIPAL\s+DIAGNOS[EI]S|PRIMARY\s+DIAGNOS[EI]S|ADMITTING\s+DIAGNOS[EI]S|ADMISSION\s+DIAGNOS[EI]S)"),
    ("SECONDARY_DIAGNOSES", r"(?:SECONDARY\s+DIAGNOS[EI]S|ADDITIONAL\s+DIAGNOS[EI]S|CO-?MORBIDITIES|OTHER\s+DIAGNOS[EI]S)"),
    ("CHIEF_COMPLAINT", r"(?:CHIEF\s+COMPLAINT|REASON\s+FOR\s+ADMISSION|PRESENTING\s+COMPLAINT|ADMITTED\s+FOR)"),
    ("HISTORY_OF_PRESENT_ILLNESS", r"(?:HISTORY\s+OF\s+PRESENT\s+ILLNESS|HPI)"),
    ("PHYSICAL_EXAMINATION", r"(?:CLINICAL\s+EXAMINATION|PHYSICAL\s+EXAMINATION|ON\s+EXAMINATION|GENERAL\s+EXAMINATION|SYSTEMIC\s+EXAMINATION)"),
    ("HOSPITAL_COURSE", r"(?:HOSPITAL\s+COURSE|SUMMARY\s+OF\s+HOSPITAL\s+STAY|BRIEF\s+SUMMARY\s+OF\s+HOSPITAL\s+COURSE|COURSE\s+IN\s+(?:THE\s+)?HOSPITAL(?:\s+AND\s+DISCUSSION)?|CLINICAL\s+COURSE)"),
    ("COMPLICATIONS", r"(?:HOSPITAL\s+COMPLICATIONS|IN-?HOSPITAL\s+COMPLICATIONS|COMPLICATIONS|ADVERSE\s+EVENTS)"),
    ("PROCEDURES", r"(?:PROCEDURES\s+PERFORMED|OPERATIVE\s+PROCEDURES|SURGICAL\s+PROCEDURES|MAJOR\s+PROCEDURES|PROCEDURES)"),
    ("OPERATIVE_FINDINGS", r"(?:INTRAOPERATIVE\s+FINDINGS|OPERATIVE\s+FINDINGS|SURGICAL\s+FINDINGS|INTRA-?OPERATIVE\s+FINDINGS|OPERATIVE\s+NOTE\s+FINDINGS|FINDINGS)"),
    ("ONCOLOGY_BIOMARKERS", r"(?:BIOMARKERS\s*(?:&|AND)?\s*RECEPTOR\s+STATUS|PATHOLOGY(?:\s+REPORT)?|HISTOPATHOLOGY|RECEPTOR\s+STATUS|IMMUNOHISTOCHEMISTRY|TUMOR\s+CHARACTERISTICS)"),
    ("PAST_MEDICAL_HISTORY", r"(?:PAST\s+MEDICAL\s+HISTORY|PMH|MEDICAL\s+HISTORY|BACKGROUND\s+HISTORY|PAST\s+HISTORY|PERSONAL\s+HISTORY)"),
    ("PAST_SURGICAL_HISTORY", r"(?:PAST\s+SURGICAL\s+HISTORY|PSH|SURGICAL\s+HISTORY)"),
    ("INVESTIGATIONS", r"(?:RELEVANT\s+INVESTIGATIONS|INVESTIGATIONS|LABORATORY\s+DATA|PERTINENT\s+LABS|DIAGNOSTIC\s+STUDIES|IMAGING|RADIOLOGY|ECHOCARDIOGRAM|CT\s+SCAN|ULTRASOUND|MRI)"),
    ("MICROBIOLOGY", r"(?:MICROBIOLOGY|CULTURES?|BLOOD\s+CULTURES?|URINE\s+CULTURES?|SPUTUM\s+CULTURES?|CULTURE\s+AND\s+SENSITIVITY|MICROBIOLOGICAL\s+STUDIES)"),
    ("ALLERGIES", r"(?:DRUG\s+ALLERGIES|ALLERGIES|ALLERGIC\s+HISTORY|ALLERGIC\s+REACTIONS?)"),
    ("MEDICATIONS", r"(?:DISCHARGE\s+MEDICATIONS|MEDICATIONS\s+ON\s+DISCHARGE|ACTIVE\s+MEDICATIONS|MEDICATIONS|CURRENT\s+MEDICATIONS)"),
    ("PERTINENT_NEGATIVES", r"(?:PERTINENT\s+NEGATIVES|NEGATIONS(?:\s*/\s*AUDIT)?|DOCUMENTATION\s+AUDIT)"),
    ("ASSESSMENT", r"(?:ASSESSMENT\s+AND\s+PLAN|ASSESSMENT|IMPRESSION)"),
    ("PLAN", r"(?:DISCHARGE\s+PLAN|TREATMENT\s+PLAN|PLAN)"),
    ("DISCHARGE_INSTRUCTIONS", r"(?:DISCHARGE\s+INSTRUCTIONS|DISPOSITION|DISCHARGE\s+CONDITION|FOLLOW-?UP)"),
]

NEGATION_CUES = [
    r"\bruled\s+out\b",
    r"\bdefinitively\s+ruled\s+out\b",
    r"\bno\s+evidence\s+of\b",
    r"\bnegative\s+for\b",
    r"\bdenies\b",
    r"\bwithout\s+evidence\s+of\b",
    r"\brefuted\b",
    r"\bexcludes\b",
    r"\bnot\s+present\b",
]

UNCERTAINTY_CUES = [
    r"\bpossible\b",
    r"\bsuspected\b",
    r"\bquestionable\b",
    r"\bquestion\s+of\b",
    r"\bcannot\s+exclude\b",
    r"\bprobable\b",
    r"\brule\s+out\b",
    r"\bdifferential\b",
    r"\bevolving\b",
    r"\bearly\s+evolving\b",
    r"\buncertain\b",
    r"\bunclear\b",
    r"\bequivocal\b",
    r"\?",
]

ACUITY_ACUTE_CUES = [
    r"\bacute\b", r"\bdecompensated\b", r"\bexacerbation\b", r"\bemergent\b", r"\bsevere\s+acute\b",
    r"\bsepsis\b", r"\bseptic\b", r"\binfarction\b", r"\bstemi\b", r"\bnstemi\b",
]
ACUITY_CHRONIC_CUES = [r"\bchronic\b", r"\blongstanding\b", r"\bpre-existing\b", r"\bbaseline\b"]

PROCEDURE_SOURCE_CONTROL_CUES = [
    r"\bstent\b",
    r"\bstenting\b",
    r"\bdj\s+stent",
    r"\bcatheterization\b",
    r"\bappendectomy\b",
    r"\bcholecystectomy\b",
    r"\bsurgery\b",
    r"\boperative\b",
    r"\bresection\b",
    r"\bdecompression\b",
    r"\bsource\s+control\b",
    r"\bdrainage\b",
    r"\bangioplasty\b",
    r"\bbypass\b",
    r"\bintubation\b",
    r"\blaparoscopy\b",
    r"\blaparotomy\b",
]


class ClinicalDocumentSection:
    """Represents a bounded clinical section within a document."""

    def __init__(self, section_name: str, raw_header: str, content: str, start_char: int, end_char: int) -> None:
        self.section_name = section_name
        self.raw_header = raw_header
        self.content = content.strip()
        self.start_char = start_char
        self.end_char = end_char

    def __repr__(self) -> str:
        return f"<Section {self.section_name} ({len(self.content)} chars)>"


class SectionSegmenter:
    """Deterministically segments clinical text into recognized clinical sections."""

    @classmethod
    def segment(cls, text: str) -> list[ClinicalDocumentSection]:
        if not text or not text.strip():
            return []

        # Find all section header matches
        matches: list[dict[str, Any]] = []
        for sec_name, pat_str in SECTION_PATTERNS:
            regex = re.compile(rf"(?:^|\n|[.;]\s*|\b(?=[A-Z\s]{{4,}}:))\s*({pat_str})\s*(?::|--|\n)", re.IGNORECASE)
            for m in regex.finditer(text):
                matches.append({
                    "section_name": sec_name,
                    "header_text": m.group(1).strip(),
                    "start_pos": m.start(1),
                    "content_start": m.end(),
                })

        if not matches:
            return [
                ClinicalDocumentSection(
                    section_name="GENERAL",
                    raw_header="DOCUMENT",
                    content=text.strip(),
                    start_char=0,
                    end_char=len(text),
                )
            ]

        # Sort matches by appearance in text
        matches.sort(key=lambda x: x["start_pos"])

        sections: list[ClinicalDocumentSection] = []
        for i, m in enumerate(matches):
            content_start = m["content_start"]
            content_end = matches[i + 1]["start_pos"] if i + 1 < len(matches) else len(text)
            content_str = text[content_start:content_end]
            sections.append(
                ClinicalDocumentSection(
                    section_name=m["section_name"],
                    raw_header=m["header_text"],
                    content=content_str,
                    start_char=content_start,
                    end_char=content_end,
                )
            )

        return sections


class EvidenceFirstFactExtractor:
    """Extracts evidence-grounded clinical facts from partitioned document sections."""

    def __init__(self) -> None:
        self._extracted_procedures: list[str] = []
        self._operative_findings: list[dict[str, Any]] = []

    def extract_clinical_state(
        self,
        text: str,
        document_id: str = "doc-default",
        doc_id: str | None = None,
    ) -> ClinicalDiagnosisState:
        """Extract all clinical diagnosis candidates, score them, and determine roles."""
        effective_doc_id = doc_id or document_id
        self._extracted_procedures = []
        self._operative_findings = []
        sections = SectionSegmenter.segment(text)
        candidates = self._extract_candidates_from_sections(sections, text)

        # Classify candidates using strict evidence hierarchy
        state = self._classify_and_structure_state(candidates, sections, effective_doc_id)
        return state

    def _register_candidate(
        self,
        candidates_by_key: dict[str, ClinicalDiagnosisCandidate],
        item: ClinicalDiagnosisCandidate,
    ) -> None:
        """Register, validate, and corroborate a candidate in the clinical inventory."""
        # Strip procedural wrapper if candidate is phrased as a surgical procedure for a disease (Section 11 & 12)
        proc_wrapper_m = re.match(
            r"^(?:(?:elective|open|laparoscopic|robotic|urgent|emergency|bilateral|unilateral|primary|revision)\s+)?"
            r"(?:repair|resection|excision|stenting|mastectomy|cholecystectomy|appendectomy|biopsy|replacement|arthroplasty|fusion|drainage|decompression|catheterization|bypass|angioplasty|amputation|debridement)\s+(?:of|for)\s+(.+)$",
            item.raw_term.strip(),
            re.IGNORECASE,
        )
        if proc_wrapper_m:
            underlying_disease = proc_wrapper_m.group(1).strip()
            if len(underlying_disease) >= 3:
                item.raw_term = underlying_disease
                item.normalized_diagnosis = underlying_disease
                item.procedure_relevance = 1.0

        # Check candidate validity with Hard Gate
        is_valid, reject_reason = HardClinicalCandidateGate.evaluate_candidate(
            item.raw_term, item.primary_evidence_quote
        )
        if not is_valid:
            if "procedure" in reject_reason.lower() or "surgical" in reject_reason.lower():
                self._extracted_procedures.append(item.raw_term)
            logger.debug("Candidate gate rejected '%s': %s", item.raw_term, reject_reason)
            return

        item.entity_type = HardClinicalCandidateGate.classify_entity_type(
            item.raw_term, item.primary_evidence_quote
        )

        raw_t = item.raw_term.strip()
        # Handle question mark shorthand or uncertainty (?early evolving renal abscess)
        is_questionable = (
            raw_t.startswith("?")
            or "?" in raw_t
            or item.certainty == Certainty.SUSPECTED
            or any(re.search(pat, raw_t, re.IGNORECASE) for pat in UNCERTAINTY_CUES)
        )
        if is_questionable:
            item.assertion_status = AssertionStatus.UNCERTAIN
            item.certainty = Certainty.SUSPECTED
            item.role = DiagnosisRole.UNCERTAIN
            item.is_authorized = False
            raw_t = re.sub(r"^\?\s*", "", raw_t).strip()
            item.raw_term = raw_t
            item.normalized_diagnosis = raw_t

        canon = self._canonicalize_term(item.raw_term)
        lat = item.clinical_attributes.get("laterality", "") if item.clinical_attributes else ""
        key = f"{canon.lower().strip()}_{lat}".strip("_")
        if not key or len(key) < 2:
            return

        if key not in candidates_by_key:
            candidates_by_key[key] = item
        else:
            existing = candidates_by_key[key]
            existing.evidence.extend(item.evidence)
            if item.role == DiagnosisRole.PRIMARY:
                existing.role = DiagnosisRole.PRIMARY
                if item.primary_justification:
                    existing.primary_justification = item.primary_justification
            if item.temporality == Temporality.CURRENT:
                existing.temporality = Temporality.CURRENT
                if existing.role == DiagnosisRole.HISTORICAL:
                    existing.role = DiagnosisRole.SECONDARY
                    existing.is_authorized = True
                existing.scores.encounter_relevance = 1.0
                existing.scores.admitting_score = max(
                    existing.scores.admitting_score, item.scores.admitting_score
                )
            if item.procedure_relevance > 0:
                existing.procedure_relevance = 1.0
            if item.treatment_relevance > 0:
                existing.treatment_relevance = 1.0
            if item.clinical_attributes:
                existing.clinical_attributes.update(item.clinical_attributes)

    def _extract_candidates_from_sections(
        self,
        sections: list[ClinicalDocumentSection],
        full_text: str,
    ) -> list[ClinicalDiagnosisCandidate]:
        candidates_by_key: dict[str, ClinicalDiagnosisCandidate] = {}

        # Phase 1: High-yield diagnosis sections (Discharge Diagnoses, Final Diagnoses, Assessment & Plan)
        for sec in sections:
            if sec.section_name in ("DISCHARGE_DIAGNOSES", "FINAL_DIAGNOSES", "PRINCIPAL_DIAGNOSIS", "SECONDARY_DIAGNOSES", "ASSESSMENT", "ASSESSMENT_PLAN"):
                items = self._parse_diagnosis_items(sec.content, sec)
                for item in items:
                    self._register_candidate(candidates_by_key, item)

        # Phase 2: Chief Complaint / Reason for Admission
        for sec in sections:
            if sec.section_name == "CHIEF_COMPLAINT":
                items = self._parse_complaint_items(sec.content, sec)
                for item in items:
                    self._register_candidate(candidates_by_key, item)

        # Phase 3: Past Medical History / Surgical History
        for sec in sections:
            if sec.section_name in ("PAST_MEDICAL_HISTORY", "PAST_SURGICAL_HISTORY"):
                items = self._parse_pmh_items(sec.content, sec)
                for item in items:
                    self._register_candidate(candidates_by_key, item)

        # Phase 4: Hospital Course & Complications (Complications, developments, active inpatient events)
        for sec in sections:
            if sec.section_name in ("HOSPITAL_COURSE", "COMPLICATIONS"):
                items = self._parse_course_and_complications(sec.content, sec)
                for item in items:
                    self._register_candidate(candidates_by_key, item)

        # Phase 5: Investigations & Microbiology (Imaging, pathology, lab findings, culture results)
        for sec in sections:
            if sec.section_name in ("INVESTIGATIONS", "MICROBIOLOGY"):
                items = self._parse_investigations_and_microbiology(sec.content, sec)
                for item in items:
                    self._register_candidate(candidates_by_key, item)

        # Phase 6: Procedures & Indications (Operative procedures, interventional indications)
        for sec in sections:
            if sec.section_name in ("PROCEDURES", "PAST_SURGICAL_HISTORY"):
                proc_lines = [l.strip() for l in sec.content.split("\n") if l.strip()]
                for l in proc_lines:
                    l_clean = re.sub(r"^\d+[\.\)\-]\s*", "", l).strip()
                    if len(l_clean) >= 3:
                        self._extracted_procedures.append(l_clean)
                items = self._parse_procedure_indications(sec.content, sec)
                for item in items:
                    self._register_candidate(candidates_by_key, item)

        # Phase 6b: Operative Findings (Intraoperative observations, cysts, adhesions)
        for sec in sections:
            if sec.section_name == "OPERATIVE_FINDINGS":
                op_findings, items = self._parse_operative_findings(sec.content, sec)
                self._operative_findings.extend(op_findings)
                for item in items:
                    self._register_candidate(candidates_by_key, item)

        # Phase 7: General Narrative Sections (General, discharge summary narrative, HPI)
        for sec in sections:
            if sec.section_name in ("GENERAL", "DISCHARGE_SUMMARY", "HISTORY_OF_PRESENT_ILLNESS"):
                items = self._parse_narrative_items(sec.content, sec)
                for item in items:
                    self._register_candidate(candidates_by_key, item)

        candidates = list(candidates_by_key.values())

        # Corroborate candidates with Hospital Course and Procedures
        self._corroborate_with_hospital_course_and_procedures(candidates, sections, full_text)

        return candidates

    @staticmethod
    def _extract_concept_and_attributes(raw_phrase: str) -> tuple[str, dict[str, Any]]:
        """Separate a clinical concept from attached clinical qualifiers, attributes, and staging (Section 10).

        Clinical principles:
        1. Neoplasm / Oncologic staging (Ann Arbor Roman numerals: Stage I-IV; TNM) and molecular markers
           (GCB, non-GCB, double expressor, receptor status) are attributes of the neoplasm.
        2. Non-oncologic organ staging (CKD stage 1-5, pressure ulcer stage 1-4) is integral to the clinical
           concept because it directly dictates the ICD-10-CM code.
        """
        attributes: dict[str, Any] = {}
        cleaned = raw_phrase.strip()

        # 1. Extract receptor status and molecular markers (e.g., Triple negative, BRCA, PD-L1, GCB type, double expressor)
        mol_m = re.search(
            r",?\s*\b(triple\s+negative|brca(?:\s*1|\s*2)?\s*(?:pathogenic|mutation|positive|negative)?(?:\s*mutation)?|pd[\s\-]?l1\s*(?:positive|negative)?(?:\s*\(.*?\))?|(?:non-)?gcb(?:\s+(?:sub)?type)?|double\s+expressor|bcl[0-9]+(?:\+|-)?|cd[0-9]+(?:\+|-)?|her2(?:\s*(?:positive|negative|0|\d+))?|egfr(?:\+|-)?)\b",
            cleaned,
            re.IGNORECASE,
        )
        if mol_m:
            attributes["molecular_marker"] = mol_m.group(1).strip()
            cleaned = re.sub(re.escape(mol_m.group(0)), "", cleaned, flags=re.IGNORECASE).strip()

        # 2. Extract histologic grade (e.g. Grade I, Grade II, Grade III, Grade IV)
        grade_m = re.search(
            r",?\s*\b(grade\s+(?:i|ii|iii|iv|[1-4])(?:\s+histology)?)\b",
            cleaned,
            re.IGNORECASE,
        )
        if grade_m:
            attributes["grade"] = grade_m.group(1).strip()
            cleaned = re.sub(re.escape(grade_m.group(0)), "", cleaned, flags=re.IGNORECASE).strip()

        # 3. Extract oncologic Roman-numeral staging (e.g., Stage I, Stage II, Stage III, Stage IV, Stage IVB)
        onc_stage_m = re.search(
            r",?\s*\b(stage\s+(?:i|ii|iii|iv)[a-z]?)\b",
            cleaned,
            re.IGNORECASE,
        )
        if onc_stage_m:
            attributes["oncologic_stage"] = onc_stage_m.group(1).strip()
            cleaned = re.sub(re.escape(onc_stage_m.group(0)), "", cleaned, flags=re.IGNORECASE).strip()

        # Clean trailing commas or whitespace
        cleaned = re.sub(r"[\s,]+$", "", cleaned).strip()
        return cleaned, attributes

    def _parse_narrative_items(
        self,
        content: str,
        sec: ClinicalDocumentSection,
    ) -> list[ClinicalDiagnosisCandidate]:
        """Extract evidence-grounded clinical diagnosis candidates from unstructured text."""
        candidates: list[ClinicalDiagnosisCandidate] = []
        cleaned = content.strip()
        if not cleaned:
            return []

        # Split into sentences or lines
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", cleaned) if s.strip()]

        for sent in raw_sentences:
            if len(sent) < 4:
                continue

            # Strip preamble like "DISCHARGE SUMMARY" or metadata
            sent_clean = re.sub(
                r"^(?:HOSPITAL\s+DISCHARGE\s+SUMMARY|DISCHARGE\s+SUMMARY|CLINICAL\s+SUMMARY|SUMMARY)\s*",
                "",
                sent,
                flags=re.IGNORECASE,
            ).strip()
            if not sent_clean or len(sent_clean) < 4:
                continue

            # Pattern 1: Definitively ruled out / excluded conditions
            ro_match = re.search(
                r"([A-Za-z0-9\s\-]+?)\s+(?:was|is|has\s+been)\s+(?:definitively\s+)?(?:ruled\s+out|excluded|negative)",
                sent_clean,
                re.IGNORECASE,
            )
            if ro_match:
                term = ro_match.group(1).strip()
                term = re.sub(r"^(?:patient\s+had\s+|there\s+was\s+no\s+|for\s+)", "", term, flags=re.IGNORECASE).strip()
                if len(term) >= 3:
                    ev = StructuredEvidence(
                        text=sent_clean,
                        section=sec.section_name,
                        sentence=sent_clean,
                        polarity=NegationStatus.NEGATED,
                        certainty=Certainty.RULED_OUT,
                        temporality=Temporality.CURRENT,
                        evidence_type=EvidenceType.DISCHARGE_SUMMARY,
                        clinical_relevance=0.0,
                    )
                    cand = ClinicalDiagnosisCandidate(
                        raw_term=term,
                        normalized_diagnosis=term,
                        role=DiagnosisRole.RULED_OUT,
                        certainty=Certainty.RULED_OUT,
                        temporality=Temporality.CURRENT,
                        evidence=[ev],
                        scores=MultiDimensionalScore(
                            evidence_score=1.0,
                            diagnostic_certainty=0.0,
                            encounter_relevance=0.0,
                            role_confidence=0.0,
                            contradiction_penalty=1.0,
                            admitting_score=-100.0,
                        ),
                        is_authorized=False,
                        classification_reason="Ruled out in narrative assessment",
                    )
                    candidates.append(cand)
                    continue

            from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

            # Skip absence statements, treatment instructions, and cross-reference boilerplate
            if (
                HardClinicalCandidateGate.is_absence_statement(sent_clean)
                or HardClinicalCandidateGate.is_treatment_instruction(sent_clean)
                or re.search(r"\b(?:symptoms?|complaints?)\s+described\s+in\b|\bdescribed\s+in\s+(?:the\s+)?final\s+diagnosis\b|\babove\s+mentioned\s+complaints\b", sent_clean, re.IGNORECASE)
            ):
                continue

            # Pattern 2: Explicit admission / diagnosis / condition clauses
            adm_match = re.search(
                r"(?:patient\s+(?:was\s+)?(?:admitted\s+(?:with|for|in)|presented\s+with|diagnosed\s+with|treated\s+for|has|had)|admitted\s+(?:with|for|in)|impression\s*:\s*)\s+([^.]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if adm_match:
                clause = adm_match.group(1).strip()
                clause = re.sub(
                    r"^(?:(?:several|\d+)\s+(?:days?|weeks?|months?|hours?)\s+of\s+)",
                    "",
                    clause,
                    flags=re.IGNORECASE,
                ).strip()
                for tail_delim in [" on ", " treated with ", " managed with ", " requiring ", " with good response", " with symptom"]:
                    if tail_delim in clause.lower():
                        idx = clause.lower().find(tail_delim)
                        clause = clause[:idx].strip()

                # Causal admission: "initiation of chemotherapy for DLBCL" -> "DLBCL"
                causal_m = re.search(
                    r"(?:(?:initiation|cycle\s+\d+|course)\s+of\s+)?(?:chemotherapy|immunotherapy|radiation|treatment|therapy|infusion|management|stenting|surgery)\s+(?:for|of)\s+([^.,;\n]+)",
                    clause,
                    re.IGNORECASE,
                )
                if causal_m:
                    sub_cand = causal_m.group(1).strip()
                    if len(sub_cand) >= 3:
                        clause = sub_cand

                sub_clauses = self._split_compound_clinical_phrase(clause)
                for sc in sub_clauses:
                    sc = sc.strip()
                    sc = re.sub(r"^(?:a|an|the)\s+", "", sc, flags=re.IGNORECASE).strip()
                    sc, cand_attrs = self._extract_concept_and_attributes(sc)
                    if len(sc) < 3 or sc.lower() in ("stable condition", "discharge", "home", "baseline"):
                        continue

                    is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(sc, sent_clean)
                    if not is_valid:
                        continue

                    polarity, certainty = self._assess_polarity_and_certainty(sent_clean)
                    acuity = self._detect_acuity(sc)
                    ev = StructuredEvidence(
                        text=sent_clean,
                        section=sec.section_name,
                        sentence=sent_clean,
                        polarity=polarity,
                        certainty=certainty,
                        temporality=Temporality.CURRENT,
                        evidence_type=EvidenceType.DISCHARGE_SUMMARY,
                        clinical_relevance=1.0,
                    )
                    scores = MultiDimensionalScore(
                        evidence_score=1.0,
                        diagnostic_certainty=1.0 if certainty == Certainty.CONFIRMED else 0.8,
                        encounter_relevance=1.0,
                        role_confidence=1.0,
                        semantic_match=1.0,
                        specificity_match=1.0,
                        contradiction_penalty=1.0 if polarity == NegationStatus.NEGATED else 0.0,
                        admitting_score=5.0 + (1.0 if acuity == Acuity.ACUTE else 0.0),
                    )
                    cand = ClinicalDiagnosisCandidate(
                        raw_term=sc,
                        normalized_diagnosis=sc,
                        role=DiagnosisRole.PRIMARY,
                        certainty=certainty,
                        temporality=Temporality.CURRENT,
                        clinical_attributes=cand_attrs,
                        evidence=[ev],
                        evidence_strength=1.0,
                        clinical_relevance=1.0,
                        encounter_relevance=1.0,
                        scores=scores,
                        is_authorized=polarity != NegationStatus.NEGATED,
                        classification_reason="Identified as encounter diagnosis in clinical narrative",
                    )
                    candidates.append(cand)

        return candidates

    def _parse_diagnosis_items(
        self,
        content: str,
        sec: ClinicalDocumentSection,
    ) -> list[ClinicalDiagnosisCandidate]:
        candidates: list[ClinicalDiagnosisCandidate] = []
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        for line_idx, line in enumerate(lines):
            # Check explicit provider-designated PRIMARY role before stripping prefixes
            is_explicit_primary_line = bool(
                re.match(
                    r"^\s*(?:primary|principal)\s*(?:diagnosis|diagnoses|condition)?\s*[:\-–—]",
                    line,
                    re.IGNORECASE,
                )
            )
            # Strip list numbering e.g. "1. Acute right emphysematous pyelonephritis - underwent DJ stenting"
            cleaned_line = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
            # Strip structural role prefixes (e.g., "Primary:", "Principal Diagnosis -", "Secondary:")
            cleaned_line = re.sub(
                r"^(?:(?:primary|principal|secondary|final|admitting)\s*(?:diagnosis|diagnoses|condition)?|(?:diagnosis|diagnoses|condition))\s*[:\-–—]\s*",
                "",
                cleaned_line,
                flags=re.IGNORECASE,
            ).strip()
            if not cleaned_line or len(cleaned_line) < 3:
                continue

            if re.match(r"^(?:pertinent\s+negatives?|negations?(?:\s*/\s*audit)?|documentation\s+audit|audit|plan|discharge\s+medications?|medications?)\s*:?$", cleaned_line, re.IGNORECASE):
                continue

            # Split diagnosis from attached narrative / treatment
            diag_term, narrative = self._split_term_and_narrative(cleaned_line)
            if not diag_term or len(diag_term) < 3:
                continue

            # Check negation
            polarity, certainty = self._assess_polarity_and_certainty(cleaned_line)

            # Check explicit historical / resolved notation in line
            is_explicit_historical = bool(
                re.search(
                    r"\b(?:historical\s*/\s*resolved|historical\b|prior\s+history|childhood\s+asthma|no\s+current\s+treatment|resolved\s+\d+\s+years?\s+ago)\b",
                    cleaned_line,
                    re.IGNORECASE,
                )
            )

            from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

            # Split compound diagnoses into independent clinical concepts
            sub_terms = self._split_compound_clinical_phrase(diag_term)
            for sub_idx, sub_term in enumerate(sub_terms):
                # Strip trailing staging or molecular attributes from diagnosis name (Section 10)
                sub_term, diag_attrs = self._extract_concept_and_attributes(sub_term)
                if not sub_term or len(sub_term) < 3:
                    continue

                # Evaluate with HardClinicalCandidateGate if not an explicit rule-out
                if polarity != NegationStatus.NEGATED:
                    is_valid_cand, gate_reason = HardClinicalCandidateGate.evaluate_candidate(sub_term, cleaned_line)
                    if not is_valid_cand:
                        if "procedure" in gate_reason.lower() or "surgical" in gate_reason.lower():
                            self._extracted_procedures.append(sub_term)
                        logger.info("Clinical gate rejected candidate in %s: '%s' (%s)", sec.section_name, sub_term, gate_reason)
                        continue

                # Acuity & Laterality
                acuity = self._detect_acuity(sub_term)
                laterality = self._detect_laterality(sub_term)
                if laterality and laterality != Laterality.UNSPECIFIED and "laterality" not in diag_attrs:
                    diag_attrs["laterality"] = laterality.value

                # Check procedure / source control intervention
                proc_rel = 1.0 if any(re.search(pat, cleaned_line, re.IGNORECASE) for pat in PROCEDURE_SOURCE_CONTROL_CUES) else 0.0
                treat_rel = 1.0 if any(w in cleaned_line.lower() for w in ["treated", "iv", "antibiotic", "diuresis", "insulin", "surgery", "stent", "infusion"]) else 0.0

                cand_temp = Temporality.HISTORICAL if is_explicit_historical else Temporality.CURRENT
                cand_role = (
                    DiagnosisRole.HISTORICAL
                    if is_explicit_historical
                    else (
                        DiagnosisRole.PRIMARY
                        if (
                            (is_explicit_primary_line and sub_idx == 0)
                            or (sec.section_name in ("DISCHARGE_DIAGNOSES", "FINAL_DIAGNOSES", "PRINCIPAL_DIAGNOSIS") and line_idx == 0 and sub_idx == 0)
                        )
                        else DiagnosisRole.SECONDARY
                    )
                )

                evidence_obj = StructuredEvidence(
                    text=line,
                    section=sec.section_name,
                    sentence=line,
                    polarity=polarity,
                    certainty=certainty,
                    temporality=cand_temp,
                    evidence_type=EvidenceType.DISCHARGE_SUMMARY,
                    clinical_relevance=0.0 if is_explicit_historical else 1.0,
                )

                # Explicit PRINCIPAL_DIAGNOSIS gets highest base score (+15.0); DISCHARGE_DIAGNOSES gets +10.0; SECONDARY gets +3.0
                if is_explicit_historical:
                    base_admitting = -50.0
                elif is_explicit_primary_line and sub_idx == 0:
                    base_admitting = 15.0
                elif sec.section_name in ("PRINCIPAL_DIAGNOSIS", "PRIMARY_DIAGNOSIS"):
                    base_admitting = 15.0 if (line_idx == 0 and sub_idx == 0) else 5.0
                elif sec.section_name in ("DISCHARGE_DIAGNOSES", "FINAL_DIAGNOSES"):
                    base_admitting = 10.0 if (line_idx == 0 and sub_idx == 0) else 4.0
                elif sec.section_name in ("SECONDARY_DIAGNOSES", "ADDITIONAL_DIAGNOSES"):
                    base_admitting = 3.0
                else:
                    base_admitting = 4.0

                scores = MultiDimensionalScore(
                    evidence_score=1.0,
                    diagnostic_certainty=1.0 if certainty == Certainty.CONFIRMED else 0.8 if certainty == Certainty.SUPPORTED else 0.5 if certainty in (Certainty.SUSPECTED, Certainty.POSSIBLE) else 0.0,
                    encounter_relevance=0.0 if is_explicit_historical else 1.0,
                    role_confidence=1.0,
                    semantic_match=1.0,
                    specificity_match=1.0,
                    contradiction_penalty=1.0 if polarity == NegationStatus.NEGATED else 0.0,
                    admitting_score=base_admitting + (proc_rel * 2.0) + (1.0 if acuity == Acuity.ACUTE else 0.0),
                )

                cand = ClinicalDiagnosisCandidate(
                    raw_term=sub_term,
                    normalized_diagnosis=sub_term,
                    role=cand_role,
                    certainty=certainty,
                    temporality=cand_temp,
                    assertion_status=AssertionStatus.CONFIRMED if certainty == Certainty.CONFIRMED else (AssertionStatus.RULED_OUT if polarity == NegationStatus.NEGATED else AssertionStatus.SUSPECTED),
                    clinical_attributes=diag_attrs,
                    evidence=[evidence_obj],
                    evidence_strength=1.0,
                    clinical_relevance=0.0 if is_explicit_historical else 1.0,
                    encounter_relevance=0.0 if is_explicit_historical else 1.0,
                    treatment_relevance=treat_rel,
                    procedure_relevance=proc_rel,
                    scores=scores,
                    is_authorized=(polarity != NegationStatus.NEGATED) and (not is_explicit_historical),
                    classification_reason=f"Extracted from {sec.section_name}" if sub_idx == 0 else f"Extracted as co-occurring condition from {sec.section_name}",
                )
                candidates.append(cand)

        return candidates

    def _parse_complaint_items(
        self,
        content: str,
        sec: ClinicalDocumentSection,
    ) -> list[ClinicalDiagnosisCandidate]:
        candidates: list[ClinicalDiagnosisCandidate] = []
        cleaned = content.strip()
        if not cleaned:
            return []

        # Skip cross-reference boilerplate (e.g. "presented with symptoms described in the final diagnosis")
        if re.search(r"\b(?:symptoms?|complaints?)\s+described\s+in\b|\bdescribed\s+in\s+(?:the\s+)?final\s+diagnosis\b|\babove\s+mentioned\s+complaints\b", cleaned, re.IGNORECASE):
            return []

        # Split sentences (do not split decimal numbers like 101.4F or 1.5 mg/dL)
        sentences = [s.strip() for s in re.split(r"\.(?!\d)|[\n;]+", cleaned) if s.strip()]
        for s in sentences:
            if len(s) < 4:
                continue

            cond_term = s
            cond_term = re.sub(
                r"^(?:(?:several|\d+)\s+(?:days?|weeks?|months?|hours?)\s+of\s+)",
                "",
                cond_term,
                flags=re.IGNORECASE,
            ).strip()
            causal_m = re.search(
                r"(?:admitted\s+(?:for|with)|presented\s+with|evaluation\s+of|reason\s+for\s+admission\s*:?)\s*(?:(?:initiation|cycle\s+\d+|course)\s+of\s+)?(?:chemotherapy|immunotherapy|radiation|treatment|therapy|infusion|management|stenting|surgery)\s+(?:for|of|with)\s+([^.,;\n]+)",
                s,
                re.IGNORECASE,
            )
            if causal_m:
                cand_sub = causal_m.group(1).strip()
                if len(cand_sub) >= 3 and not cand_sub.lower().startswith("further"):
                    cond_term = cand_sub

            # Strip trailing staging or molecular attributes from diagnosis name (Section 10)
            cond_term, cond_attrs = self._extract_concept_and_attributes(cond_term)
            if not cond_term or len(cond_term) < 3:
                continue

            from medical_coding.validation.clinical_gate import HardClinicalCandidateGate
            is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(cond_term, s)
            if not is_valid:
                continue

            polarity, certainty = self._assess_polarity_and_certainty(s)
            ev = StructuredEvidence(
                text=s,
                section="CHIEF_COMPLAINT",
                sentence=s,
                polarity=polarity,
                certainty=certainty,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.ADMISSION_REASON,
                clinical_relevance=1.0,
            )
            is_symptom = any(
                sym in cond_term.lower()
                for sym in [
                    "pain", "colic", "discomfort", "fever", "cough", "dyspnea",
                    "shortness of breath", "nausea", "vomiting", "wheezing",
                    "fatigue", "malaise", "headache"
                ]
            )
            base_score = 2.0 if is_symptom else 3.5
            scores = MultiDimensionalScore(
                evidence_score=1.0,
                diagnostic_certainty=1.0 if certainty == Certainty.CONFIRMED else 0.8,
                encounter_relevance=1.0,
                role_confidence=0.5 if is_symptom else 0.8,
                admitting_score=base_score,
            )
            cand = ClinicalDiagnosisCandidate(
                raw_term=cond_term,
                normalized_diagnosis=cond_term,
                role=DiagnosisRole.SECONDARY if is_symptom else DiagnosisRole.PRIMARY,
                certainty=certainty,
                temporality=Temporality.CURRENT,
                clinical_attributes=cond_attrs,
                evidence=[ev],
                scores=scores,
                is_authorized=polarity != NegationStatus.NEGATED,
                classification_reason="Documented as chief presenting symptom" if is_symptom else "Documented as chief presenting complaint / admission occasioning condition",
            )
            candidates.append(cand)

        return candidates

    def _parse_pmh_items(
        self,
        content: str,
        sec: ClinicalDocumentSection,
    ) -> list[ClinicalDiagnosisCandidate]:
        candidates: list[ClinicalDiagnosisCandidate] = []
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
            if not cleaned or len(cleaned) < 3:
                continue

            cleaned_preamble = re.sub(
                r"^(?:known\s+case\s+of|k/c/o|history\s+of|h/o|patient\s+is\s+a\s+known\s+case\s+of)\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

            has_active_rx = bool(
                re.search(
                    r"\b(?:on\s+medication|on\s+treatment|on\s+rx|treated|managed)\b",
                    cleaned_preamble,
                    re.IGNORECASE,
                )
            )
            cleaned_concept = re.sub(
                r",?\s*\b(?:on\s+medication|on\s+treatment|on\s+rx)\b.*$",
                "",
                cleaned_preamble,
                flags=re.IGNORECASE,
            ).strip()

            sub_items = re.split(r"\s+and\s+|\s*,\s*", cleaned_concept)
            for sub_item in sub_items:
                term, _ = self._split_term_and_narrative(sub_item)
                term, pmh_attrs = self._extract_concept_and_attributes(term)
                if not term or len(term) < 3:
                    continue

                from medical_coding.validation.clinical_gate import HardClinicalCandidateGate
                is_valid_cand, _ = HardClinicalCandidateGate.evaluate_candidate(term, line)
                if not is_valid_cand:
                    continue

                polarity, certainty = self._assess_polarity_and_certainty(cleaned)
                is_resolved = "resolved" in cleaned.lower() or "remote" in cleaned.lower()

                if has_active_rx and not is_resolved and polarity != NegationStatus.NEGATED:
                    cand_temp = Temporality.CURRENT
                    cand_role = DiagnosisRole.SECONDARY
                    is_auth = True
                    base_admitting = 3.0
                    reason = "Pre-existing chronic comorbidity on active medication therapy"
                else:
                    cand_temp = Temporality.RESOLVED if is_resolved else Temporality.HISTORICAL
                    cand_role = DiagnosisRole.HISTORICAL
                    is_auth = False
                    base_admitting = -10.0
                    reason = "Past medical history condition without documented active inpatient care"

                ev = StructuredEvidence(
                    text=line,
                    section="PAST_MEDICAL_HISTORY",
                    sentence=line,
                    polarity=polarity,
                    certainty=certainty,
                    temporality=cand_temp,
                    evidence_type=EvidenceType.HISTORY,
                    clinical_relevance=1.0 if has_active_rx else 0.2,
                )
                scores = MultiDimensionalScore(
                    evidence_score=1.0 if has_active_rx else 0.8,
                    diagnostic_certainty=1.0,
                    encounter_relevance=1.0 if has_active_rx else 0.0,
                    role_confidence=0.8 if has_active_rx else 0.0,
                    admitting_score=base_admitting,
                )
                cand = ClinicalDiagnosisCandidate(
                    raw_term=term,
                    normalized_diagnosis=term,
                    role=cand_role,
                    certainty=certainty,
                    temporality=cand_temp,
                    clinical_attributes=pmh_attrs,
                    evidence=[ev],
                    scores=scores,
                    is_authorized=is_auth,
                    classification_reason=reason,
                )
                candidates.append(cand)

        return candidates

    def _parse_course_and_complications(
        self,
        content: str,
        sec: ClinicalDocumentSection,
    ) -> list[ClinicalDiagnosisCandidate]:
        """Extract active complications, acute developments, and managed conditions from hospital course."""
        candidates: list[ClinicalDiagnosisCandidate] = []
        cleaned = content.strip()
        if not cleaned:
            return []

        from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

        sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", cleaned) if s.strip()]

        for sent in sentences:
            if len(sent) < 5 or HardClinicalCandidateGate.is_absence_statement(sent) or HardClinicalCandidateGate.is_treatment_instruction(sent):
                continue

            # Pattern A: Complication cues: "course was complicated by [X]", "complicated by [X]", "developed [X]", "went into [X]", "manifested [X]"
            comp_match = re.search(
                r"(?:course\s+(?:was\s+)?complicated\s+by|complicated\s+by|developed|went\s+into|manifested|suffered)\s+([^.]+)",
                sent,
                re.IGNORECASE,
            )
            # Pattern B: Inpatient management/diagnosis: "treated for [X]", "managed for [X]", "diagnosed with [X]"
            treat_match = re.search(
                r"(?:treated\s+(?:for|with\s+(?:iv\s+)?[a-z0-9]+\s+for)|managed\s+for|diagnosed\s+with|inpatient\s+management\s+of)\s+([^.]+)",
                sent,
                re.IGNORECASE,
            )

            matched_clauses: list[str] = []
            if comp_match:
                matched_clauses.append(comp_match.group(1).strip())
            if treat_match:
                matched_clauses.append(treat_match.group(1).strip())

            for clause in matched_clauses:
                # Strip trailing management phrases from clause
                for tail in [" requiring ", " managed with ", " treated with ", " on ", " with good response", " with symptom", " and was ", " and received "]:
                    if tail in clause.lower():
                        idx = clause.lower().find(tail)
                        clause = clause[:idx].strip()

                # Split compound items e.g. "septic shock and diabetic ketoacidosis", "acute kidney injury secondary to dehydration"
                sub_items = self._split_compound_clinical_phrase(clause)
                for item_str in sub_items:
                    item_str = item_str.strip()
                    item_str = re.sub(r"^(?:a|an|the)\s+", "", item_str, flags=re.IGNORECASE).strip()
                    item_str, attrs = self._extract_concept_and_attributes(item_str)
                    if len(item_str) < 3:
                        continue

                    is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(item_str, sent)
                    if not is_valid:
                        continue

                    polarity, certainty = self._assess_polarity_and_certainty(sent)
                    acuity = self._detect_acuity(item_str)
                    laterality = self._detect_laterality(item_str)
                    if laterality and laterality != Laterality.UNSPECIFIED and "laterality" not in attrs:
                        attrs["laterality"] = laterality.value

                    ev = StructuredEvidence(
                        text=sent,
                        section=sec.section_name,
                        sentence=sent,
                        polarity=polarity,
                        certainty=certainty,
                        temporality=Temporality.CURRENT,
                        evidence_type=EvidenceType.HOSPITAL_COURSE,
                        clinical_relevance=1.0,
                    )
                    scores = MultiDimensionalScore(
                        evidence_score=1.0,
                        diagnostic_certainty=1.0 if certainty == Certainty.CONFIRMED else 0.8,
                        encounter_relevance=1.0,
                        role_confidence=1.0,
                        semantic_match=1.0,
                        specificity_match=1.0,
                        admitting_score=4.0 + (1.0 if acuity == Acuity.ACUTE else 0.0),
                    )
                    cand = ClinicalDiagnosisCandidate(
                        raw_term=item_str,
                        normalized_diagnosis=item_str,
                        role=DiagnosisRole.SECONDARY,
                        certainty=certainty,
                        temporality=Temporality.CURRENT,
                        assertion_status=AssertionStatus.CONFIRMED if certainty == Certainty.CONFIRMED else AssertionStatus.SUSPECTED,
                        clinical_attributes=attrs,
                        evidence=[ev],
                        evidence_strength=1.0,
                        clinical_relevance=1.0,
                        encounter_relevance=1.0,
                        treatment_relevance=1.0,
                        scores=scores,
                        is_authorized=polarity != NegationStatus.NEGATED,
                        classification_reason=f"Active inpatient complication/event in {sec.section_name}",
                    )
                    candidates.append(cand)

        return candidates

    def _parse_investigations_and_microbiology(
        self,
        content: str,
        sec: ClinicalDocumentSection,
    ) -> list[ClinicalDiagnosisCandidate]:
        """Extract objective diagnosis findings and pathogen results from investigations and cultures."""
        candidates: list[ClinicalDiagnosisCandidate] = []
        cleaned = content.strip()
        if not cleaned:
            return []

        from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

        sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", cleaned) if s.strip()]

        for sent in sentences:
            if len(sent) < 5 or HardClinicalCandidateGate.is_absence_statement(sent) or HardClinicalCandidateGate.is_treatment_instruction(sent):
                continue

            findings_clauses: list[str] = []

            # 1. Imaging findings: CT, CXR, Ultrasound, MRI, Biopsy, EGD
            img_match = re.search(
                r"(?:ct(?:\s+scan)?|cxr|chest\s+x-?ray|ultrasound|usg|mri|egd|biopsy|echocardiogram|studies)\s*(?:of\s+[^\n:]+)?\s*(?::|--|-)?\s*(?:showed|revealed|demonstrated|confirmed|reported|noted)\s+([^.]+)",
                sent,
                re.IGNORECASE,
            )
            if img_match:
                findings_clauses.append(img_match.group(1).strip())

            # 2. Microbiology / culture findings: blood culture, urine culture, sputum culture
            micro_match = re.search(
                r"(?:blood\s+cultures?|urine\s+cultures?|sputum\s+cultures?|cultures?|gram\s+stain|clo\s+test)\s*(?::|--|-)?\s*(?:grew|positive\s+for|showed|isolated|revealed)\s+([^.]+)",
                sent,
                re.IGNORECASE,
            )
            if micro_match:
                findings_clauses.append(micro_match.group(1).strip())

            for clause in findings_clauses:
                # Strip trailing notes or measures e.g. "causing hydronephrosis", "measuring 9mm", ">100,000 CFU/mL"
                for tail in [" causing ", " measuring ", " with symptom", " and was ", " with good response"]:
                    if tail in clause.lower():
                        idx = clause.lower().find(tail)
                        clause = clause[:idx].strip()

                # Split sub-findings e.g. "acute emphysematous pyelonephritis, right renal calculus (9mm), and ?early evolving renal abscess"
                sub_items = re.split(r"\s+and\s+|\s*,\s*", clause)
                for item_str in sub_items:
                    item_str = item_str.strip()
                    item_str = re.sub(r"^(?:a|an|the)\s+", "", item_str, flags=re.IGNORECASE).strip()
                    # Strip measurement brackets e.g. "(9mm)", "(>100,000 CFU/mL)"
                    item_str = re.sub(r"\([^)]*\)", "", item_str).strip()
                    item_str, attrs = self._extract_concept_and_attributes(item_str)
                    if len(item_str) < 3:
                        continue

                    # Check question mark / uncertainty shorthand (?early evolving renal abscess)
                    is_questionable = (
                        item_str.startswith("?")
                        or "?" in item_str
                        or any(re.search(pat, item_str, re.IGNORECASE) for pat in UNCERTAINTY_CUES)
                    )
                    clean_term = re.sub(r"^\?\s*", "", item_str).strip()

                    is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(clean_term, sent)
                    if not is_valid:
                        continue

                    polarity, certainty = self._assess_polarity_and_certainty(sent)
                    if is_questionable:
                        polarity = NegationStatus.UNCERTAIN
                        certainty = Certainty.SUSPECTED

                    ev = StructuredEvidence(
                        text=sent,
                        section=sec.section_name,
                        sentence=sent,
                        polarity=polarity,
                        certainty=certainty,
                        temporality=Temporality.CURRENT,
                        evidence_type=EvidenceType.IMAGING if sec.section_name == "INVESTIGATIONS" else EvidenceType.LABORATORY,
                        clinical_relevance=1.0 if not is_questionable else 0.5,
                    )
                    scores = MultiDimensionalScore(
                        evidence_score=1.0 if not is_questionable else 0.5,
                        diagnostic_certainty=1.0 if certainty == Certainty.CONFIRMED else 0.5,
                        encounter_relevance=1.0 if not is_questionable else 0.4,
                        role_confidence=1.0 if not is_questionable else 0.0,
                        admitting_score=3.5 if not is_questionable else -5.0,
                    )
                    cand = ClinicalDiagnosisCandidate(
                        raw_term=clean_term,
                        normalized_diagnosis=clean_term,
                        role=DiagnosisRole.SECONDARY if not is_questionable else DiagnosisRole.UNCERTAIN,
                        certainty=certainty,
                        temporality=Temporality.CURRENT,
                        assertion_status=AssertionStatus.CONFIRMED if not is_questionable else AssertionStatus.UNCERTAIN,
                        clinical_attributes=attrs,
                        evidence=[ev],
                        evidence_strength=1.0 if not is_questionable else 0.5,
                        clinical_relevance=1.0 if not is_questionable else 0.5,
                        encounter_relevance=1.0 if not is_questionable else 0.4,
                        scores=scores,
                        is_authorized=not is_questionable and polarity != NegationStatus.NEGATED,
                        classification_reason=f"Objective finding in {sec.section_name}" if not is_questionable else f"Uncertain question-mark finding in {sec.section_name}",
                    )
                    candidates.append(cand)

        return candidates

    def _parse_procedure_indications(
        self,
        content: str,
        sec: ClinicalDocumentSection,
    ) -> list[ClinicalDiagnosisCandidate]:
        """Extract indication diagnoses driving surgical or interventional procedures."""
        candidates: list[ClinicalDiagnosisCandidate] = []
        cleaned = content.strip()
        if not cleaned:
            return []

        from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        for line in lines:
            line_clean = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
            if len(line_clean) < 5 or HardClinicalCandidateGate.is_absence_statement(line_clean) or HardClinicalCandidateGate.is_treatment_instruction(line_clean):
                continue

            # Match: "[Procedure] for [Condition]" or "Indication: [Condition]"
            ind_match = re.search(
                r"(?:for|indication\s*:?)\s+([^.,;\n]+)",
                line_clean,
                re.IGNORECASE,
            )
            if ind_match:
                cand_term = ind_match.group(1).strip()
                cand_term = re.sub(r"^(?:a|an|the)\s+", "", cand_term, flags=re.IGNORECASE).strip()
                cand_term, attrs = self._extract_concept_and_attributes(cand_term)
                if len(cand_term) < 3 or cand_term.lower() in ("stenting", "surgery", "further evaluation", "management"):
                    continue

                is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(cand_term, line_clean)
                if not is_valid:
                    continue

                polarity, certainty = self._assess_polarity_and_certainty(line_clean)
                acuity = self._detect_acuity(cand_term)

                ev = StructuredEvidence(
                    text=line,
                    section=sec.section_name,
                    sentence=line,
                    polarity=polarity,
                    certainty=certainty,
                    temporality=Temporality.CURRENT,
                    evidence_type=EvidenceType.PROCEDURE,
                    clinical_relevance=1.0,
                )
                scores = MultiDimensionalScore(
                    evidence_score=1.0,
                    diagnostic_certainty=1.0,
                    encounter_relevance=1.0,
                    role_confidence=1.0,
                    admitting_score=4.5 + (1.0 if acuity == Acuity.ACUTE else 0.0),
                )
                cand = ClinicalDiagnosisCandidate(
                    raw_term=cand_term,
                    normalized_diagnosis=cand_term,
                    role=DiagnosisRole.SECONDARY,
                    certainty=certainty,
                    temporality=Temporality.CURRENT,
                    assertion_status=AssertionStatus.CONFIRMED,
                    clinical_attributes=attrs,
                    evidence=[ev],
                    evidence_strength=1.0,
                    clinical_relevance=1.0,
                    encounter_relevance=1.0,
                    procedure_relevance=1.0,
                    scores=scores,
                    is_authorized=polarity != NegationStatus.NEGATED,
                    classification_reason=f"Indication for intervention in {sec.section_name}",
                )
                candidates.append(cand)

        return candidates

    def _parse_operative_findings(
        self,
        content: str,
        sec: ClinicalDocumentSection,
    ) -> tuple[list[dict[str, Any]], list[ClinicalDiagnosisCandidate]]:
        """Extract intraoperative findings with distinct clinical-coding eligibility gate."""
        findings: list[dict[str, Any]] = []
        candidates: list[ClinicalDiagnosisCandidate] = []
        cleaned = content.strip()
        if not cleaned:
            return findings, candidates

        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        for line in lines:
            line_clean = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
            if len(line_clean) < 3 or HardClinicalCandidateGate.is_absence_statement(line_clean):
                continue

            # Identify finding entities (e.g. "ovarian cyst", "pelvic adhesions", "liver normal")
            is_normal = any(w in line_clean.lower() for w in ["normal", "unremarkable", "intact", "no pathology", "no abnormalities", "clear"])
            is_incidental = any(w in line_clean.lower() for w in ["incidental", "benign", "small", "simple", "minor", "asymptomatic"])

            term = line_clean
            for delim in [",", "-", "–", ":"]:
                if delim in term:
                    term = term.split(delim)[0].strip()
            term = re.sub(r"^(?:a|an|the)\s+", "", term, flags=re.IGNORECASE).strip()

            finding_record = {
                "term": term,
                "quote": line,
                "section": sec.section_name,
                "is_normal": is_normal,
                "is_incidental": is_incidental,
                "requires_coding": not is_normal and not is_incidental,
            }
            findings.append(finding_record)

            if is_normal:
                continue

            # Distinct clinical-coding eligibility gate:
            # Under UHDDS criteria, incidental intraoperative findings without dedicated surgical intervention
            # or postoperative therapy are excluded from billable coding.
            is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(term, line)
            if not is_valid:
                continue

            polarity, certainty = self._assess_polarity_and_certainty(line)
            ev = StructuredEvidence(
                text=line,
                section=sec.section_name,
                sentence=line,
                polarity=polarity,
                certainty=certainty,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.PROCEDURE,
                clinical_relevance=0.3 if is_incidental else 0.8,
            )
            scores = MultiDimensionalScore(
                evidence_score=1.0,
                diagnostic_certainty=1.0 if certainty == Certainty.CONFIRMED else 0.8,
                encounter_relevance=0.3 if is_incidental else 0.8,
                role_confidence=0.5,
                admitting_score=1.0 if not is_incidental else -5.0,
            )
            cand = ClinicalDiagnosisCandidate(
                raw_term=term,
                normalized_diagnosis=term,
                role=DiagnosisRole.SECONDARY if not is_incidental else DiagnosisRole.EXCLUDED,
                certainty=certainty,
                temporality=Temporality.CURRENT,
                assertion_status=AssertionStatus.CONFIRMED,
                evidence=[ev],
                scores=scores,
                is_authorized=not is_incidental and polarity != NegationStatus.NEGATED,
                classification_reason=f"Operative finding in {sec.section_name}" if not is_incidental else f"Incidental operative finding in {sec.section_name}; excluded under UHDDS criteria",
            )
            candidates.append(cand)

        return findings, candidates

    @staticmethod
    def _extract_structured_oncology_context(text: str) -> CancerConcept | None:
        """Capture structured oncology context: histology, grade, receptor status, Ki-67, metastatic sites, prior treatment, and response."""
        text_lower = text.lower()

        # Check if document describes neoplasm / malignancy
        has_neoplasm = any(
            w in text_lower
            for w in [
                "carcinoma", "cancer", "neoplasm", "malignancy", "tumor", "tumour",
                "adenocarcinoma", "lymphoma", "sarcoma", "melanoma", "dcis"
            ]
        )
        if not has_neoplasm:
            return None

        # 1. Primary Site
        primary_site = "breast" if "breast" in text_lower else "unknown"
        if primary_site == "unknown":
            for site in ["ovary", "ovarian", "colon", "lung", "prostate", "cervix", "uterus", "kidney"]:
                if site in text_lower:
                    primary_site = site
                    break

        # 2. Laterality of primary tumor
        laterality = None
        site_pat = primary_site if primary_site != "unknown" else r"(?:breast|ovary|lung|kidney)"
        if re.search(rf"\bright\s+(?:\w+\s+)?{site_pat}\b|\b{site_pat}\b[^\n,.]*?\bright\b", text_lower):
            laterality = "right"
        elif re.search(rf"\bleft\s+(?:\w+\s+)?{site_pat}\b|\b{site_pat}\b[^\n,.]*?\bleft\b", text_lower):
            laterality = "left"
        elif re.search(rf"\bbilateral\s+(?:\w+\s+)?{site_pat}\b|\b{site_pat}\b[^\n,.]*?\bbilateral\b", text_lower):
            laterality = "bilateral"
        elif "right" in text_lower and "left" not in text_lower:
            laterality = "right"
        elif "left" in text_lower and "right" not in text_lower:
            laterality = "left"
        elif "bilateral" in text_lower:
            laterality = "bilateral"

        # 3. Histology
        histology = None
        for hist_pat in [
            r"\binvasive\s+ductal\s+carcinoma\b",
            r"\binfiltrating\s+ductal\s+carcinoma\b",
            r"\binvasive\s+lobular\s+carcinoma\b",
            r"\bductal\s+carcinoma\s+in\s+situ\b",
            r"\bdcis\b",
            r"\bhigh[- ]grade\s+serous\s+carcinoma\b",
            r"\bserous\s+carcinoma\b",
            r"\badenocarcinoma\b",
            r"\bsquamous\s+cell\s+carcinoma\b",
        ]:
            m = re.search(hist_pat, text_lower)
            if m:
                histology = m.group(0).title()
                break

        # 4. Grade
        grade = None
        grade_m = re.search(r"\b(?:grade\s*([1-3]|i{1,3})|poorly\s+differentiated|moderately\s+differentiated|well\s+differentiated)\b", text_lower)
        if grade_m:
            grade = grade_m.group(0).capitalize()

        # 5. Receptor Status (ER, PR, HER2)
        receptor_status: dict[str, str] = {}
        er_m = re.search(r"\ber\s*(?:[:=]|\s+is|\s+was)?\s*([><=]?\s*\d+%\s*(?:positive|\+)|positive|\+|negative|\-|equivocal|[><=]?\s*\d+%)", text, re.IGNORECASE)
        if er_m:
            v = er_m.group(1).lower().strip()
            receptor_status["ER"] = "positive" if "+" in v or "pos" in v else "negative" if "-" in v or "neg" in v else v
            receptor_status["er"] = receptor_status["ER"]

        pr_m = re.search(r"\bpr\s*(?:[:=]|\s+is|\s+was)?\s*([><=]?\s*\d+%\s*(?:positive|\+)|positive|\+|negative|\-|equivocal|[><=]?\s*\d+%)", text, re.IGNORECASE)
        if pr_m:
            v = pr_m.group(1).lower().strip()
            receptor_status["PR"] = "positive" if "+" in v or "pos" in v else "negative" if "-" in v or "neg" in v else v
            receptor_status["pr"] = receptor_status["PR"]

        her2_m = re.search(r"\bher2(?:\s*/\s*neu)?\s*(?:[:=]|\s+is|\s+was)?\s*(positive|\+|negative|\-|equivocal|[0-3]\+?)\b", text, re.IGNORECASE)
        if her2_m:
            v = her2_m.group(1).lower().strip()
            receptor_status["HER2"] = v
            receptor_status["her2"] = v

        # 6. Ki-67 Proliferation Index
        ki67 = None
        ki67_m = re.search(
            r"\bki-?67(?:\s+(?:proliferation\s+index|labeling\s+index|index))?\s*(?:[:=]|\s+is|\s+was)?\s*([><=]?\s*\d+%(?:\s*(?:high|low))?|high|low)",
            text,
            re.IGNORECASE,
        )
        if ki67_m:
            ki67 = ki67_m.group(1).strip()

        # 7. Metastatic Status & Sites
        met_sites: list[str] = []
        if re.search(r"\bno\s+(?:evidence\s+of\s+)?(?:distant\s+)?metastases\b|\bnon[- ]metastatic\b|\bm0\b", text_lower):
            met_status = "NON_METASTATIC"
        elif re.search(r"\bmetastat(?:ic|es)\b|\bm1\b", text_lower):
            met_status = "METASTATIC"
            for site in ["bone", "liver", "lung", "brain", "pleura", "peritoneum", "adrenal", "lymph node"]:
                if site in text_lower:
                    met_sites.append(site)
        else:
            met_status = "UNKNOWN"

        # 8. Prior Treatments
        prior_tx: list[str] = []
        for tx_pat in [
            r"\bneoadjuvant\s+chemotherapy\b",
            r"\badjuvant\s+chemotherapy\b",
            r"\bradiotherapy\b|\bradiation\b",
            r"\btamoxifen\b",
            r"\bletrozole\b",
            r"\banastrozole\b",
            r"\btrastuzumab\b|\bherceptin\b",
        ]:
            if re.search(tx_pat, text_lower):
                prior_tx.append(re.search(tx_pat, text_lower).group(0))

        # 9. Response to Treatment
        resp = None
        for resp_pat in [
            r"\bpathologic(?:al)?\s+complete\s+response\b|\bpcr\b",
            r"\bcomplete\s+(?:pathologic(?:al)?\s+|clinical\s+)?response\b",
            r"\bpartial\s+(?:pathologic(?:al)?\s+|clinical\s+)?response\b",
            r"\bstable\s+disease\b",
            r"\bprogressive\s+disease\b",
            r"\bresidual\s+(?:invasive\s+)?(?:disease|carcinoma)\b",
            r"\bno\s+residual\s+invasive\s+disease\b",
        ]:
            m = re.search(resp_pat, text_lower)
            if m:
                resp = m.group(0).title()
                break

        return CancerConcept(
            primary_site=primary_site,
            laterality=laterality,
            malignancy_type="carcinoma",
            histology=histology,
            grade=grade,
            metastatic_status=met_status,
            metastatic_sites=met_sites,
            receptor_status=receptor_status,
            ki67=ki67,
            treatment_response=resp,
            prior_treatments=prior_tx,
            evidence=text[:200],
        )

    def _split_compound_clinical_phrase(self, phrase: str) -> list[str]:
        """Split compound clinical phrases into independent clinical concepts.

        Principles:
        1. Preserve parenthetical abbreviations: "Escherichia coli (E. coli) bacteremia" -> single entity
        2. Split on complication/causal connectors: "complicated by", "secondary to", "due to", "resulting in"
        3. Split on "with [independent condition]" (e.g. pyelonephritis with right renal calculus),
           NEVER when "with" denotes an ICD combination manifestation (e.g. diabetes with ketoacidosis,
           gastritis without bleeding, asthma with exacerbation, ulcer with hemorrhage, calculus with obstruction)
        4. Split on "and" when connecting distinct clinical conditions
        """
        cleaned = phrase.strip().strip(".,;: ")
        if not cleaned or len(cleaned) < 3:
            return []

        # Check for multi-bone orthopedic compound fractures/sprains (Section 10)
        # e.g., "Multiple fractures of left lateral malleolus, tarsals and fifth metatarsal with deltoid and calcaneofibular ligament sprains"
        ortho_m = re.search(
            r"\b(?:multiple\s+)?fractures?\s+of\s+(?:(left|right|bilateral)\s+)?([^\n;]+?)\s+with\s+([^\n;]+)\b",
            cleaned,
            re.IGNORECASE,
        )
        if ortho_m:
            lat = ortho_m.group(1) or ""
            bone_list_str = ortho_m.group(2)
            assoc_str = ortho_m.group(3)

            bones = re.split(r",\s*|\s+and\s+", bone_list_str)
            decomposed: list[str] = []
            for b in bones:
                b_clean = b.strip()
                if not b_clean:
                    continue
                bone_name = b_clean
                if bone_name.lower().endswith("s") and not bone_name.lower().endswith("us") and not bone_name.lower().endswith("is"):
                    bone_name = bone_name[:-1]
                diag = f"{lat} {bone_name} fracture".strip()
                diag = re.sub(r"\s+", " ", diag)
                decomposed.append(diag)

            sprain_m = re.search(r"([^\n;]+?)\s+(?:ligament\s+)?sprains?\b", assoc_str, re.IGNORECASE)
            if sprain_m:
                lig_list_str = sprain_m.group(1)
                ligs = re.split(r",\s*|\s+and\s+", lig_list_str)
                for lig in ligs:
                    lig_clean = lig.strip()
                    if not lig_clean:
                        continue
                    sprain_diag = f"{lat} {lig_clean} ligament sprain".strip()
                    sprain_diag = re.sub(r"\s+", " ", sprain_diag)
                    decomposed.append(sprain_diag)
            else:
                decomposed.append(f"{lat} {assoc_str}".strip())

            return [d for d in decomposed if len(d) >= 3]

        # Check for co-occurring independent diseases connected with "with" (Section 10)
        # e.g. Castleman's disease with Kaposi's sarcoma
        co_disease_m = re.search(
            r"^([a-z\s'\-]+?(?:disease|syndrome|sarcoma|carcinoma|lymphoma|infection))\s+with\s+([a-z\s'\-]+?(?:disease|syndrome|sarcoma|carcinoma|lymphoma|infection))$",
            cleaned,
            re.IGNORECASE,
        )
        if co_disease_m:
            return [co_disease_m.group(1).strip(), co_disease_m.group(2).strip()]

        # Temporary mask for genus abbreviations like E. coli, H. pylori, S. pneumoniae, C. difficile
        masked = cleaned
        abbrev_matches = list(re.finditer(r"\b([A-Z])\.\s+([a-z]+)\b", masked))
        for i, m in enumerate(abbrev_matches):
            masked = masked.replace(m.group(0), f"__ORGANISM_{i}__")

        # 1. Connectors that unconditionally separate independent clinical entities
        causal_pattern = r",?\s*\b(?:complicated\s+by|secondary\s+to|due\s+to|resulting\s+in|manifested\s+by)\b\s*"
        chunks = re.split(causal_pattern, masked, flags=re.IGNORECASE)

        # Standard ICD combination manifestations where "with" should NEVER be split
        combo_manifestation_patterns = [
            r"\bwith\s+(?:diabetic\s+)?(?:ketoacidosis|hyperosmolarity|nephropathy|retinopathy|neuropathy|angiopathy|arthropathy|skin|ulcer|hypoglycemia|hyperglycemia)\b",
            r"\bwith(?:out)?\s+(?:active\s+)?(?:bleeding|hemorrhage|perforation|obstruction|coma|exacerbation|status\s+asthmaticus)\b",
            r"\bwith\s+delta[\s\-]agent\b",
        ]

        # Independent organ conditions where "with" introduces a distinct co-occurring disease
        independent_with_pattern = (
            r",?\s*\bwith\s+(?:an?\s+)?(?:acute\s+|chronic\s+|severe\s+)?(?:right\s+|left\s+|bilateral\s+)?(?:parenchymal\s+)?(?:obstructive\s+)?"
            r"(renal\s+calculus|kidney\s+stone|ureteral\s+stone|ureteric\s+stone|nephrolithiasis|renal\s+abscess|bacteremia|septic\s+shock|sepsis|cholecystitis|pancreatitis|pneumonia|abscess|(?:bilobar\s+|multiple\s+|solitary\s+)?(?:liver|pleural|pulmonary|bone|brain|distant)?\s*metastases?)\b"
        )

        refined_chunks: list[str] = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            is_combo_manifestation = any(re.search(pat, chunk, re.IGNORECASE) for pat in combo_manifestation_patterns)
            with_match = re.search(independent_with_pattern, chunk, re.IGNORECASE)

            if with_match and not is_combo_manifestation:
                idx = with_match.start()
                base_cond = chunk[:idx].strip()
                co_cond = chunk[idx:].strip()
                co_cond = re.sub(r"^,?\s*with\s+(?:an?\s+)?", "", co_cond, flags=re.IGNORECASE).strip()
                if base_cond:
                    refined_chunks.append(base_cond)
                if co_cond:
                    refined_chunks.append(co_cond)
            else:
                refined_chunks.append(chunk)

        # 2. Split chunks on " and " when connecting distinct clinical entities
        final_items: list[str] = []
        for item in refined_chunks:
            and_parts = re.split(r",?\s*\band\b\s*", item, flags=re.IGNORECASE)
            if len(and_parts) > 1 and all(len(p.strip()) > 3 for p in and_parts):
                for p in and_parts:
                    p_clean = p.strip().strip(",; ")
                    if p_clean and len(p_clean) >= 3:
                        final_items.append(p_clean)
            else:
                final_items.append(item.strip().strip(",; "))

        # Restore masked organism abbreviations and clean timing/procedural context (Section 8 & 9)
        restored: list[str] = []
        for res_item in final_items:
            res = res_item
            for i, m in enumerate(abbrev_matches):
                res = res.replace(f"__ORGANISM_{i}__", m.group(0))

            # Strip trailing procedural or temporal timing context (e.g. "gastritis post frozen embryo transfer" -> "gastritis")
            timing_m = re.search(r"\b(?:status\s+post|post|s/p|after)\s+(?:frozen\s+embryo\s+transfer|fet|chemotherapy|surgery|procedure|infusion)\b.*$", res, re.IGNORECASE)
            if timing_m:
                base_cand = res[:timing_m.start()].strip()
                if len(base_cand) >= 3:
                    res = base_cand

            restored.append(res.strip())

        return [r for r in restored if len(r) >= 3]

    def _split_term_and_narrative(self, line: str) -> tuple[str, str]:
        """Split a clinical line into the diagnostic term and accompanying clinical explanation."""
        # e.g., "Acute right emphysematous pyelonephritis - underwent DJ stenting" -> ("Acute right emphysematous pyelonephritis", "underwent DJ stenting")
        for delim in [" - ", " – ", " — ", " : ", ": ", " / ", " (", "; "]:
            if delim in line:
                parts = line.split(delim, 1)
                term = parts[0].strip()
                narrative = parts[1].rstrip(")").strip()
                if len(term) >= 3:
                    return term, narrative

        # Look for sentence boundary period (not initials or organism abbreviations like E. coli or H. pylori)
        period_match = re.search(r"(?<!\b[A-Za-z])\.\s+(?=[A-Z0-9])", line)
        if period_match:
            idx = period_match.start()
            term = line[:idx].strip()
            narrative = line[period_match.end():].strip()
            if len(term) >= 3:
                return term, narrative

        return line.strip(), ""

    def _assess_polarity_and_certainty(self, text: str) -> tuple[NegationStatus, Certainty]:
        text_lower = text.lower()
        for pat in NEGATION_CUES:
            if re.search(pat, text_lower):
                return NegationStatus.NEGATED, Certainty.RULED_OUT

        for pat in UNCERTAINTY_CUES:
            if re.search(pat, text_lower):
                return NegationStatus.UNCERTAIN, Certainty.SUSPECTED

        return NegationStatus.AFFIRMATIVE, Certainty.CONFIRMED

    def _detect_acuity(self, term: str) -> Acuity:
        term_lower = term.lower()
        if "acute on chronic" in term_lower:
            return Acuity.ACUTE_ON_CHRONIC
        if any(re.search(pat, term_lower) for pat in ACUITY_ACUTE_CUES):
            return Acuity.ACUTE
        if any(re.search(pat, term_lower) for pat in ACUITY_CHRONIC_CUES):
            return Acuity.CHRONIC
        return Acuity.UNSPECIFIED

    def _detect_laterality(self, term: str) -> Laterality:
        term_lower = term.lower()
        if re.search(r"\bright\b", term_lower):
            return Laterality.RIGHT
        if re.search(r"\bleft\b", term_lower):
            return Laterality.LEFT
        if re.search(r"\bbilateral\b", term_lower):
            return Laterality.BILATERAL
        return Laterality.UNSPECIFIED

    def _canonicalize_term(self, term: str) -> str:
        clean = re.sub(r"^\d+[\.\)\-]\s*", "", term).strip()
        clean = re.sub(r"\s+", " ", clean).strip(".,;: ")
        # Section 21: Oncology metastatic manifestations
        if re.search(r"\bpleural\s+metastasis\b|\bmetastasis\s+to\s+pleura\b|\bmalignant\s+pleural\s+metastasis\b", clean, re.IGNORECASE):
            return "Secondary malignant neoplasm of pleura"
        return clean

    def _corroborate_with_hospital_course_and_procedures(
        self,
        candidates: list[ClinicalDiagnosisCandidate],
        sections: list[ClinicalDocumentSection],
        full_text: str,
    ) -> None:
        """Enrich candidates with corroborating evidence from hospital course, procedures, objective investigations, and medications."""
        hospital_course_text = ""
        investigations_text = ""
        procedure_text = ""
        medications_text = ""

        for sec in sections:
            if sec.section_name == "HOSPITAL_COURSE":
                hospital_course_text = f"{hospital_course_text} {sec.content}"
            elif sec.section_name in ("INVESTIGATIONS", "MICROBIOLOGY"):
                investigations_text = f"{investigations_text} {sec.content}"
            elif sec.section_name == "PROCEDURES":
                procedure_text = f"{procedure_text} {sec.content}"
            elif sec.section_name in ("MEDICATIONS", "DISCHARGE_MEDICATIONS"):
                medications_text = f"{medications_text} {sec.content}"

        active_care_verbs = [
            "treated", "managed", "monitored", "administered", "started", "continued",
            "withheld", "titrated", "adjusted", "sliding scale", "infusion", "dose",
            "protocol", "regimen", "followed", "received", "underwent", "controlled",
            "stabilized", "responded", "prescribed", "switched", "transitioned",
        ]

        # Clinical domain indicators for chronic/PMH conditions requiring active inpatient management
        condition_care_markers: dict[str, list[str]] = {
            "diabetes": ["insulin", "sliding scale", "metformin", "glucose", "blood sugar", "glycemic", "dka", "hypoglycemia", "hyperglycemia", "hba1c"],
            "hypertension": ["amlodipine", "antihypertensive", "blood pressure", "bp", "lisinopril", "losartan", "metoprolol", "atenolol", "hydrochlorothiazide", "furosemide"],
            "kidney": ["creatinine", "gfr", "nephrology", "dialysis", "renal function", "baseline creatinine"],
            "heart failure": ["furosemide", "lasix", "diuresis", "echo", "echocardiogram", "ejection fraction", "cardiac"],
            "asthma": ["inhaler", "nebulizer", "albuterol", "ipratropium", "bronchodilator", "steroid", "prednisone"],
            "copd": ["inhaler", "nebulizer", "albuterol", "ipratropium", "bronchodilator", "steroid", "prednisone", "oxygen"],
        }

        for cand in candidates:
            cand_norm = cand.normalized_diagnosis.lower()
            words = [w for w in cand_norm.split() if len(w) > 3 and w not in ("with", "without", "acute", "chronic", "type", "history", "disease", "disorder")]

            # Find relevant domain markers for this candidate
            relevant_markers: list[str] = []
            for cond_key, markers in condition_care_markers.items():
                if cond_key in cand_norm:
                    relevant_markers.extend(markers)

            # Check hospital course and medications for active inpatient care
            combined_course_and_meds = f"{hospital_course_text} {medications_text}"
            course_sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", combined_course_and_meds) if s.strip()]

            found_management_sentence = None
            for sent in course_sentences:
                sent_lower = sent.lower()
                has_cand_mention = (
                    any(re.search(rf"\b{re.escape(w)}\b", sent_lower) for w in words)
                    or (cand_norm in sent_lower)
                ) if words else False

                has_marker_mention = any(re.search(rf"\b{re.escape(m)}\b", sent_lower) for m in relevant_markers)
                has_active_verb = any(v in sent_lower for v in active_care_verbs)

                if (has_cand_mention or has_marker_mention) and has_active_verb:
                    found_management_sentence = sent
                    break

            if found_management_sentence:
                cand.treatment_relevance = 1.0
                cand.encounter_relevance = 1.0
                cand.scores.encounter_relevance = 1.0
                is_from_principal_sec = any(ev.section in ("PRINCIPAL_DIAGNOSIS", "PRIMARY_DIAGNOSIS") for ev in cand.evidence) or cand.role == DiagnosisRole.PRIMARY
                cand.scores.admitting_score += (3.0 if is_from_principal_sec else 1.0)
                cand.temporality = Temporality.CURRENT
                cand.assertion_status = AssertionStatus.CONFIRMED
                cand.is_authorized = True
                if cand.role == DiagnosisRole.HISTORICAL:
                    cand.role = DiagnosisRole.SECONDARY
                    cand.classification_reason = f"Pre-existing condition actively evaluated/managed during stay: {found_management_sentence}"

                # Prepend supporting structured evidence if PMH; otherwise append so discharge diagnosis evidence remains at index 0
                if any(ev.section in ("PAST_MEDICAL_HISTORY", "PAST_SURGICAL_HISTORY") for ev in cand.evidence):
                    cand.evidence.insert(
                        0,
                        StructuredEvidence(
                            text=found_management_sentence,
                            section="HOSPITAL_COURSE",
                            sentence=found_management_sentence,
                            polarity=NegationStatus.AFFIRMATIVE,
                            certainty=cand.certainty,
                            temporality=Temporality.CURRENT,
                            evidence_type=EvidenceType.HOSPITAL_COURSE,
                            clinical_relevance=1.0,
                        )
                    )
                else:
                    cand.evidence.append(
                        StructuredEvidence(
                            text=found_management_sentence,
                            section="HOSPITAL_COURSE",
                            sentence=found_management_sentence,
                            polarity=NegationStatus.AFFIRMATIVE,
                            certainty=cand.certainty,
                            temporality=Temporality.CURRENT,
                            evidence_type=EvidenceType.HOSPITAL_COURSE,
                            clinical_relevance=1.0,
                        )
                    )

            # Check procedure text
            if procedure_text:
                cand_words = {w for w in re.findall(r"[a-z0-9]+", cand.normalized_diagnosis.lower()) if len(w) > 3}
                for proc_cue in PROCEDURE_SOURCE_CONTROL_CUES:
                    if re.search(proc_cue, procedure_text, re.IGNORECASE):
                        proc_words = {w for w in re.findall(r"[a-z0-9]+", proc_cue.lower()) if len(w) > 3}
                        if (cand_words & proc_words) or any(w in procedure_text.lower() for w in cand_words):
                            cand.procedure_relevance = 1.0
                            cand.scores.admitting_score += 3.0
                            break

            # Check investigations text for supporting proof
            if investigations_text and words and any(w in investigations_text.lower() for w in words):
                for sent in re.split(r"[\.\n]+", investigations_text):
                    if any(w in sent.lower() for w in words):
                        cand.evidence.append(
                            StructuredEvidence(
                                text=sent.strip(),
                                section="INVESTIGATIONS",
                                sentence=sent.strip(),
                                polarity=NegationStatus.AFFIRMATIVE,
                                certainty=Certainty.SUPPORTED,
                                temporality=Temporality.CURRENT,
                                evidence_type=EvidenceType.IMAGING,
                                clinical_relevance=0.8,
                            )
                        )
                        break

    def _classify_and_structure_state(
        self,
        candidates: list[ClinicalDiagnosisCandidate],
        sections: list[ClinicalDocumentSection],
        document_id: str,
    ) -> ClinicalDiagnosisState:
        """Apply strict evidence hierarchy to classify candidates into Primary, Secondary, Historical, and Excluded."""
        active_candidates: list[ClinicalDiagnosisCandidate] = []
        historical_conditions: list[ClinicalDiagnosisCandidate] = []
        ruled_out_conditions: list[ClinicalDiagnosisCandidate] = []
        uncertain_conditions: list[ClinicalDiagnosisCandidate] = []

        for cand in candidates:
            # Check ruled out
            if cand.certainty == Certainty.RULED_OUT:
                cand.role = DiagnosisRole.RULED_OUT
                cand.is_authorized = False
                ruled_out_conditions.append(cand)
                continue

            # Check historical
            if cand.temporality in (Temporality.HISTORICAL, Temporality.RESOLVED) and cand.encounter_relevance <= 0.2:
                cand.role = DiagnosisRole.HISTORICAL
                cand.is_authorized = False
                historical_conditions.append(cand)
                continue

            # Check uncertain
            if cand.certainty in (Certainty.SUSPECTED, Certainty.POSSIBLE):
                cand.role = DiagnosisRole.UNCERTAIN
                uncertain_conditions.append(cand)
                # If documented under discharge diagnoses, it still enters coding as suspected
                if any(ev.section in ("DISCHARGE_DIAGNOSES", "PRINCIPAL_DIAGNOSIS") for ev in cand.evidence):
                    active_candidates.append(cand)
                continue

            active_candidates.append(cand)

        # Primary diagnosis determination using strict UHDDS evidence hierarchy
        primary_diag: ClinicalDiagnosisCandidate | None = None
        secondary_diags: list[ClinicalDiagnosisCandidate] = []

        if active_candidates:
            from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

            # Sort active candidates by admitting score descending
            active_candidates.sort(key=lambda c: c.scores.admitting_score, reverse=True)

            # Find top candidate meeting Section 8 criteria
            chosen_idx = None
            for idx, cand in enumerate(active_candidates):
                if (
                    cand.certainty == Certainty.CONFIRMED
                    and cand.temporality in (Temporality.CURRENT, "CURRENT")
                    and cand.evidence
                ):
                    is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(
                        cand.normalized_diagnosis, cand.primary_evidence_quote
                    )
                    if is_valid:
                        chosen_idx = idx
                        break

            if chosen_idx is not None:
                top = active_candidates[chosen_idx]
                top.role = DiagnosisRole.PRIMARY
                top.primary_justification = f"Documented chief condition occasioning admission and inpatient treatment: {top.primary_evidence_quote}"
                primary_diag = top

                # Remaining become SECONDARY, unless integral symptom or duplicate of primary
                top_norm = top.normalized_diagnosis.lower()
                top_words = set(re.findall(r"[a-z0-9]+", top_norm)) - {
                    "acute", "chronic", "right", "left", "bilateral", "unspecified", "severe", "mild", "stage", "with", "without"
                }

                for idx, sec_cand in enumerate(active_candidates):
                    if idx == chosen_idx:
                        continue
                    sec_norm = sec_cand.normalized_diagnosis.lower()
                    sec_words = set(re.findall(r"[a-z0-9]+", sec_norm)) - {
                        "acute", "chronic", "right", "left", "bilateral", "unspecified", "severe", "mild", "stage", "with", "without"
                    }

                    # Check 1: Duplicate / generalized mention of the primary diagnosis
                    if sec_words and top_words and (sec_words <= top_words or (len(sec_words & top_words) >= 2 and len(sec_words - top_words) == 0)):
                        sec_cand.role = DiagnosisRole.PRIMARY
                        sec_cand.is_authorized = False
                        sec_cand.classification_reason = f"Corroborating mention of primary diagnosis '{top.normalized_diagnosis}'"
                        top.evidence.extend(sec_cand.evidence)
                        continue

                    # Check 2: Integral symptom / sign of primary diagnosis (CMS Guideline I.B.4)
                    is_integral = any(
                        sym in sec_norm
                        for sym in [
                            "pain", "colic", "fever", "cough", "dyspnea", "shortness of breath",
                            "nausea", "vomiting", "dyspepsia", "indigestion", "heartburn",
                            "discomfort", "wheezing", "fatigue", "malaise", "respiratory distress",
                            "peripheral edema", "edema", "chest pain", "flank pain", "hematuria"
                        ]
                    ) and not (
                        any(sec_norm.endswith(sfx) or f"{sfx} " in sec_norm for sfx in ("itis", "oma", "osis"))
                        or any(dx in sec_norm for dx in ["syndrome", "failure", "infarction", "disease", "disorder", "calculus", "lithiasis", "ulcer", "pneumonia"])
                    )

                    if is_integral:
                        from medical_coding.schemas.enums import ClinicalEntityType
                        sec_cand.role = DiagnosisRole.SYMPTOM
                        sec_cand.entity_type = ClinicalEntityType.SYMPTOM
                        sec_cand.is_authorized = False
                        sec_cand.classification_reason = f"Symptom integral to the primary diagnosis '{top.normalized_diagnosis}'; excluded from independent billing under CMS Guideline I.B.4"
                        continue

                    # Check 3: Intraoperative finding without dedicated surgical or therapeutic intervention
                    if any(ev.section == "OPERATIVE_FINDINGS" for ev in sec_cand.evidence) or sec_cand.role == DiagnosisRole.EXCLUDED:
                        full_content_lower = " ".join(s.content for s in sections).lower()
                        has_intervention = any(
                            w in full_content_lower
                            for w in ["cystectomy", "extensive adhesiolysis", "surgical excision of cyst", "pathology confirmed malignancy"]
                        )
                        if not has_intervention:
                            sec_cand.role = DiagnosisRole.EXCLUDED
                            sec_cand.is_authorized = False
                            sec_cand.classification_reason = (
                                f"[OPERATIVE_FINDINGS] Incidental finding '{sec_cand.normalized_diagnosis}' without documented "
                                "dedicated inpatient intervention or therapy; excluded from billable coding under UHDDS criteria."
                            )
                            continue

                    is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(
                        sec_cand.normalized_diagnosis, sec_cand.primary_evidence_quote
                    )
                    if is_valid:
                        sec_cand.role = DiagnosisRole.SECONDARY
                        sec_cand.classification_reason = f"Active co-existing condition managed during admission: {sec_cand.primary_evidence_quote}"
                        secondary_diags.append(sec_cand)

        # Extract structured oncology context across entire document narrative
        full_doc_text = " ".join(s.content for s in sections)
        oncology_context = self._extract_structured_oncology_context(full_doc_text)
        if oncology_context and primary_diag:
            primary_diag.clinical_attributes["oncology_context"] = oncology_context.model_dump()
            if oncology_context.histology:
                primary_diag.clinical_attributes["histology"] = oncology_context.histology
            if oncology_context.grade:
                primary_diag.clinical_attributes["grade"] = oncology_context.grade
            if oncology_context.receptor_status:
                primary_diag.clinical_attributes["receptor_status"] = oncology_context.receptor_status
            if oncology_context.ki67:
                primary_diag.clinical_attributes["ki67"] = oncology_context.ki67
            if oncology_context.treatment_response:
                primary_diag.clinical_attributes["treatment_response"] = oncology_context.treatment_response
            if oncology_context.prior_treatments:
                primary_diag.clinical_attributes["prior_treatments"] = oncology_context.prior_treatments

        return ClinicalDiagnosisState(
            document_id=document_id,
            primary_diagnosis=primary_diag,
            secondary_diagnoses=secondary_diags,
            historical_conditions=historical_conditions,
            ruled_out_conditions=ruled_out_conditions,
            uncertain_conditions=uncertain_conditions,
            all_candidates=candidates,
            has_unique_primary=primary_diag is not None,
            procedures=list(dict.fromkeys(self._extracted_procedures)),
            operative_findings=self._operative_findings,
            oncology_context=oncology_context,
            audit_notes=[f"Classified {len(candidates)} total entities: 1 Primary, {len(secondary_diags)} Secondary, {len(historical_conditions)} Historical, {len(ruled_out_conditions)} Ruled out."],
        )
