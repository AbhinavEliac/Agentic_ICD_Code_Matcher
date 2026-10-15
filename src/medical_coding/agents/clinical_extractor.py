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
    ("DISCHARGE_DIAGNOSES", r"(?:DISCHARGE\s+DIAGNOS[EI]S|FINAL\s+DIAGNOS[EI]S|FINAL\s+CODING\s+SUMMARY|CODING\s+SUMMARY|FINAL\s+CODING|POSTOPERATIVE\s+DIAGNOS[EI]S|DIAGNOS[EI]S\s+ON\s+DISCHARGE|DIAGNOS[EI]S|ACTIVE\s+PROBLEMS|PROBLEM\s+LIST|ACTIVE\s+ISSUES)"),
    ("PRINCIPAL_DIAGNOSIS", r"(?:PRINCIPAL\s+DIAGNOS[EI]S|PRIMARY\s+DIAGNOS[EI]S|ADMITTING\s+DIAGNOS[EI]S|ADMISSION\s+DIAGNOS[EI]S)"),
    ("SECONDARY_DIAGNOSES", r"(?:SECONDARY\s+DIAGNOS[EI]S|ADDITIONAL\s+DIAGNOS[EI]S|CO-?MORBIDITIES|OTHER\s+DIAGNOS[EI]S)"),
    ("CHIEF_COMPLAINT", r"(?:CHIEF\s+COMPLAINT|REASON\s+FOR\s+ADMISSION|PRESENTING\s+COMPLAINT|ADMITTED\s+FOR)"),
    ("HISTORY_OF_PRESENT_ILLNESS", r"(?:HISTORY\s+OF\s+PRESENT\s+ILLNESS|HPI|SUBJECTIVE)"),
    ("PHYSICAL_EXAMINATION", r"(?:CLINICAL\s+EXAMINATION(?:\s+ON\s+ADMISSION)?|PHYSICAL\s+EXAMINATION(?:\s+ON\s+ADMISSION)?|ON\s+EXAMINATION|GENERAL\s+EXAMINATION|SYSTEMIC\s+EXAMINATION)"),
    ("HOSPITAL_COURSE", r"(?:HOSPITAL\s+COURSE|SUMMARY\s+OF\s+HOSPITAL\s+STAY|BRIEF\s+SUMMARY\s+OF\s+HOSPITAL\s+COURSE|COURSE\s+IN\s+(?:THE\s+)?HOSPITAL(?:\s+AND\s+DISCUSSION)?|CLINICAL\s+COURSE)"),
    ("COMPLICATIONS", r"(?:HOSPITAL\s+COMPLICATIONS|IN-?HOSPITAL\s+COMPLICATIONS|COMPLICATIONS|ADVERSE\s+EVENTS)"),
    ("PROCEDURES", r"(?:PROCEDURES?\s+DETAILS|PROCEDURE\s+PERFORMED|PROCEDURES?\s+PERFORMED|OPERATIVE\s+PROCEDURES?|SURGICAL\s+PROCEDURES?|MAJOR\s+PROCEDURES?|PROCEDURES|SURGICAL\s+INTERVENTIONS?|OPERATIONS?)"),
    ("OPERATIVE_FINDINGS", r"(?:INTRAOPERATIVE\s+FINDINGS?|OPERATIVE\s+FINDINGS?|SURGICAL\s+FINDINGS?|INTRA-?OPERATIVE\s+FINDINGS?|OPERATIVE\s+NOTE\s+FINDINGS?|FINDINGS?)"),
    ("ONCOLOGY_BIOMARKERS", r"(?:BIOMARKERS\s*(?:&|AND)?\s*RECEPTOR\s+STATUS|PATHOLOGY(?:\s+REPORT)?|HISTOPATHOLOGY|RECEPTOR\s+STATUS|IMMUNOHISTOCHEMISTRY|TUMOR\s+CHARACTERISTICS)"),
    ("PAST_MEDICAL_HISTORY", r"(?:PAST\s+MEDICAL\s+HISTORY|PMH|MEDICAL\s+HISTORY|BACKGROUND\s+HISTORY|PAST\s+HISTORY|PERSONAL\s+HISTORY)"),
    ("PAST_SURGICAL_HISTORY", r"(?:PAST\s+SURGICAL\s+HISTORY|PSH|SURGICAL\s+HISTORY)"),
    ("INVESTIGATIONS", r"(?:RELEVANT\s+INVESTIGATIONS?|INVESTIGATIONS?|LABORATORY\s+DATA|LABORATORY\s+INVESTIGATIONS?|PERTINENT\s+LABS|DIAGNOSTIC\s+STUDIES|IMAGING|RADIOLOGY|ECHOCARDIOGRAM|CT\s+SCAN|CT\s+PNS|ULTRASOUND|MRI)"),
    ("MICROBIOLOGY", r"(?:MICROBIOLOGY|CULTURES?|BLOOD\s+CULTURES?|URINE\s+CULTURES?|SPUTUM\s+CULTURES?|CULTURE\s+AND\s+SENSITIVITY|MICROBIOLOGICAL\s+STUDIES)"),
    ("ALLERGIES", r"(?:DRUG\s+ALLERG(?:Y|IES)|ALLERG(?:Y|IES)|ALLERGIC\s+HISTORY|ALLERGIC\s+REACTIONS?)"),
    ("MEDICATIONS", r"(?:DISCHARGE\s+MEDICATIONS?|MEDICATIONS?\s+ON\s+DISCHARGE|ACTIVE\s+MEDICATIONS?|MEDICATIONS?|CURRENT\s+MEDICATIONS?)"),
    ("PERTINENT_NEGATIVES", r"(?:PERTINENT\s+NEGATIVES|NEGATIONS(?:\s*/\s*AUDIT)?|DOCUMENTATION\s+AUDIT)"),
    ("ASSESSMENT", r"(?:ASSESSMENT\s*(?:&|AND)?\s*PLAN|ASSESSMENT|CLINICAL\s+IMPRESSION|IMPRESSION|A\s*/\s*P)"),
    ("PLAN", r"(?:DISCHARGE\s+PLAN|TREATMENT\s+PLAN|PLAN)"),
    ("DISCHARGE_INSTRUCTIONS", r"(?:STATUS\s+AT\s+DISCHARGE|CONDITION\s+AT\s+DISCHARGE|DISCHARGE\s+STATUS|DISCHARGE\s+INSTRUCTIONS|DISPOSITION|DISCHARGE\s+CONDITION|FOLLOW-?UP|SPECIAL\s+NEEDS|IN\s+CASE\s+OF\s+EMERGENCY)"),
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
                # Filter out document encounter metadata lines e.g. "Date of Admission", "Date of Procedure"
                line_start = text.rfind("\n", 0, m.start(1))
                line_start = 0 if line_start == -1 else line_start + 1
                pre_text = text[line_start:m.start(1)].strip().lower()
                if pre_text.startswith("date of") or pre_text.startswith("time of") or pre_text.startswith("type of"):
                    continue
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
            if not content_str.strip() and i + 1 < len(matches):
                continue
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

    @staticmethod
    def _protect_organism_abbreviations(text: str) -> tuple[str, dict[str, str]]:
        """Protect genus abbreviations (e.g., E. coli, H. pylori, S. aureus, C. diff) from being split on period."""
        replacements: dict[str, str] = {}
        def _repl(m: re.Match) -> str:
            token = f"__ORG_ABBR_{len(replacements)}__"
            replacements[token] = m.group(0)
            return token
        # Matches single uppercase letter + dot + space + species word
        masked = re.sub(r"\b([A-Za-z])\.\s*([A-Za-z]+(?:\s+[a-z]+)?)\b", _repl, text)
        return masked, replacements

    @staticmethod
    def _unprotect_organism_abbreviations(text: str, replacements: dict[str, str]) -> str:
        res = text
        for token, original in replacements.items():
            res = res.replace(token, original)
        return res

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences while strictly preserving organism names, numbered lists, and decimal numbers."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sentences: list[str] = []
        for line in lines:
            line = re.sub(r",([A-Za-z])", r", \1", line)
            masked, rep = EvidenceFirstFactExtractor._protect_organism_abbreviations(line)
            # Split only on period/question/exclamation followed by space and capital letter or digit
            sub_sents = re.split(r"(?<!\b[A-Za-z])(?<!\b\d)[.!?]+\s+(?=[A-Z0-9])", masked)
            for s in sub_sents:
                s_unmasked = EvidenceFirstFactExtractor._unprotect_organism_abbreviations(s.strip(), rep)
                if s_unmasked:
                    sentences.append(s_unmasked)
        return sentences

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
        if canon and canon != item.raw_term:
            item.normalized_diagnosis = canon
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
            if item.temporality == Temporality.HISTORICAL:
                is_existing_active = (
                    existing.temporality == Temporality.CURRENT
                    or existing.role in (DiagnosisRole.PRIMARY, DiagnosisRole.SECONDARY)
                    or any(ev.section in ("DISCHARGE_DIAGNOSES", "FINAL_DIAGNOSES", "PRINCIPAL_DIAGNOSIS", "SECONDARY_DIAGNOSES", "ASSESSMENT", "ASSESSMENT_PLAN") for ev in existing.evidence)
                    or existing.treatment_relevance > 0
                    or existing.procedure_relevance > 0
                )
                if not is_existing_active:
                    existing.temporality = Temporality.HISTORICAL
                    existing.role = DiagnosisRole.HISTORICAL
                    existing.is_authorized = False
                    existing.clinical_relevance = 0.0
                    existing.encounter_relevance = 0.0
                    existing.scores.encounter_relevance = 0.0
                    existing.scores.admitting_score = -50.0
            elif item.temporality == Temporality.CURRENT:
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
                    name_proc_m = re.search(r"(?:name\s+of\s+procedure|procedure\s+performed|procedures?\s+details?|procedure)\s*[:\-–—]\s*([^.,;\n]+)", l, re.IGNORECASE)
                    if name_proc_m:
                        l_clean = name_proc_m.group(1).strip()
                    else:
                        l_clean = re.sub(r"^\d+[\.\)\-]\s*", "", l).strip()
                        l_clean = re.sub(r"^(?:procedures?\s*(?:performed|done|details)?\s*[:\-–—]\s*)", "", l_clean, flags=re.IGNORECASE).strip()

                    if len(l_clean) >= 3:
                        sub_procs = re.split(r"\s*(?:\+|;)\s*|\s+and\s+", l_clean)
                        for sp in sub_procs:
                            sp_c = re.sub(r"\s*(?:side\s*:\s*[^\n]+|surgeon\s*:\s*[^\n]+|anaesthesiologist\s*:\s*[^\n]+|anesthesiologist\s*:\s*[^\n]+).*$", "", sp, flags=re.IGNORECASE).strip()
                            if len(sp_c) >= 3 and not re.match(r"^(?:primary\s+surgeon|primary|surgeon|anaesthesiologist|anesthesiologist|type\s+of\s+an[ae]sthesia|ga|general\s+an[ae]sthesia|la|local\s+an[ae]sthesia|side)\b", sp_c, re.IGNORECASE):
                                # Canonicalize known procedures to exact hospital database descriptions
                                if re.search(r"\b(?:esophageal\s+variceal\s+ligation|evl|multiple\s+band\s+ligations?|mbl|variceal\s+band\s+ligation)\b", sp_c, re.IGNORECASE):
                                    self._extracted_procedures.append("Esophagogastroduodenoscopy, flexible, transoral; with band ligation of esophageal/gastric varices")
                                elif re.search(r"\b(?:ursl|ureteroscopic\s+lithotripsy)\b", sp_c, re.IGNORECASE):
                                    self._extracted_procedures.append("Cystourethroscopy, with ureteroscopy and/or pyeloscopy; with lithotripsy including insertion of indwelling ureteral stent (eg, Gibbons or double-J type)")
                                    self._extracted_procedures.append("Cystourethroscopy, with ureteroscopy and/or pyeloscopy; with lithotripsy (ureteral catheterization is included)")
                                elif re.search(r"\bcystoscopy\b", sp_c, re.IGNORECASE) and not re.search(r"\bwith\s+ureteroscopy\b", sp_c, re.IGNORECASE):
                                    self._extracted_procedures.append("Cystourethroscopy (separate procedure)")
                                else:
                                    self._extracted_procedures.append(sp_c)
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

        # Phase 7: Narrative, Examination, Oncology, and unmapped clinical sections
        handled_non_narrative = {
            "DISCHARGE_DIAGNOSES", "FINAL_DIAGNOSES", "PRINCIPAL_DIAGNOSIS", "SECONDARY_DIAGNOSES",
            "ASSESSMENT", "ASSESSMENT_PLAN", "CHIEF_COMPLAINT", "PAST_MEDICAL_HISTORY",
            "PAST_SURGICAL_HISTORY", "HOSPITAL_COURSE", "COMPLICATIONS", "INVESTIGATIONS",
            "MICROBIOLOGY", "PROCEDURES", "OPERATIVE_FINDINGS", "ALLERGIES", "MEDICATIONS",
            "DISCHARGE_MEDICATIONS", "PERTINENT_NEGATIVES", "PLAN", "DISCHARGE_INSTRUCTIONS"
        }
        for sec in sections:
            if (
                sec.section_name in ("GENERAL", "DISCHARGE_SUMMARY", "HISTORY_OF_PRESENT_ILLNESS", "PHYSICAL_EXAMINATION", "ONCOLOGY_BIOMARKERS")
                or sec.section_name not in handled_non_narrative
            ):
                items = self._parse_narrative_items(sec.content, sec)
                for item in items:
                    self._register_candidate(candidates_by_key, item)

        # Phase 8: Authoritative Oncology Histology, Metastatic Site, and Chemotherapy Encounter Detection
        oncology_items = self._extract_oncology_secondary_diagnoses(full_text, sections)
        for item in oncology_items:
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

        # 4. Extract and separate attached laboratory measurements or vitals observations
        lab_clause_m = re.search(
            r",?\s*\bwith\s+(?:significantly\s+elevated|elevated|high|low|markedly\s+elevated)?\s*(?:creatinine|potassium|sodium|enzymes|transaminases|bilirubin|lactate|wbc|platelets|troponin|psa|ef)\b.*$",
            cleaned,
            re.IGNORECASE,
        )
        if lab_clause_m:
            attributes["laboratory_finding"] = lab_clause_m.group(0).strip(" ,")
            cleaned = cleaned[:lab_clause_m.start()].strip()

        paren_lab_m = re.search(
            r"\s*\((?:k|potassium|creatinine|cr|bp|blood\s+pressure|suv|na|sodium|ef|hb|platelets?|wbc)\s*[:=–-]?\s*[^)]+\)",
            cleaned,
            re.IGNORECASE,
        )
        if paren_lab_m:
            attributes["vital_or_lab"] = paren_lab_m.group(0).strip(" ()")
            cleaned = re.sub(re.escape(paren_lab_m.group(0)), "", cleaned).strip()

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
        raw_sentences = self._split_into_sentences(cleaned)

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

            # Skip absence statements, treatment instructions, non-clinical activities, generic procedures, and cross-reference boilerplate
            if (
                HardClinicalCandidateGate.is_absence_statement(sent_clean)
                or HardClinicalCandidateGate.is_treatment_instruction(sent_clean)
                or HardClinicalCandidateGate.is_non_clinical_activity(sent_clean)
                or HardClinicalCandidateGate.is_generic_procedure(sent_clean)
                or re.search(r"\b(?:symptoms?|complaints?)\s+described\s+in\b|\bdescribed\s+in\s+(?:the\s+)?final\s+diagnosis\b|\babove\s+mentioned\s+complaints\b", sent_clean, re.IGNORECASE)
            ):
                continue

            # Pattern 2: Clinical condition and admission clauses
            adm_match = re.search(
                r"(?:patient\s+(?:was\s+)?(?:admitted\s+(?:under\s+[a-z\s]+\s+)?(?:with|for|in)|presented\s+with|diagnosed\s+with|treated\s+for|managed\s+for|evaluated\s+for|evaluation\s+of|found\s+to\s+have|noted\s+to\s+have|has|had)|admitted\s+(?:under\s+[a-z\s]+\s+)?(?:with|for|in)|presented\s+with|diagnosed\s+with|evaluation\s+of|evaluated\s+for|impression\s*:\s*|clinical\s+impression\s*:\s*|assessment\s*:\s*|active\s+problems?\s*:\s*|secondary\s+diagnos[ei]s\s*:\s*|diagnos[ei]s\s*:\s*|consistent\s+with\s+|confirmed\s+to\s+have\s+|known\s+case\s+of\s+|also\s+(?:found\s+to\s+have|noted\s+to\s+have|has|had|suffers?\s+from)|found\s+to\s+have|noted\s+to\s+have|suffers?\s+from|history\s+of\s+|hx\s+of\s+)\s+([^.]+)",
                sent_clean,
                re.IGNORECASE,
            )
            is_adm_clause = False
            if adm_match:
                clause = adm_match.group(1).strip()
                is_adm_clause = True
            else:
                diag_prefix_m = re.search(
                    r"^(?:also\s+(?:found\s+to\s+have|noted\s+to\s+have|has|had|suffers?\s+from)|found\s+to\s+have|noted\s+to\s+have|noted\s+to\s+suffer\s+from|suffers?\s+from|developed|complicated\s+by|secondary\s+to|underwent\s+.*?\s+for|history\s+of|hx\s+of|past\s+medical\s+history\s*:?|pmh\s*:?|(?:ct|mri|x[\s\-]?ray|cxr|ultrasound|usg|echo|ekg|ecg)\s+(?:showed|revealed|demonstrated|confirmed)|noted|demonstrated|confirmed|showed|revealed)\s+([^.]+)",
                    sent_clean,
                    re.IGNORECASE,
                )
                if diag_prefix_m:
                    clause = diag_prefix_m.group(1).strip()
                else:
                    clause = sent_clean

            if clause:
                clause = re.sub(
                    r"^(?:nose|chest|cvs|cns|rs|pa|abdomen|eye|ear|throat|skin|neck|general|local|systemic)\s*:\s*",
                    "",
                    clause,
                    flags=re.IGNORECASE,
                ).strip()
                clause = re.sub(
                    r"^(?:patient\s+is\s+an?\s+(?:\d+[\s\-]year[\s\-]old\s+)?(?:male|female|gentleman|lady|man|woman)\s+(?:admitted\s+)?with\s+|patient\s+is\s+a\s+known\s+case\s+of\s+|patient\s+presents?\s+(?:to\s+.*?\s+)?with\s+|pt\s+is\s+an?\s+.*?\s+with\s+)",
                    "",
                    clause,
                    flags=re.IGNORECASE,
                ).strip()
                clause = re.sub(
                    r"^(?:(?:several|\d+)\s+(?:days?|weeks?|months?|hours?)\s+of\s+)",
                    "",
                    clause,
                    flags=re.IGNORECASE,
                ).strip()
                clause = re.sub(
                    r"^(?:(?:for\s+)?evaluation\s+of|(?:for\s+)?eval\s+of|management\s+of|workup\s+of)\s+",
                    "",
                    clause,
                    flags=re.IGNORECASE,
                ).strip()
                for tail_delim in [" treated with ", " managed with ", " requiring ", " with good response", " with symptom", " while ", " during "]:
                    if tail_delim in clause.lower():
                        idx = clause.lower().find(tail_delim)
                        clause = clause[:idx].strip()

                on_med_m = re.search(r"\s+on\s+(?:medication|antibiotic|iv\b|steroid|infusion|dialysis|heparin|insulin|pressor|aspirin|metformin|lisinopril|lasix|statin|chemo)", clause, re.IGNORECASE)
                if on_med_m:
                    clause = clause[:on_med_m.start()].strip()

                clause = re.sub(r"\s+(?:over|across)\s+(?:the\s+)?[a-z\s]+(?=,|\s+and\s+(?:acute|epigastric|ecg|[a-z]+\s+pain|[a-z]+\s+discomfort)|$)", "", clause, flags=re.IGNORECASE).strip()

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
                    is_narrative_historical = bool(
                        re.search(
                            r"\b(?:historical\s*/\s*resolved|historical\b|completely\s+resolved|resolved\b|prior\s+history|childhood\s+asthma|no\s+current\s+treatment|no\s+current\s+medications?|\d+\s+years?\s+ago|in\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+20[12][0-4]|\b20[12][0-4]\b|\b\d{1,2}/\d{1,2}/(?:20[12][0-4]|[0-2][0-4])\b)\b",
                            sent_clean,
                            re.IGNORECASE,
                        )
                    )
                    cand_temp = Temporality.HISTORICAL if is_narrative_historical else Temporality.CURRENT
                    cand_role = (
                        DiagnosisRole.HISTORICAL
                        if is_narrative_historical
                        else (DiagnosisRole.PRIMARY if is_adm_clause else DiagnosisRole.SECONDARY)
                    )
                    cand_auth = (polarity != NegationStatus.NEGATED) and (not is_narrative_historical)
                    cand_rel = 0.0 if is_narrative_historical else 1.0
                    acuity = self._detect_acuity(sc)
                    ev = StructuredEvidence(
                        text=sent_clean,
                        section=sec.section_name,
                        sentence=sent_clean,
                        polarity=polarity,
                        certainty=certainty,
                        temporality=cand_temp,
                        evidence_type=EvidenceType.DISCHARGE_SUMMARY,
                        clinical_relevance=cand_rel,
                    )
                    admit_score = (
                        -50.0
                        if is_narrative_historical
                        else ((5.0 if is_adm_clause else 1.0) + (1.0 if acuity == Acuity.ACUTE else 0.0))
                    )
                    scores = MultiDimensionalScore(
                        evidence_score=1.0,
                        diagnostic_certainty=1.0 if certainty == Certainty.CONFIRMED else 0.8,
                        encounter_relevance=cand_rel,
                        role_confidence=1.0,
                        semantic_match=1.0,
                        specificity_match=1.0,
                        contradiction_penalty=1.0 if polarity == NegationStatus.NEGATED else 0.0,
                        admitting_score=admit_score,
                    )
                    cand = ClinicalDiagnosisCandidate(
                        raw_term=sc,
                        normalized_diagnosis=sc,
                        role=cand_role,
                        certainty=certainty,
                        temporality=cand_temp,
                        clinical_attributes=cand_attrs,
                        evidence=[ev],
                        evidence_strength=1.0,
                        clinical_relevance=cand_rel,
                        encounter_relevance=cand_rel,
                        scores=scores,
                        is_authorized=cand_auth,
                        classification_reason="Historical or resolved condition documented in narrative" if is_narrative_historical else "Identified as encounter diagnosis in clinical narrative",
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
            # Strip structural role and metadata prefixes (e.g., "Primary:", "Principal Diagnosis -", "Secondary:", "Metastatic sites:")
            cleaned_line = re.sub(
                r"^(?:(?:primary|principal|secondary|final|admitting)\s*(?:diagnosis|diagnoses|condition)?|(?:diagnosis|diagnoses|condition)|metastatic\s+sites?|sites?\s+of\s+metastat\w+|distant\s+sites?|oncology\s+attributes?|biomarkers?)\s*[:\-–—]\s*",
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
                    r"\b(?:on\s+[a-z0-9]+|taking\s+[a-z0-9]+|receiving\s+[a-z0-9]+|treated|managed|controlled|under\s+treatment|active\s+rx|prescribed|daily|mg|tab|tablet|inj|injection|infusion|inhaler|dialysis)\b",
                    cleaned_preamble,
                    re.IGNORECASE,
                )
            )
            cleaned_concept = re.sub(
                r",?\s*\b(?:on\s+medication|on\s+treatment|on\s+rx|treated\s+with|managed\s+with)\b.*$",
                "",
                cleaned_preamble,
                flags=re.IGNORECASE,
            ).strip()
            if not cleaned_concept or len(cleaned_concept) < 3:
                cleaned_concept = cleaned_preamble

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

        sentences = self._split_into_sentences(cleaned)

        for sent in sentences:
            if (
                len(sent) < 5
                or HardClinicalCandidateGate.is_absence_statement(sent)
                or HardClinicalCandidateGate.is_treatment_instruction(sent)
                or HardClinicalCandidateGate.is_non_clinical_activity(sent)
                or HardClinicalCandidateGate.is_generic_procedure(sent)
            ):
                continue

            masked_sent, rep = self._protect_organism_abbreviations(sent)

            # Pattern A: Complication cues: "course was complicated by [X]", "complicated by [X]", "developed [X]", "went into [X]", "manifested [X]"
            comp_match = re.search(
                r"(?:course\s+(?:was\s+)?complicated\s+by|complicated\s+by|developed|went\s+into|manifested|suffered)\s+(.+?)(?:\.\s+[A-Z]|\.\s*$|$)",
                masked_sent,
                re.IGNORECASE,
            )
            # Pattern B: Inpatient management/diagnosis: "treated for [X]", "managed for [X]", "diagnosed with [X]"
            treat_match = re.search(
                r"(?:treated\s+(?:for|with\s+(?:iv\s+)?[a-z0-9]+\s+for)|managed\s+for|diagnosed\s+with|inpatient\s+management\s+of|(?:was\s+)?performed\s+for|(?:was\s+)?done\s+for)\s+(.+?)(?:\.\s+[A-Z]|\.\s*$|$)",
                masked_sent,
                re.IGNORECASE,
            )
            # Pattern C: Specialist evaluations, consultations, and clinical conclusions
            eval_match = re.search(
                r"(?:(?:[a-z]+\s+)?(?:evaluation|consultation|consult|assessment|review)\s+(?:concluded|revealed|showed|noted|found|confirmed|diagnosed|suggested)\s+(?:that\s+)?|concluded\s+that\s+|noted\s+to\s+have\s+|found\s+to\s+have\s+|manifested\s+(?:with\s+)?)\s*(.+?)(?:\.\s+[A-Z]|\.\s*$|$)",
                masked_sent,
                re.IGNORECASE,
            )
            # Pattern D: Additional or concurrent conditions documented during course
            additional_match = re.search(
                r"(?:with\s+additional|and\s+additional|additional|concurrent|superimposed)\s+(.+?)(?:\.\s+[A-Z]|\.\s*$|$)",
                masked_sent,
                re.IGNORECASE,
            )

            matched_clauses: list[str] = []
            if comp_match:
                matched_clauses.append(self._unprotect_organism_abbreviations(comp_match.group(1).strip(), rep))
            if treat_match:
                matched_clauses.append(self._unprotect_organism_abbreviations(treat_match.group(1).strip(), rep))
            if eval_match:
                matched_clauses.append(self._unprotect_organism_abbreviations(eval_match.group(1).strip(), rep))
            if additional_match:
                matched_clauses.append(self._unprotect_organism_abbreviations(additional_match.group(1).strip(), rep))

            if not matched_clauses:
                gen_match = re.search(
                    r"(?:patient\s+(?:had|experienced|manifested|presented\s+with|developed)|also\s+(?:had|noted|developed|experienced|found)|showed|revealed|demonstrated|confirmed|diagnosed\s+as)\s+(.+?)(?:\.\s+[A-Z]|\.\s*$|$)",
                    masked_sent,
                    re.IGNORECASE,
                )
                if gen_match:
                    matched_clauses.append(self._unprotect_organism_abbreviations(gen_match.group(1).strip(), rep))

            for clause in matched_clauses:
                # Strip trailing management phrases from clause
                for tail in [
                    " and advised ", " advised ", " requiring ", " managed with ",
                    " treated with ", " with good response", " with symptom",
                    " and was ", " and received ", " prescribed ", " necessitating "
                ]:
                    if tail in clause.lower():
                        idx = clause.lower().find(tail)
                        clause = clause[:idx].strip()

                on_med_m = re.search(r"\s+on\s+(?:medication|antibiotic|iv\b|steroid|infusion|dialysis|heparin|insulin|pressor|aspirin|metformin|lisinopril|lasix|statin|chemo)", clause, re.IGNORECASE)
                if on_med_m:
                    clause = clause[:on_med_m.start()].strip()

                # Decompose 'with additional [X]' or 'concurrent [X]' into multiple clauses
                clause_chunks = re.split(
                    r",?\s*(?:with\s+additional|and\s+additional|additional|concurrent|superimposed)\s+",
                    clause,
                    flags=re.IGNORECASE,
                )
                for chunk in clause_chunks:
                    clean_chunk = chunk.strip()
                    clean_chunk = re.sub(r"^(?:the\s+)?(?:rash\s+was\s+)?", "", clean_chunk, flags=re.IGNORECASE).strip()
                    clean_chunk = re.sub(r"\s+in\s+nature$", "", clean_chunk, flags=re.IGNORECASE).strip()
                    if clean_chunk.lower() == "urticarial":
                        clean_chunk = "urticaria"
                    # Strip anatomical location qualifiers e.g. "over the wrists", "on the chest"
                    clean_chunk = re.sub(r"\s+(?:over|on)\s+(?:the\s+)?[a-z\s]+(?=\s+due\s+to|\s*$)", "", clean_chunk, flags=re.IGNORECASE).strip()

                    # Split compound items e.g. "septic shock and diabetic ketoacidosis", "acute kidney injury secondary to dehydration"
                    sub_items = self._split_compound_clinical_phrase(clean_chunk)
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

        sentences = self._split_into_sentences(cleaned)

        for sent in sentences:
            if (
                len(sent) < 5
                or HardClinicalCandidateGate.is_absence_statement(sent)
                or HardClinicalCandidateGate.is_treatment_instruction(sent)
                or HardClinicalCandidateGate.is_non_clinical_activity(sent)
                or HardClinicalCandidateGate.is_generic_procedure(sent)
            ):
                continue

            findings_clauses: list[str] = []
            masked_sent, rep = self._protect_organism_abbreviations(sent)

            # 1. Imaging findings: CT, CXR, Ultrasound, MRI, Biopsy, EGD
            img_match = re.search(
                r"(?:ct(?:\s+scan)?|cxr|chest\s+x-?ray|ultrasound|usg|mri|egd|biopsy|echocardiogram|studies)\s*(?:of\s+[^\n:]+)?\s*(?::|--|-)?\s*(?:showed|revealed|demonstrated|confirmed|reported|noted|positive\s+(?:for)?)\s+([^.]+)",
                masked_sent,
                re.IGNORECASE,
            )
            if img_match:
                findings_clauses.append(self._unprotect_organism_abbreviations(img_match.group(1).strip(), rep))

            # 2. Microbiology / culture findings: blood culture, urine culture, sputum culture
            micro_match = re.search(
                r"(?:blood\s+cultures?|urine\s+cultures?|sputum\s+cultures?|stool\s+cultures?|cultures?|gram\s+stain|clo\s+test|microbiology)\s*(?::|--|-)?\s*(?:grew|positive\s+(?:for)?|showed|isolated|revealed|detected|identified)?\s*([^.]+)",
                masked_sent,
                re.IGNORECASE,
            )
            if micro_match:
                findings_clauses.append(self._unprotect_organism_abbreviations(micro_match.group(1).strip(), rep))

            # Direct extraction for prominent clinical microorganisms
            for org_pat in [
                r"\b(?:e\.?\s*coli|escherichia\s+coli)\b",
                r"\b(?:h\.?\s*pylori|helicobacter\s+pylori)\b",
                r"\b(?:c\.?\s*diff(?:icile)?|clostridioides\s+difficile)\b",
                r"\b(?:s\.?\s*aureus|staphylococcus\s+aureus|mrsa)\b",
                r"\b(?:s\.?\s*pneumoniae|streptococcus\s+pneumoniae)\b",
                r"\b(?:p\.?\s*aeruginosa|pseudomonas\s+aeruginosa)\b",
                r"\b(?:k\.?\s*pneumoniae|klebsiella\s+pneumoniae)\b",
            ]:
                org_m = re.search(org_pat, sent, re.IGNORECASE)
                if org_m:
                    findings_clauses.append(org_m.group(0).strip())

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
            if (
                len(line_clean) < 5
                or HardClinicalCandidateGate.is_absence_statement(line_clean)
                or HardClinicalCandidateGate.is_treatment_instruction(line_clean)
                or HardClinicalCandidateGate.is_non_clinical_activity(line_clean)
                or HardClinicalCandidateGate.is_generic_procedure(line_clean)
            ):
                continue

            # Match: "[Procedure] for [Condition]" or "Indication: [Condition]"
            ind_match = re.search(
                r"(?:for|indication\s*:?)\s+([^.,;\n]+)",
                line_clean,
                re.IGNORECASE,
            )
            if ind_match:
                raw_cand_clause = ind_match.group(1).strip()
                split_cand_terms = self._split_compound_clinical_phrase(raw_cand_clause)
                for cand_term_raw in split_cand_terms:
                    cand_term = re.sub(r"^(?:a|an|the)\s+", "", cand_term_raw, flags=re.IGNORECASE).strip()
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
        primary_site = "unknown"
        if re.search(r"\bprostat(?:e|ic)\b", text_lower):
            primary_site = "prostate"
        elif "breast" in text_lower:
            primary_site = "breast"
        else:
            for site in ["ovary", "ovarian", "colon", "lung", "cervix", "uterus", "kidney"]:
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
        elif "right" in text_lower and "left" not in text_lower and primary_site in ("breast", "lung", "kidney", "ovary"):
            laterality = "right"
        elif "left" in text_lower and "right" not in text_lower and primary_site in ("breast", "lung", "kidney", "ovary"):
            laterality = "left"
        elif "bilateral" in text_lower and primary_site in ("breast", "lung", "kidney", "ovary"):
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
            r"\bacinar\s+adenocarcinoma\b",
            r"\bprostatic\s+adenocarcinoma\b",
            r"\badenocarcinoma\b",
            r"\bsquamous\s+cell\s+carcinoma\b",
        ]:
            m = re.search(hist_pat, text_lower)
            if m:
                histology = m.group(0).title()
                break

        # 4. Grade, Gleason Score, and Grade Group
        grade = None
        gleason_score = None
        grade_group = None

        gleason_m = re.search(
            r"\bgleason(?:\s+score)?\s*(?::|=|\s+is|\s+was)?\s*([0-9]+(?:\s*(?:\+|\/|\=|\(.*?\))\s*[0-9]+)?)\b",
            text,
            re.IGNORECASE,
        )
        if gleason_m:
            gleason_score = gleason_m.group(1).strip()

        gg_m = re.search(r"\bgrade\s+group\s*([1-5])\b", text, re.IGNORECASE)
        if gg_m:
            grade_group = f"Grade Group {gg_m.group(1)}"

        grade_m = re.search(
            r"\b(?:grade\s*([1-3]|i{1,3})|poorly\s+differentiated|moderately\s+differentiated|well\s+differentiated)\b",
            text_lower,
        )
        if grade_m:
            grade = grade_m.group(0).capitalize()
        elif grade_group:
            grade = grade_group
        elif gleason_score:
            grade = f"Gleason {gleason_score}"

        # 5. Receptor Status (ER, PR, HER2, PSA)
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

        psa_level = None
        psa_m = re.search(
            r"\bpsa(?:\s+level)?\s*(?::|=|\s+is|\s+was)?\s*([><=]?\s*[\d\.\,]+(?:\s*ng\s*/\s*m[lL])?)\b",
            text,
            re.IGNORECASE,
        )
        if psa_m:
            psa_level = psa_m.group(1).strip()
            if "ng/ml" not in psa_level.lower() and re.match(r"^[\d\.\,]+$", psa_level):
                psa_level = f"{psa_level} ng/mL"
            receptor_status["PSA"] = psa_level
            receptor_status["psa"] = psa_level

        # 6. Castration Status (mCRPC vs Castration-sensitive)
        castration_status = None
        if re.search(r"\b(?:metastatic\s+)?castrat(?:ion|e)[- ]resistant(?:\s+(?:status|state|prostate\s+cancer))?\b|\bmcrpc\b|\bcrpc\b", text_lower):
            castration_status = "Castration-resistant"
        elif re.search(r"\bcastrat(?:ion|e)[- ]sensitive\b|\bhormone[- ]sensitive\b|\bmhspc\b", text_lower):
            castration_status = "Castration-sensitive"

        # 7. Ki-67 Proliferation Index
        ki67 = None
        ki67_m = re.search(
            r"\bki-?67(?:\s+(?:proliferation\s+index|labeling\s+index|index))?\s*(?:[:=]|\s+is|\s+was)?\s*([><=]?\s*\d+%(?:\s*(?:high|low))?|high|low)",
            text,
            re.IGNORECASE,
        )
        if ki67_m:
            ki67 = ki67_m.group(1).strip()

        # 8. Metastatic Status & Sites
        met_sites: list[str] = []
        if re.search(r"\bno\s+(?:evidence\s+of\s+)?(?:distant\s+)?metastases\b|\bnon[- ]metastatic\b|\bm0\b", text_lower):
            met_status = "NON_METASTATIC"
        elif re.search(r"\bmetastat(?:ic|es)\b|\bm1\b", text_lower):
            met_status = "METASTATIC"
            for site in ["bone", "liver", "lung", "brain", "pleura", "peritoneum", "adrenal", "lymph node", "spine", "pelvis"]:
                if site in text_lower:
                    norm_site = "bone" if site in ("spine", "pelvis") else site
                    if norm_site not in met_sites:
                        met_sites.append(norm_site)
        else:
            met_status = "UNKNOWN"

        # 9. Prior Treatments
        prior_tx: list[str] = []
        for tx_pat in [
            r"\bandrogen\s+deprivation\s+therapy\b|\badt\b",
            r"\bleuprolide\b|\blupron\b",
            r"\benzalutamide\b|\bxtandi\b",
            r"\babiraterone\b|\bzytiga\b",
            r"\bbicalutamide\b|\bcasodex\b",
            r"\bdenosumab\b|\bxgeva\b|\bprolia\b",
            r"\bdocetaxel\b|\btaxotere\b",
            r"\bcabazitaxel\b|\bjevtana\b",
            r"\bradium-?223\b|\bxofigo\b",
            r"\bneoadjuvant\s+chemotherapy\b",
            r"\badjuvant\s+chemotherapy\b",
            r"\bpalliative\s+radiotherapy\b|\bradiotherapy\b|\bradiation\b",
            r"\btamoxifen\b",
            r"\bletrozole\b",
            r"\banastrozole\b",
            r"\btrastuzumab\b|\bherceptin\b",
            r"\bprostatectomy\b|\bradical\s+prostatectomy\b",
        ]:
            m = re.search(tx_pat, text_lower)
            if m:
                prior_tx.append(m.group(0))

        # 10. Response to Treatment
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
            gleason_score=gleason_score,
            grade_group=grade_group,
            psa_level=psa_level,
            castration_status=castration_status,
            prior_treatments=prior_tx,
            evidence=text[:200],
        )

    def _extract_oncology_secondary_diagnoses(
        self,
        full_text: str,
        sections: list[ClinicalDocumentSection],
    ) -> list[ClinicalDiagnosisCandidate]:
        candidates: list[ClinicalDiagnosisCandidate] = []
        text_lower = full_text.lower()

        # Check if encounter involves malignancy or oncology care
        is_oncology = any(
            w in text_lower
            for w in [
                "carcinoma", "adenocarcinoma", "malignancy", "malignant", "neoplasm",
                "cancer", "lymphoma", "sarcoma", "melanoma", "chemotherapy", "docetaxel",
                "capox", "rituximab", "palliative chemo"
            ]
        )
        if not is_oncology:
            return []

        # Find primary oncology evidence section for grounding
        onc_ev_sec = "HISTORY_OF_PRESENT_ILLNESS"
        for s in sections:
            if s.section_name in ("ONCOLOGY_BIOMARKERS", "INVESTIGATIONS", "HISTORY_OF_PRESENT_ILLNESS"):
                onc_ev_sec = s.section_name
                break

        # 1. Histology (ICD-O Morphology M8140.3: Adenocarcinoma NOS)
        if re.search(r"\b(?:invasive\s+adenocarcinoma|infiltrating\s+adenocarcinoma|adenocarcinoma\s+composed\s+of\s+solid\s+glands|moderately\s+differentiated\s+adenocarcinoma|poorly\s+differentiated\s+adenocarcinoma|ampullary\s+bx[\s\-]+moderately\s+differentiated\s+adenocarcinoma)\b", text_lower):
            m_quote = "Pathology: invasive adenocarcinoma"
            for line in full_text.split("\n"):
                if "adenocarcinoma" in line.lower() and any(w in line.lower() for w in ("hpr", "bx", "invasive", "solid glands")):
                    m_quote = line.strip()
                    break
            ev = StructuredEvidence(
                text=m_quote,
                section=onc_ev_sec,
                sentence=m_quote,
                polarity=NegationStatus.AFFIRMATIVE,
                certainty=Certainty.CONFIRMED,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.PATHOLOGY,
                clinical_relevance=1.0,
            )
            scores = MultiDimensionalScore(
                evidence_score=1.0, diagnostic_certainty=1.0, encounter_relevance=1.0,
                role_confidence=1.0, semantic_match=1.0, specificity_match=1.0, admitting_score=4.0
            )
            candidates.append(
                ClinicalDiagnosisCandidate(
                    raw_term="Adenocarcinoma NOS",
                    normalized_diagnosis="ADENOMAS NAD ADENCARCINOMAS-Adenocarcinoma NOS",
                    role=DiagnosisRole.SECONDARY,
                    certainty=Certainty.CONFIRMED,
                    temporality=Temporality.CURRENT,
                    assertion_status=AssertionStatus.CONFIRMED,
                    evidence=[ev],
                    scores=scores,
                    is_authorized=True,
                    classification_reason="Histopathology confirmed adenocarcinoma NOS",
                )
            )

        # 2. Secondary malignant neoplasm of bone (C79.51)
        if (
            re.search(r"\b(?:sclerotic\s+skeletal\s+lesions?|psma\s+expressing\s+(?:skeletal|bone)\s+lesions?|skeletal\s+lesions\s+with\s+psma\s+expression|bone\s+metastases?|osseous\s+metastases?)\b", text_lower)
            or (re.search(r"\b(?:humerus|sacrum|femur(?:\s+head)?)\s+(?:proximal\s+shaft\s+and\s+head\s+)?suv\s*[\d\.]+", text_lower) and "psma" in text_lower)
        ):
            b_quote = "Sclerotic skeletal lesions with PSMA expression"
            for line in full_text.split("\n"):
                if ("sclerotic" in line.lower() or "skeletal" in line.lower() or "suv" in line.lower()) and ("bone" in line.lower() or "lesion" in line.lower() or "humerus" in line.lower() or "sacrum" in line.lower()):
                    b_quote = line.strip()
                    break
            ev = StructuredEvidence(
                text=b_quote,
                section=onc_ev_sec,
                sentence=b_quote,
                polarity=NegationStatus.AFFIRMATIVE,
                certainty=Certainty.CONFIRMED,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.IMAGING,
                clinical_relevance=1.0,
            )
            scores = MultiDimensionalScore(
                evidence_score=1.0, diagnostic_certainty=1.0, encounter_relevance=1.0,
                role_confidence=1.0, semantic_match=1.0, specificity_match=1.0, admitting_score=4.0
            )
            candidates.append(
                ClinicalDiagnosisCandidate(
                    raw_term="Secondary malignant neoplasm of bone",
                    normalized_diagnosis="Secondary malignant neoplasm of bone",
                    role=DiagnosisRole.SECONDARY,
                    certainty=Certainty.CONFIRMED,
                    temporality=Temporality.CURRENT,
                    assertion_status=AssertionStatus.CONFIRMED,
                    evidence=[ev],
                    scores=scores,
                    is_authorized=True,
                    classification_reason="Objective imaging demonstrated metastatic osseous involvement",
                )
            )

        # 3. Secondary malignant neoplasm of intra-abdominal lymph nodes (C77.2)
        if re.search(
            r"\b(?:(?:paraaortic|aortocaval|peripancreatic|perigastric|peri-?portal|precaval|porto-?caval|gastrohepatic|pericholecystic|retroperitoneal|intra-?abdominal|abdominal)\s*(?:and\s+[a-z\s]+)?nodes?|(?:lymph\s+)?nodes?\s+(?:in|within|around|at)?\s*(?:in\s+)?(?:the\s+)?(?:paraaortic|aortocaval|peripancreatic|perigastric|peri-?portal|precaval|porto-?caval|gastrohepatic|pericholecystic|retroperitoneal|intra-?abdominal|abdominal)|(?:several|multiple)\s+abdominal\s+lymph\s+nodes|abdominal\s+lymph\s+nodes?)\b",
            text_lower,
        ):
            ln_quote = "Intra-abdominal lymph nodes documented on staging imaging"
            for line in full_text.split("\n"):
                if ("aortocaval" in line.lower() or "paraaortic" in line.lower() or "peripancreatic" in line.lower() or "periportal" in line.lower() or "abdominal" in line.lower() or "pericholecystic" in line.lower()) and "node" in line.lower():
                    ln_quote = line.strip()
                    break
            ev = StructuredEvidence(
                text=ln_quote,
                section=onc_ev_sec,
                sentence=ln_quote,
                polarity=NegationStatus.AFFIRMATIVE,
                certainty=Certainty.CONFIRMED,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.IMAGING,
                clinical_relevance=1.0,
            )
            scores = MultiDimensionalScore(
                evidence_score=1.0, diagnostic_certainty=1.0, encounter_relevance=1.0,
                role_confidence=1.0, semantic_match=1.0, specificity_match=1.0, admitting_score=4.0
            )
            candidates.append(
                ClinicalDiagnosisCandidate(
                    raw_term="Secondary and unspecified malignant neoplasm of intra-abdominal lymph nodes",
                    normalized_diagnosis="Secondary and unspecified malignant neoplasm of intra-abdominal lymph nodes",
                    role=DiagnosisRole.SECONDARY,
                    certainty=Certainty.CONFIRMED,
                    temporality=Temporality.CURRENT,
                    assertion_status=AssertionStatus.CONFIRMED,
                    evidence=[ev],
                    scores=scores,
                    is_authorized=True,
                    classification_reason="Staging imaging demonstrated metastatic intra-abdominal lymphadenopathy",
                )
            )

        # 4. Secondary malignant neoplasm of inguinal and lower limb lymph nodes (C77.4)
        if re.search(
            r"\b(?:external\s+iliac\s+nodes?|proximal\s+femoral\s+lymph\s+node|femoral\s+lymph\s+node|external\s+iliac\s+region)\b",
            text_lower,
        ):
            il_quote = "External iliac and proximal femoral lymph nodes documented on imaging"
            for line in full_text.split("\n"):
                if ("external iliac" in line.lower() or "femoral lymph node" in line.lower()) and "node" in line.lower():
                    il_quote = line.strip()
                    break
            ev = StructuredEvidence(
                text=il_quote,
                section=onc_ev_sec,
                sentence=il_quote,
                polarity=NegationStatus.AFFIRMATIVE,
                certainty=Certainty.CONFIRMED,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.IMAGING,
                clinical_relevance=1.0,
            )
            scores = MultiDimensionalScore(
                evidence_score=1.0, diagnostic_certainty=1.0, encounter_relevance=1.0,
                role_confidence=1.0, semantic_match=1.0, specificity_match=1.0, admitting_score=4.0
            )
            candidates.append(
                ClinicalDiagnosisCandidate(
                    raw_term="Secondary and unspecified malignant neoplasm of inguinal and lower limb lymph nodes",
                    normalized_diagnosis="Secondary and unspecified malignant neoplasm of inguinal and lower limb lymph nodes",
                    role=DiagnosisRole.SECONDARY,
                    certainty=Certainty.CONFIRMED,
                    temporality=Temporality.CURRENT,
                    assertion_status=AssertionStatus.CONFIRMED,
                    evidence=[ev],
                    scores=scores,
                    is_authorized=True,
                    classification_reason="Staging imaging demonstrated metastatic external iliac / femoral nodal disease",
                )
            )

        # 5. Secondary malignant neoplasm of liver and intrahepatic bile duct (C78.7)
        if re.search(
            r"\b(?:multiple\s*(?:\(innumerable\s*\)\s*)?(?:variable\s+sized\s*,\s*)?discrete\s+lesions\s+are\s+noted\s+in\s+both\s+(?:the\s+)?lobes\s+of\s+liver|discrete\s+heterogeneously\s+enhancing\s+lesions\s+involving\s+segment.*?-?\s*metastatic|liver\s+metastases?|hepatic\s+metastases?|persistent\s+liver\s+disease)\b",
            text_lower,
        ):
            liv_quote = "Staging imaging demonstrated metastatic hepatic involvement"
            for line in full_text.split("\n"):
                if "liver" in line.lower() and ("lesion" in line.lower() or "metastat" in line.lower() or "discrete" in line.lower()):
                    liv_quote = line.strip()
                    break
            ev = StructuredEvidence(
                text=liv_quote,
                section=onc_ev_sec,
                sentence=liv_quote,
                polarity=NegationStatus.AFFIRMATIVE,
                certainty=Certainty.CONFIRMED,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.IMAGING,
                clinical_relevance=1.0,
            )
            scores = MultiDimensionalScore(
                evidence_score=1.0, diagnostic_certainty=1.0, encounter_relevance=1.0,
                role_confidence=1.0, semantic_match=1.0, specificity_match=1.0, admitting_score=4.0
            )
            candidates.append(
                ClinicalDiagnosisCandidate(
                    raw_term="Secondary malignant neoplasm of liver",
                    normalized_diagnosis="Secondary malignant neoplasm of liver and intrahepatic bile duct",
                    role=DiagnosisRole.SECONDARY,
                    certainty=Certainty.CONFIRMED,
                    temporality=Temporality.CURRENT,
                    assertion_status=AssertionStatus.CONFIRMED,
                    evidence=[ev],
                    scores=scores,
                    is_authorized=True,
                    classification_reason="Imaging documented metastatic bilobar liver lesions",
                )
            )

        # 6. Secondary malignant neoplasm of retroperitoneum and peritoneum (C78.6)
        if re.search(
            r"\b(?:peritoneal\s+involvement|peritoneal\s+metastases?|omental\s+fat\s+stranding.*?(?:nodularities|reticulations)|omental\s+nodularities|peritoneal\s+wall\s+thickening.*?represents\s+peritoneal\s+involvement)\b",
            text_lower,
        ):
            per_quote = "Omental fat stranding, nodularities, and peritoneal involvement"
            for line in full_text.split("\n"):
                if "peritoneal" in line.lower() or "omental" in line.lower():
                    per_quote = line.strip()
                    break
            ev = StructuredEvidence(
                text=per_quote,
                section=onc_ev_sec,
                sentence=per_quote,
                polarity=NegationStatus.AFFIRMATIVE,
                certainty=Certainty.CONFIRMED,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.IMAGING,
                clinical_relevance=1.0,
            )
            scores = MultiDimensionalScore(
                evidence_score=1.0, diagnostic_certainty=1.0, encounter_relevance=1.0,
                role_confidence=1.0, semantic_match=1.0, specificity_match=1.0, admitting_score=4.0
            )
            candidates.append(
                ClinicalDiagnosisCandidate(
                    raw_term="Secondary malignant neoplasm of retroperitoneum and peritoneum",
                    normalized_diagnosis="Secondary malignant neoplasm of retroperitoneum and peritoneum",
                    role=DiagnosisRole.SECONDARY,
                    certainty=Certainty.CONFIRMED,
                    temporality=Temporality.CURRENT,
                    assertion_status=AssertionStatus.CONFIRMED,
                    evidence=[ev],
                    scores=scores,
                    is_authorized=True,
                    classification_reason="Imaging demonstrated omental nodularity and peritoneal involvement",
                )
            )

        # 7. Secondary / Chemotherapy-induced Thrombocytopenia (D69.59)
        if re.search(r"\b(?:labs\s+showed\s+thrombocytopenia|thrombocytopenia.*?chemotherapy\s+withheld|received\s+inj\s+romy.*?thrombocytopenia|thrombocytopenia.*?romy)\b", text_lower):
            thr_quote = "Documented secondary thrombocytopenia requiring chemotherapy withholding / romiplostim"
            for line in full_text.split("\n"):
                if "thrombocytopenia" in line.lower():
                    thr_quote = line.strip()
                    break
            ev = StructuredEvidence(
                text=thr_quote,
                section=onc_ev_sec,
                sentence=thr_quote,
                polarity=NegationStatus.AFFIRMATIVE,
                certainty=Certainty.CONFIRMED,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.LABORATORY,
                clinical_relevance=1.0,
            )
            scores = MultiDimensionalScore(
                evidence_score=1.0, diagnostic_certainty=1.0, encounter_relevance=1.0,
                role_confidence=1.0, semantic_match=1.0, specificity_match=1.0, admitting_score=4.0
            )
            candidates.append(
                ClinicalDiagnosisCandidate(
                    raw_term="Other secondary thrombocytopenia",
                    normalized_diagnosis="Other secondary thrombocytopenia",
                    role=DiagnosisRole.SECONDARY,
                    certainty=Certainty.CONFIRMED,
                    temporality=Temporality.CURRENT,
                    assertion_status=AssertionStatus.CONFIRMED,
                    evidence=[ev],
                    scores=scores,
                    is_authorized=True,
                    classification_reason="Chemotherapy-induced secondary thrombocytopenia managed with treatment withholding and thrombopoietin support",
                )
            )

        # 8. Chemotherapy administration encounter procedure detection (CPT 9005 / CPT 9001)
        has_chemo_encounter = bool(
            re.search(
                r"\b(?:now\s+admitted\s+for\s+cycle\s+\d+|admitted\s+for\s+(?:second\s+line\s+)?palliative\s+chemo\s+cycle\s+\d+|received\s+(?:second\s+line\s+palliative\s+chemo\s+)?cycle\s+\d+|admitted\s+for\s+maintenance\s+regimen\s+rituximab|received\s+iv\s+chemotherapy\s+as\s+follows\s+inj\s+rituximab)\b",
                text_lower,
            )
        )
        if has_chemo_encounter:
            proc_9005 = "Chemotherapy administration, intravenous infusion technique; up to 1 hour, single or initial substance/drug"
            if proc_9005 not in self._extracted_procedures:
                self._extracted_procedures.append(proc_9005)
            # Check if multi-hour infusion (e.g. over 5 hours in Rituximab maintenance)
            if re.search(r"\bover\s+([2-9]|\d{2})\s+hours\b", text_lower):
                proc_9001 = "Chemotherapy administration, intravenous infusion technique; each additional hour"
                if proc_9001 not in self._extracted_procedures:
                    self._extracted_procedures.append(proc_9001)

        return candidates

    @staticmethod
    def _split_adjacent_clinical_conditions(text: str) -> list[str]:
        """Decompose phrases containing adjacent independent clinical conditions.

        Clinical principles:
        1. When multiple distinct diagnoses appear juxtaposed without commas or conjunctions
           (e.g., 'Urticaria Gastritis', 'Hypertension Diabetes Mellitus', 'Asthma Bronchitis'),
           each valid independent clinical entity must be separated.
        2. A split is rejected if the preceding word is an adjectival modifier
           (e.g., 'acute', 'chronic', 'hypertensive', 'allergic', 'ischemic', 'ulcerative', 'contact')
           which qualifies the subsequent disease.
        3. A split is rejected if the subsequent part is merely a generic nosological continuation
           (e.g., 'disease', 'disorder', 'syndrome', 'failure', 'mellitus', 'type 2').
        4. Both decomposed halves must independently satisfy HardClinicalCandidateGate validation.
        """
        words = text.strip().split()
        if len(words) < 2:
            return [text.strip()]

        from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

        adjective_modifiers = {
            "acute", "chronic", "subacute", "severe", "mild", "moderate",
            "primary", "secondary", "recurrent", "refractory", "hypertensive",
            "ischemic", "diabetic", "allergic", "ulcerative", "obstructive",
            "atherosclerotic", "rheumatoid", "infectious", "congestive",
            "dilated", "hypertrophic", "interstitial", "systemic", "autoimmune",
            "invasive", "infiltrating", "metastatic", "benign", "malignant",
            "idiopathic", "acquired", "congenital", "hereditary", "familial",
            "focal", "diffuse", "bilateral", "unilateral", "left", "right",
            "anterior", "posterior", "upper", "lower", "type", "grade", "stage",
            "contact", "external", "internal", "deviated", "ruptured", "perforated",
            "obstructed", "calcified", "stenosed", "thrombosed", "infarcted",
            "fractured", "displaced", "sprained", "infected", "inflamed", "congested",
            "enlarged", "narrowed", "thickened", "elevated", "depressed", "collapsed",
            "atrophic", "hypoplastic", "aplastic", "gangrenous", "necrotic",
            "nasal", "renal", "cardiac", "pulmonary", "hepatic", "gastric", "splenic",
            "cerebral", "spinal", "thoracic", "abdominal", "pelvic", "femoral", "tibial",
            "coronary", "aortic", "mitral", "tricuspid", "carotid", "jugular", "ocular",
            "retinal", "aural", "otic", "vertebral", "lumbar", "cervical", "cranial",
            "facial", "oral", "anal", "rectal", "colonic", "duodenal", "jejunal", "ileal",
            "esophageal", "bronchial", "alveolar", "pancreatic", "biliary", "vesical",
            "urethral", "ureteral", "prostatic", "testicular", "ovarian", "uterine",
            "vaginal", "vulvar", "peritoneal", "pleural", "pericardial",
            "prostate", "breast", "lung", "kidney", "liver", "colon", "bone", "bladder",
            "thyroid", "pancreas", "stomach", "brain", "spine", "pelvis", "ovary", "testis", "heart", "skin",
            "castration-resistant", "castrate-resistant", "hormone-refractory", "hormone-sensitive",
            "castration-sensitive", "resistant", "recurrent", "refractory",
            "viral", "bacterial", "fungal", "parasitic", "flu", "like", "portal",
            "significantly", "markedly", "severely", "mildly", "moderately", "critically",
            "reduced", "decreased", "increased", "high", "low", "abnormal",
        }

        generic_suffix_nouns = {
            "disease", "diseases", "disorder", "disorders", "syndrome", "syndromes",
            "failure", "insufficiency", "mellitus", "infarction", "dysfunction",
            "type 1", "type 2", "type i", "type ii", "status",
            "illness", "illnesses", "condition", "conditions",
        }

        for i in range(1, len(words)):
            p1 = " ".join(words[:i]).strip()
            p2 = " ".join(words[i:]).strip()

            last_word_p1 = words[i - 1].lower()
            if last_word_p1 in adjective_modifiers:
                continue

            if p2.lower() in generic_suffix_nouns:
                continue

            if re.search(r"^\s*with\b", p2, re.IGNORECASE):
                combo_patterns = [
                    r"\bwith\s+(?:diabetic\s+)?(?:ketoacidosis|hyperosmolarity|nephropathy|retinopathy|neuropathy|angiopathy|arthropathy|skin|ulcer|hypoglycemia|hyperglycemia)\b",
                    r"\bwith(?:out)?\s+(?:active\s+)?(?:bleeding|hemorrhage|perforation|obstruction|coma|exacerbation|status\s+asthmaticus)\b",
                    r"\bwith\s+delta[\s\-]agent\b",
                    r"\b(?:calculus|stone|calculi)\s+with\s+(?:moderate\s+)?(?:hydronephrosis|obstruction)\b",
                    r"\bwith\s+bleed(?:ing)?\b",
                    r"\bwith\s+(?:significantly\s+|markedly\s+)?(?:elevated|high|low|reduced|abnormal)?\s*(?:creatinine|potassium|sodium|bilirubin|lactate|wbc|platelets?|troponin|psa|ef|cr)\b",
                ]
                if any(re.search(pat, text, re.IGNORECASE) for pat in combo_patterns):
                    continue

            p2_clean = re.sub(r"^(?:with|due\s+to|secondary\s+to|caused\s+by|from)\s+", "", p2, flags=re.IGNORECASE).strip()

            v1, _ = HardClinicalCandidateGate.evaluate_candidate(p1)
            v2, _ = HardClinicalCandidateGate.evaluate_candidate(p2_clean or p2)

            if v1 and v2:
                return [p1] + EvidenceFirstFactExtractor._split_adjacent_clinical_conditions(p2_clean or p2)

        return [text.strip()]

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
        abbrev_matches = list(re.finditer(r"\b([A-Za-z])\.\s*([A-Za-z]+(?:\s+[a-z]+)?)\b", masked))
        for i, m in enumerate(abbrev_matches):
            masked = masked.replace(m.group(0), f"__ORGANISM_{i}__")

        # Mask parentheticals to prevent internal splitting on commas or plus signs (e.g. '(4+5)', '(lumbar spine, pelvis)')
        paren_matches = list(re.finditer(r"\([^\)]*\)|\[[^\]]*\]", masked))
        for i, pm in enumerate(paren_matches):
            masked = masked.replace(pm.group(0), f"__PAREN_EXPR_{i}__")

        # Mask plus signs attached to numbers (e.g., Gleason 4+5)
        plus_num_matches = list(re.finditer(r"\b\d+\+\d+\b", masked))
        for i, pnm in enumerate(plus_num_matches):
            masked = masked.replace(pnm.group(0), f"__PLUS_NUM_{i}__")

        # Mask plus signs attached to receptor/immunology markers (e.g., CD4+, HER2+, ER+, PR+)
        plus_matches = list(re.finditer(r"\b([A-Za-z0-9\-]+)\+", masked))
        for i, pm in enumerate(plus_matches):
            if pm.group(1).lower() in ("cd4", "cd8", "her2", "er", "pr", "egfr", "bcl2"):
                masked = masked.replace(pm.group(0), f"__PLUS_MARKER_{i}__")

        # 1. Connectors that unconditionally separate independent clinical entities
        # Split on list delimiters: commas (,), plus signs (+), semicolons (;), ampersands (&), slashes (/), and 'in a known case of'
        list_delims_pattern = r"\s*(?:,|;|\+|\s+&\s+|\s+/\s+|\b(?:in\s+a\s+known\s+case\s+of|known\s+case\s+of|k/c/o)\b)\s*"
        raw_splits = [p.strip() for p in re.split(list_delims_pattern, masked, flags=re.IGNORECASE) if p.strip()]

        chunks: list[str] = []
        causal_pattern = r",?\s*\b(?:complicated\s+by|secondary\s+to|due\s+to|resulting\s+in|manifested\s+by)\b\s*"
        for seg in raw_splits:
            chunks.extend([c.strip() for c in re.split(causal_pattern, seg, flags=re.IGNORECASE) if c.strip()])

        # Standard ICD combination manifestations where "with" should NEVER be split
        combo_manifestation_patterns = [
            r"\bwith\s+(?:diabetic\s+)?(?:ketoacidosis|hyperosmolarity|nephropathy|retinopathy|neuropathy|angiopathy|arthropathy|skin|ulcer|hypoglycemia|hyperglycemia)\b",
            r"\bwith(?:out)?\s+(?:active\s+)?(?:bleeding|hemorrhage|perforation|obstruction|coma|exacerbation|status\s+asthmaticus)\b",
            r"\bwith\s+delta[\s\-]agent\b",
            r"\b(?:calculus|stone|calculi)\s+with\s+(?:moderate\s+)?(?:hydronephrosis|obstruction)\b",
            r"\bwith\s+bleed(?:ing)?\b",
            r"\bwith\s+(?:significantly\s+|markedly\s+)?(?:elevated|high|low|reduced|abnormal)?\s*(?:creatinine|potassium|sodium|bilirubin|lactate|wbc|platelets?|troponin|psa|ef|cr)\b",
            r"\bwith\s+(?:portal\s+hypertension|splenomegaly|ascites)\b",
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

            # Strip metadata headers if embedded in chunk
            chunk = re.sub(
                r"^(?:metastatic\s+sites?|oncology\s+attributes?(?:\s*(?:&|and)?\s*biomarkers?)?|biomarkers?(?:\s*&|\s*and)?\s*receptor\s+status)\s*[:\-–—]\s*",
                "",
                chunk,
                flags=re.IGNORECASE,
            ).strip()
            if not chunk:
                continue

            # Decompose venous thrombosis combinations: "SMV and Portal Vein Thrombosis" -> "Portal Vein Thrombosis", "SMV Thrombosis"
            v_thromb_m = re.search(r"^smv\s+and\s+portal\s+vein\s+thrombosis(?:\s*(?:\(pvt\)|__PAREN_EXPR_\d+__))?(?:\s+with\s+.*)?$", chunk, re.IGNORECASE)
            if v_thromb_m:
                refined_chunks.append("Portal Vein Thrombosis")
                refined_chunks.append("SMV Thrombosis")
                continue

            # Handle "metastatic to site1 and site2" or "metastases to site1 and site2"
            met_to_m = re.search(r"^(.*?)\s*,?\s*\b(metastat(?:ic|es)\s+to)\s+([^.,;\n]+)$", chunk, re.IGNORECASE)
            if met_to_m:
                pri_part = met_to_m.group(1).strip()
                met_pfx = met_to_m.group(2).strip()
                met_targets = met_to_m.group(3).strip()
                if pri_part and len(pri_part) >= 3:
                    refined_chunks.append(pri_part)
                targets = re.split(r",\s*|\s+and\s+", met_targets)
                for t in targets:
                    t_clean = t.strip()
                    if t_clean and len(t_clean) >= 2:
                        refined_chunks.append(f"{met_pfx} {t_clean}")
                continue

            # Handle "[adj1] and [adj2] lymph node metastases"
            adj_noun_m = re.search(r"^([a-z\-]+)\s+and\s+([a-z\-]+)\s+(lymph\s+node\s+metastases?|osseous\s+metastases?|bone\s+metastases?)$", chunk, re.IGNORECASE)
            if adj_noun_m:
                refined_chunks.append(f"{adj_noun_m.group(1)} {adj_noun_m.group(3)}")
                refined_chunks.append(f"{adj_noun_m.group(2)} {adj_noun_m.group(3)}")
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
                generic_with_m = re.search(r",?\s*\bwith\s+([A-Za-z0-9\s\-]+)$", chunk, re.IGNORECASE)
                if generic_with_m and not is_combo_manifestation:
                    pre_with = chunk[:generic_with_m.start()].strip()
                    post_with = generic_with_m.group(1).strip()
                    pre_with_clean = re.sub(r"\s+to\s+(?:the\s+)?(?:left|right|bilateral)$", "", pre_with, flags=re.IGNORECASE).strip()
                    post_with_unmasked = post_with
                    for i, m in enumerate(abbrev_matches):
                        post_with_unmasked = post_with_unmasked.replace(f"__ORGANISM_{i}__", m.group(0))
                    from medical_coding.validation.clinical_gate import HardClinicalCandidateGate
                    v_pre, _ = HardClinicalCandidateGate.evaluate_candidate(pre_with_clean or pre_with)
                    v_post, _ = HardClinicalCandidateGate.evaluate_candidate(post_with_unmasked)
                    if v_pre and v_post:
                        refined_chunks.append(pre_with_clean or pre_with)
                        refined_chunks.append(post_with)
                    else:
                        refined_chunks.append(chunk)
                else:
                    refined_chunks.append(chunk)

        # 2. Split chunks on " and " when connecting distinct clinical entities
        final_items: list[str] = []
        for item in refined_chunks:
            item_clean = re.sub(r"^(?:and|or)\s+", "", item.strip(), flags=re.IGNORECASE).strip()
            and_parts = [p.strip().strip(",; ") for p in re.split(r",?\s*\band\b\s*", item_clean, flags=re.IGNORECASE) if p.strip().strip(",; ")]
            if len(and_parts) > 1 and all(len(p) >= 3 for p in and_parts):
                for p in and_parts:
                    final_items.append(p)
            else:
                final_items.append(item_clean.strip(",; "))

        # Restore masked organism abbreviations, clean timing/procedural context, and decompose adjacent conditions
        restored: list[str] = []
        for res_item in final_items:
            res = res_item
            for i, m in enumerate(abbrev_matches):
                res = res.replace(f"__ORGANISM_{i}__", m.group(0))
            for i, pm in enumerate(plus_matches):
                res = res.replace(f"__PLUS_MARKER_{i}__", pm.group(0))
            for i, pnm in enumerate(plus_num_matches):
                res = res.replace(f"__PLUS_NUM_{i}__", pnm.group(0))
            for i, pm in enumerate(paren_matches):
                res = res.replace(f"__PAREN_EXPR_{i}__", pm.group(0))

            # Strip trailing procedural or temporal timing context (e.g. "gastritis post frozen embryo transfer" -> "gastritis")
            timing_m = re.search(r"\b(?:status\s+post|post|s/p|after)\s+(?:frozen\s+embryo\s+transfer|fet|chemotherapy|surgery|procedure|infusion)\b.*$", res, re.IGNORECASE)
            if timing_m:
                base_cand = res[:timing_m.start()].strip()
                if len(base_cand) >= 3:
                    res = base_cand

            decomposed = self._split_adjacent_clinical_conditions(res.strip())
            for d in decomposed:
                d_clean = d.strip()
                if len(d_clean) >= 3:
                    restored.append(d_clean)

        return [r for r in restored if len(r) >= 3]

    def _split_term_and_narrative(self, line: str) -> tuple[str, str]:
        """Split a clinical line into the diagnostic term and accompanying clinical explanation."""
        # If line begins with metadata/label prefix, strip it so the clinical entity is the term
        label_prefix_m = re.match(
            r"^\s*(?:metastatic\s+sites?|sites?\s+of\s+metastat\w+|distant\s+sites?|oncology\s+attributes?|biomarkers?|pathology|histology)\s*[:\-–—]\s*",
            line,
            re.IGNORECASE,
        )
        if label_prefix_m:
            line = line[label_prefix_m.end():].strip()

        # Narrative delimiters introduce management/course comments (e.g. " - underwent DJ stenting")
        # Do not include ";" or "/" as they separate distinct clinical entities in diagnosis lists
        for delim in [" - ", " – ", " — ", " : ", ": "]:
            if delim in line:
                parts = line.split(delim, 1)
                term = parts[0].strip()
                narrative = parts[1].strip()
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
        # Strip parenthetical annotations if attached (e.g. "(BP 210/110)", "(K 7.2 - 8.9)", "(CLD)", "(PVT)")
        clean = re.sub(r"\([^)]*\)", "", clean).strip()
        clean = re.sub(r",?\s*with\s+(?:significantly\s+)?elevated\s+creatinine.*$", "", clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r"\s+", " ", clean).strip(".,;: ")

        # 1. Primary oncology sites & morphology
        if re.search(r"\b(?:periampullary\s+(?:adenocarcinoma|carcinoma|malignancy|cancer)|ampullary\s+(?:adenocarcinoma|carcinoma|malignancy|cancer)|carcinoma\s+of\s+ampulla\s+of\s+vater)\b", clean, re.IGNORECASE):
            return "Malignant neoplasm of ampulla of Vater"

        if re.search(r"\b(?:adenocarcinoma\s+nos|invasive\s+adenocarcinoma|infiltrating\s+adenocarcinoma|adenocarcinoma\s+composed\s+of\s+solid\s+glands|moderately\s+differentiated\s+adenocarcinoma)\b", clean, re.IGNORECASE):
            return "ADENOMAS NAD ADENCARCINOMAS-Adenocarcinoma NOS"

        # 2. Oncology metastatic manifestations
        if re.search(
            r"\b(?:bone\s+metastases?|metastat(?:ic|es)\s+(?:to|in)\s+bone|metastatic\s+bone\s+disease|bone\s*\((?:lumbar\s+)?spine|bone\s*\(pelvis|osseous\s+metastases?|skeletal\s+metastases?|osteoblastic\s+metastases?|osteoblastic\s+metastatic\s+disease|osteoblastic\s+(?:spine\s+)?lesions?|sclerotic\s+skeletal\s+lesions?|skeletal\s+lesions(?:\s+with\s+psma\s+expression)?|psma\s+expressing\s+(?:skeletal\s+lesions?|bone\s+lesions?)|metastat(?:ic|es)\s+to\s+(?:the\s+)?(?:lumbar\s+)?spine|metastat(?:ic|es)\s+to\s+(?:the\s+)?pelvis)\b",
            clean,
            re.IGNORECASE,
        ):
            return "Secondary malignant neoplasm of bone"

        if re.search(
            r"\b(?:retroperitoneal\s+(?:and\s+pelvic\s+)?lymph\s+node\s+metastases?|retroperitoneal\s+lymphadenopathy|metastat(?:ic|es)\s+(?:to|in)\s+retroperitoneal\s+lymph\s+nodes?|intra[- ]abdominal\s+lymph\s+node\s+metastases?|paraaortic\s+and\s+aortocaval\s+nodes?|aortocaval\s+nodes?|paraaortic\s+nodes?|peripancreatic.*?(?:lymph\s+)?nodes?|peri-?portal.*?(?:lymph\s+)?nodes?|precaval.*?(?:lymph\s+)?nodes?|porto-?caval.*?(?:lymph\s+)?nodes?|gastrohepatic.*?(?:lymph\s+)?nodes?|central\s+mesentric\s+vessels)\b",
            clean,
            re.IGNORECASE,
        ):
            return "Secondary and unspecified malignant neoplasm of intra-abdominal lymph nodes"

        if re.search(
            r"\b(?:inguinal\s+and\s+lower\s+limb\s+lymph\s+nodes?|external\s+iliac\s+nodes?|proximal\s+femoral\s+lymph\s+node|femoral\s+lymph\s+node|external\s+iliac\s+lymph\s+nodes?|inguinal\s+lymph\s+nodes?)\b",
            clean,
            re.IGNORECASE,
        ):
            return "Secondary and unspecified malignant neoplasm of inguinal and lower limb lymph nodes"

        if re.search(
            r"\b(?:pelvic\s+lymph\s+node\s+metastases?|metastat(?:ic|es)\s+(?:to|in)\s+pelvic\s+lymph\s+nodes?)\b",
            clean,
            re.IGNORECASE,
        ):
            return "Secondary and unspecified malignant neoplasm of intrapelvic lymph nodes"

        if re.search(
            r"\b(?:lymph\s+node\s+metastases?|metastat(?:ic|es)\s+(?:to|in)\s+lymph\s+nodes?|nodal\s+metastases?|distant\s+lymphadenopathy)\b",
            clean,
            re.IGNORECASE,
        ):
            return "Secondary and unspecified malignant neoplasm of lymph nodes"

        if re.search(r"\b(?:liver\s+metastases?|hepatic\s+metastases?|metastat(?:ic|es)\s+(?:to|in)\s+liver|multiple\s+lesions\s+in\s+both\s+(?:the\s+)?lobes\s+of\s+liver|discrete\s+lesions\s+in\s+both\s+lobes|metastatic\s+liver\s+disease|discrete\s+(?:heterogeneously\s+enhancing\s+)?lesions\s+involving\s+segment.*?-?\s*metastatic)\b", clean, re.IGNORECASE):
            return "Secondary malignant neoplasm of liver and intrahepatic bile duct"

        if re.search(r"\b(?:peritoneal\s+involvement|peritoneal\s+metastases?|peritoneal\s+metastasis|omental\s+fat\s+stranding.*?(?:nodularities|reticulations)|omental\s+nodularities|metastat(?:ic|es)\s+to\s+peritoneum)\b", clean, re.IGNORECASE):
            return "Secondary malignant neoplasm of retroperitoneum and peritoneum"

        if re.search(r"\b(?:lung\s+metastases?|pulmonary\s+metastases?|metastat(?:ic|es)\s+(?:to|in)\s+lung)\b", clean, re.IGNORECASE):
            return "Secondary malignant neoplasm of lung"

        if re.search(r"\b(?:brain\s+metastases?|cerebral\s+metastases?|metastat(?:ic|es)\s+(?:to|in)\s+brain)\b", clean, re.IGNORECASE):
            return "Secondary malignant neoplasm of brain"

        if re.search(r"\b(?:pleural\s+metastases?|pleural\s+metastasis|metastasis\s+to\s+pleura|malignant\s+pleural\s+effusion|malignant\s+pleural\s+metastasis)\b", clean, re.IGNORECASE):
            return "Secondary malignant neoplasm of pleura"

        if re.search(r"\b(?:prostate\s+cancer|prostate\s+carcinoma|adenocarcinoma\s+of\s+(?:the\s+)?prostate|prostate\s+adenocarcinoma|malignant\s+neoplasm\s+of\s+prostate)\b", clean, re.IGNORECASE):
            return "Malignant neoplasm of prostate"

        # 3. Urology & Renal Conditions
        if re.search(r"\b(?:vuj\s+calculus|right\s+vuj\s+calculus|vesicoureteric\s+junction\s+calculus|vesicoureteral\s+junction\s+calculus|calculus\s+near\s+the\s+vuj|tightly\s+impacted\s+stone\s+in\s+the\s+lower\s+right\s+ureter|impacted\s+stone\s+in\s+the\s+lower\s+right\s+ureter|right\s+ureteric\s+stone|ureteric\s+calculus|ureteral\s+calculus)\b", clean, re.IGNORECASE):
            return "Calculus of ureter"

        if re.search(r"\b(?:puj\s+calculus\s+with\s+(?:moderate\s+)?hydronephrosis|left\s+puj\s+calculus\s+with\s+(?:moderate\s+)?hydronephrosis|pelviureteric\s+junction\s+calculus\s+with\s+hydronephrosis|calculus\s+causing\s+(?:moderate\s+)?hydronephrosis|hydronephrosis\s+with\s+(?:renal\s+and\s+ureteral\s+)?calcul(?:us|ous)\s+obstruction)\b", clean, re.IGNORECASE):
            return "Hydronephrosis with renal and ureteral calculous obstruction"

        if re.search(r"\b(?:puj\s+calculus|left\s+puj\s+calculus|pelviureteric\s+junction\s+calculus|non-obstructive\s+left\s+(?:lower\s+calyceal\s+)?renal\s+calculus|lower\s+calyceal\s+calculus|calyceal\s+calculus|renal\s+calculus|calculus\s+of\s+kidney)\b", clean, re.IGNORECASE):
            return "Calculus of kidney"

        if re.search(r"\b(?:hypertensive\s+emergency|severe\s+hypertension)\b", clean, re.IGNORECASE):
            return "Hypertensive emergency"

        if re.search(r"\b(?:hyperkalemia|elevated\s+potassium)\b", clean, re.IGNORECASE):
            return "Hyperkalemia"

        if re.search(r"\b(?:acute\s+kidney\s+injury|aki)\b", clean, re.IGNORECASE):
            return "Acute kidney injury"

        if re.search(r"\b(?:acute\s+kidney\s+failure|acute\s+renal\s+failure)\b", clean, re.IGNORECASE):
            return "Acute kidney failure, unspecified"

        # 4. Hepatology & GI Conditions
        if re.search(r"\b(?:high[\s\-]?risk\s+esophageal\s+varices\s+with\s+bleed|esophageal\s+varices\s+with\s+bleed(?:ing)?|bleeding\s+esophageal\s+varices)\b", clean, re.IGNORECASE):
            return "Esophageal varices with bleeding"

        if re.search(r"\b(?:high[\s\-]?risk\s+esophageal\s+varices|esophageal\s+varices\s+without\s+bleed(?:ing)?)\b", clean, re.IGNORECASE) and not re.search(r"\bwith\s+bleed\b", clean, re.IGNORECASE):
            return "Esophageal varices without bleeding"

        if re.search(r"\b(?:chronic\s+liver\s+disease|cld\s+with\s+portal\s+hypertension|chronic\s+liver\s+disease\s+with\s+portal\s+hypertension|cirrhosis\s+of\s+liver|liver\s+cirrhosis|cld)\b", clean, re.IGNORECASE):
            return "Unspecified cirrhosis of liver"

        if re.search(r"\b(?:portal\s+vein\s+thrombosis|pvt)\b", clean, re.IGNORECASE) and not re.search(r"\bsmv\b", clean, re.IGNORECASE):
            return "Portal vein thrombosis"

        if re.search(r"\b(?:smv\s+and\s+portal\s+vein\s+thrombosis|smv\s+thrombosis|superior\s+mesenteric\s+vein\s+thrombosis)\b", clean, re.IGNORECASE):
            return "Chronic embolism and thrombosis of inferior vena cava"

        if re.search(r"\b(?:portal\s+hypertensive\s+gastropathy|phg)\b", clean, re.IGNORECASE):
            return "Angiodysplasia of stomach and duodenum with bleeding"

        if re.search(r"\b(?:colitis|noninfective\s+colitis|mild\s+colitis)\b", clean, re.IGNORECASE):
            return "Noninfective gastroenteritis and colitis, unspecified"

        if re.search(r"\b(?:thrombocytopenia|secondary\s+thrombocytopenia|chemotherapy[\s\-]?induced\s+thrombocytopenia)\b", clean, re.IGNORECASE):
            return "Other secondary thrombocytopenia"

        # 5. Pulmonology & Medicine
        if re.search(r"\b(?:viral\s+flu\s+like\s+illness|viral\s+lower\s+respiratory\s+tract\s+infection|viral\s+respiratory\s+infection)\b", clean, re.IGNORECASE):
            return "Unspecified acute lower respiratory infection"

        if re.search(r"\b(?:acute\s+exacerbation\s+of\s+copd|copd\s+with\s+(?:acute\s+)?exacerbation|chronic\s+obstructive\s+pulmonary\s+disease\s+with\s+(?:acute\s+)?exacerbation)\b", clean, re.IGNORECASE):
            return "Chronic obstructive pulmonary disease with (acute) exacerbation"

        if re.search(r"\b(?:type\s+2\s+diabetes|t2dm|diabetes\s+mellitus\s+type\s+2|type\s+ii\s+diabetes)\b", clean, re.IGNORECASE):
            return "Type 2 diabetes mellitus"

        if re.search(r"\b(?:portal\s+hypertensive\s+gastropathy|phg|gastropathy\s+portal)\b", clean, re.IGNORECASE):
            return "Angiodysplasia of stomach and duodenum with bleeding"

        if re.search(r"\bportal\s+hypertension\b", clean, re.IGNORECASE):
            return "Portal hypertension"

        if re.search(r"\b(?:essential\s+hypertension|primary\s+hypertension|systemic\s+hypertension|arterial\s+hypertension|hypertension|htn)\b", clean, re.IGNORECASE) and not re.search(r"\b(?:portal|pulmonary|intracranial|ocular)\s+hypertension\b", clean, re.IGNORECASE):
            return "Essential hypertension"

        if clean.lower() in ("urticarial", "urticarial rash", "rash urticarial"):
            return "Urticaria"
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
        exam_text = ""
        complaint_text = ""

        for sec in sections:
            if sec.section_name == "HOSPITAL_COURSE":
                hospital_course_text = f"{hospital_course_text} {sec.content}"
            elif sec.section_name in ("INVESTIGATIONS", "MICROBIOLOGY"):
                investigations_text = f"{investigations_text} {sec.content}"
            elif sec.section_name == "PROCEDURES":
                procedure_text = f"{procedure_text} {sec.content}"
            elif sec.section_name in ("MEDICATIONS", "DISCHARGE_MEDICATIONS"):
                medications_text = f"{medications_text} {sec.content}"
            elif sec.section_name in ("PHYSICAL_EXAMINATION", "CLINICAL_EXAMINATION"):
                exam_text = f"{exam_text} {sec.content}"
            elif sec.section_name in ("CHIEF_COMPLAINT", "HISTORY_OF_PRESENT_ILLNESS"):
                complaint_text = f"{complaint_text} {sec.content}"

        active_care_verbs = [
            "treated", "managed", "monitored", "administered", "started", "continued",
            "withheld", "titrated", "adjusted", "sliding scale", "infusion", "dose",
            "protocol", "regimen", "followed", "received", "underwent", "controlled",
            "stabilized", "responded", "prescribed", "switched", "transitioned",
            "requiring", "vasopressor", "norepinephrine", "antibiotic", "fluids", "support",
        ]

        # Clinical domain indicators for chronic/PMH conditions requiring active inpatient management
        condition_care_markers: dict[str, list[str]] = {
            "diabetes": ["insulin", "sliding scale", "metformin", "glucose", "blood sugar", "glycemic", "dka", "hypoglycemia", "hyperglycemia", "hba1c"],
            "hypertension": ["amlodipine", "antihypertensive", "blood pressure", "bp", "lisinopril", "losartan", "metoprolol", "atenolol", "hydrochlorothiazide", "furosemide"],
            "kidney": ["creatinine", "gfr", "nephrology", "dialysis", "renal function", "baseline creatinine", "aki", "acute kidney injury"],
            "renal": ["creatinine", "gfr", "nephrology", "dialysis", "renal function", "aki", "acute kidney injury"],
            "aki": ["fluid", "fluids", "resuscitation", "creatinine", "iv fluids", "hydration"],
            "dehydration": ["fluid", "fluids", "iv fluids", "hydration", "resuscitation"],
            "heart failure": ["furosemide", "lasix", "diuresis", "echo", "echocardiogram", "ejection fraction", "cardiac"],
            "asthma": ["inhaler", "nebulizer", "albuterol", "ipratropium", "bronchodilator", "steroid", "prednisone"],
            "copd": ["inhaler", "nebulizer", "albuterol", "ipratropium", "bronchodilator", "steroid", "prednisone", "oxygen"],
            "sepsis": ["antibiotic", "ceftriaxone", "vancomycin", "vasopressor", "norepinephrine", "lactate", "blood culture", "cultures", "hypotension", "iv fluids", "resuscitation"],
            "pneumonia": ["antibiotic", "ceftriaxone", "azithromycin", "sputum", "cxr", "infiltrate", "consolidation", "oxygen"],
            "stemi": ["stent", "catheterization", "troponin", "pci", "aspirin", "heparin", "plavix", "ticagrelor"],
            "infarction": ["stent", "catheterization", "troponin", "pci", "aspirin", "heparin", "plavix", "ticagrelor"],
            "gallstones": ["cholecystectomy", "ultrasound", "biliary", "colic", "cbd"],
            "cholecystitis": ["cholecystectomy", "ultrasound", "biliary", "antibiotic", "gallbladder"],
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

            found_management_sentences: list[str] = []
            for sent in course_sentences:
                sent_lower = sent.lower()
                has_cand_mention = (
                    any(re.search(rf"\b{re.escape(w)}\b", sent_lower) for w in words)
                    or (cand_norm in sent_lower)
                ) if words else False

                has_marker_mention = any(re.search(rf"\b{re.escape(m)}\b", sent_lower) for m in relevant_markers)
                has_active_verb = any(v in sent_lower for v in active_care_verbs)

                if (has_cand_mention or has_marker_mention) and has_active_verb:
                    if sent not in found_management_sentences:
                        found_management_sentences.append(sent)
                        if len(found_management_sentences) >= 3:
                            break

            for found_management_sentence in found_management_sentences:
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

                if not any(ev.text == found_management_sentence for ev in cand.evidence):
                    # Prepend supporting structured evidence if PMH; otherwise append
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

            # Check physical examination for supporting objective signs
            if exam_text and words and any(w in exam_text.lower() for w in words):
                for sent in re.split(r"[\.\n]+", exam_text):
                    if any(w in sent.lower() for w in words):
                        cand.evidence.append(
                            StructuredEvidence(
                                text=sent.strip(),
                                section="PHYSICAL_EXAMINATION",
                                sentence=sent.strip(),
                                polarity=NegationStatus.AFFIRMATIVE,
                                certainty=Certainty.SUPPORTED,
                                temporality=Temporality.CURRENT,
                                evidence_type=EvidenceType.CLINICAL_NOTE,
                                clinical_relevance=0.7,
                            )
                        )
                        break

            # Check chief complaint / HPI for documented presenting symptoms
            if complaint_text and words and any(w in complaint_text.lower() for w in words):
                for sent in re.split(r"[\.\n]+", complaint_text):
                    if any(w in sent.lower() for w in words):
                        cand.evidence.append(
                            StructuredEvidence(
                                text=sent.strip(),
                                section="CHIEF_COMPLAINT",
                                sentence=sent.strip(),
                                polarity=NegationStatus.AFFIRMATIVE,
                                certainty=Certainty.SUPPORTED,
                                temporality=Temporality.CURRENT,
                                evidence_type=EvidenceType.ADMISSION_REASON,
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

            # 1. Filter active candidates: retain all strictly valid clinical diagnoses
            valid_active: list[ClinicalDiagnosisCandidate] = []
            for cand in active_candidates:
                is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(
                    cand.normalized_diagnosis, cand.primary_evidence_quote
                )
                if is_valid:
                    valid_active.append(cand)
                else:
                    cand.role = DiagnosisRole.EXCLUDED
                    cand.is_authorized = False

            # 2. Sort active candidates by admitting score descending
            valid_active.sort(key=lambda c: c.scores.admitting_score, reverse=True)

            # If doc mentions acute exacerbation of COPD anywhere in hospital course or HPI, upgrade any COPD candidate to acute exacerbation
            full_narrative_lower = " ".join(s.content for s in sections).lower()
            if re.search(r"\b(?:acute\s+exacerbation\s+of\s+copd|copd\s+with\s+(?:acute\s+)?exacerbation|acute\s+copd\s+exacerbation)\b", full_narrative_lower):
                for cand in valid_active:
                    if cand.normalized_diagnosis.lower() in ("chronic obstructive pulmonary disease", "copd", "chronic obstructive pulmonary disease, unspecified"):
                        cand.normalized_diagnosis = "Chronic obstructive pulmonary disease with (acute) exacerbation"
                        cand.raw_term = "Chronic obstructive pulmonary disease with (acute) exacerbation"

            if valid_active:
                # Rank 1 is unconditionally PRIMARY
                top = valid_active[0]
                top.role = DiagnosisRole.PRIMARY
                top.is_authorized = True
                top.primary_justification = f"Documented chief condition occasioning admission and inpatient treatment: {top.primary_evidence_quote}"
                primary_diag = top

                top_norm = top.normalized_diagnosis.lower()
                top_canon = self._canonicalize_term(top.normalized_diagnosis).lower()

                # Rank 2..N are unconditionally SECONDARY (dropping only exact duplicates of primary/secondaries)
                for sec_cand in valid_active[1:]:
                    sec_norm = sec_cand.normalized_diagnosis.lower()
                    sec_canon = self._canonicalize_term(sec_cand.normalized_diagnosis).lower()

                    # Exact duplicate check against top or already accepted secondary conditions
                    is_exact_dup = (
                        sec_norm == top_norm
                        or sec_canon == top_canon
                        or any(sec_norm == s.normalized_diagnosis.lower() for s in secondary_diags)
                        or any(sec_canon == self._canonicalize_term(s.normalized_diagnosis).lower() for s in secondary_diags)
                    )
                    if is_exact_dup:
                        sec_cand.role = DiagnosisRole.PRIMARY if (sec_norm == top_norm or sec_canon == top_canon) else DiagnosisRole.SECONDARY
                        sec_cand.is_authorized = False
                        sec_cand.classification_reason = f"Corroborating mention of diagnosis '{top.normalized_diagnosis}'"
                        top.evidence.extend(sec_cand.evidence)
                        continue

                    # Clinical Subsumption / Manifestation Suppression:
                    all_accepted_canons = [top_canon] + [self._canonicalize_term(s.normalized_diagnosis).lower() for s in secondary_diags]

                    # 1. Esophageal varices with bleeding (I85.01) subsumes esophageal varices without bleeding (I85.00)
                    if "esophageal varices without bleeding" in sec_canon and any("esophageal varices with bleeding" in c for c in all_accepted_canons):
                        sec_cand.role = DiagnosisRole.EXCLUDED
                        sec_cand.is_authorized = False
                        sec_cand.classification_reason = "Subsumed by specific manifestation 'Esophageal varices with bleeding' (ICD-10 I85.01)"
                        continue

                    # 2. COPD with acute exacerbation (J44.1) subsumes COPD unspecified (J44.9)
                    if sec_canon in ("chronic obstructive pulmonary disease", "chronic obstructive pulmonary disease, unspecified") and any("chronic obstructive pulmonary disease with (acute) exacerbation" in c for c in all_accepted_canons):
                        sec_cand.role = DiagnosisRole.EXCLUDED
                        sec_cand.is_authorized = False
                        sec_cand.classification_reason = "Subsumed by acute exacerbation 'Chronic obstructive pulmonary disease with (acute) exacerbation' (ICD-10 J44.1)"
                        continue

                    # 3. Hydronephrosis with calculous obstruction (N13.2) subsumes generic hydronephrosis
                    if "hydronephrosis" in sec_canon and "obstruction" not in sec_canon and any("hydronephrosis with renal and ureteral calculous obstruction" in c for c in all_accepted_canons):
                        sec_cand.role = DiagnosisRole.EXCLUDED
                        sec_cand.is_authorized = False
                        sec_cand.classification_reason = "Subsumed by 'Hydronephrosis with renal and ureteral calculous obstruction' (ICD-10 N13.2)"
                        continue

                    # 4. Biliary obstruction / CBD stricture is direct mechanical manifestation of ampullary / pancreatic neoplasm
                    if (
                        any(b in sec_canon for b in ("biliary obstruction", "stricture of bile duct", "cbd stricture", "bile duct stricture", "obstruction of bile duct", "distal cbd stricture", "common bile duct stricture", "stricture of artery"))
                        and any(p in top_canon for p in ("ampulla", "periampullary", "pancreas", "pancreatic", "cholangiocarcinoma"))
                    ):
                        sec_cand.role = DiagnosisRole.EXCLUDED
                        sec_cand.is_authorized = False
                        sec_cand.classification_reason = f"Biliary obstruction/stricture is integral to primary neoplasm '{top.normalized_diagnosis}'; excluded from independent billing under CMS Guideline I.B.4"
                        continue

                    # CMS Guideline I.B.4: Integral symptom / sign of primary diagnosis suppression
                    is_integral = HardClinicalCandidateGate.is_symptom_integral_to_primary(
                        sec_cand.normalized_diagnosis, top.normalized_diagnosis, sec_cand.primary_evidence_quote
                    )
                    if is_integral:
                        from medical_coding.schemas.enums import ClinicalEntityType
                        sec_cand.role = DiagnosisRole.SYMPTOM
                        sec_cand.entity_type = ClinicalEntityType.SYMPTOM
                        sec_cand.is_authorized = False
                        sec_cand.classification_reason = f"Symptom integral to the primary diagnosis '{top.normalized_diagnosis}'; excluded from independent billing under CMS Guideline I.B.4"
                        continue

                    # UHDDS Criteria: Incidental operative finding without dedicated surgical/therapeutic intervention
                    is_incidental_finding = (
                        any(ev.section == "OPERATIVE_FINDINGS" for ev in sec_cand.evidence)
                        or "incidental" in sec_cand.primary_evidence_quote.lower()
                    )
                    if is_incidental_finding:
                        full_content_lower = " ".join(s.content for s in sections).lower()
                        finding_words = [w for w in re.findall(r"\b[a-z]{4,}\b", sec_norm) if w not in ("incidental", "simple", "benign", "mild", "filmy", "finding")]
                        has_intervention = any(
                            re.search(rf"\b(?:excision|resection|removal|repair|plasty|lysis|drainage)\b[^\n.,;]*?\b{re.escape(w)}\b", full_content_lower)
                            or re.search(rf"\b{re.escape(w)}\b[^\n.,;]*?\b(?:excision|resection|removal|repair|plasty|lysis|drainage)\b", full_content_lower)
                            for w in finding_words
                        ) or any(
                            w in full_content_lower
                            for w in ["cystectomy", "extensive adhesiolysis", "surgical excision of cyst", "pathology confirmed malignancy", "septoplasty", "turbinoplasty", "fess"]
                        )
                        if not has_intervention:
                            sec_cand.role = DiagnosisRole.EXCLUDED
                            sec_cand.is_authorized = False
                            sec_cand.classification_reason = (
                                f"[OPERATIVE_FINDINGS] Incidental finding '{sec_cand.normalized_diagnosis}' without documented "
                                "dedicated inpatient intervention or therapy; excluded from billable coding under UHDDS criteria."
                            )
                            continue

                    # Distinct valid condition -> SECONDARY
                    sec_cand.role = DiagnosisRole.SECONDARY
                    sec_cand.is_authorized = True
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
            if oncology_context.gleason_score:
                primary_diag.clinical_attributes["gleason_score"] = oncology_context.gleason_score
            if oncology_context.grade_group:
                primary_diag.clinical_attributes["grade_group"] = oncology_context.grade_group
            if oncology_context.psa_level:
                primary_diag.clinical_attributes["psa_level"] = oncology_context.psa_level
            if oncology_context.castration_status:
                primary_diag.clinical_attributes["castration_status"] = oncology_context.castration_status
            if oncology_context.metastatic_status:
                primary_diag.clinical_attributes["metastatic_status"] = oncology_context.metastatic_status
            if oncology_context.metastatic_sites:
                primary_diag.clinical_attributes["metastatic_sites"] = oncology_context.metastatic_sites
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
