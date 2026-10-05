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
from uuid import uuid4

from medical_coding.schemas.clinical import (
    EvidenceLocation,
    ExtractedClinicalCondition,
)
from medical_coding.schemas.enums import (
    Acuity,
    Certainty,
    ClinicalEntityType,
    ConditionStatus,
    DiagnosisRole,
    EvidenceType,
    Laterality,
    NegationStatus,
    Temporality,
)
from medical_coding.schemas.evidence import (
    ClinicalDiagnosisCandidate,
    ClinicalDiagnosisState,
    MultiDimensionalScore,
    StructuredEvidence,
)
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)

# Standard section header regexes
SECTION_PATTERNS: list[tuple[str, str]] = [
    ("DISCHARGE_SUMMARY", r"(?:HOSPITAL\s+DISCHARGE\s+SUMMARY|DISCHARGE\s+SUMMARY|CLINICAL\s+SUMMARY)"),
    ("DISCHARGE_DIAGNOSES", r"(?:DISCHARGE\s+DIAGNOS[EI]S|FINAL\s+DIAGNOS[EI]S|POSTOPERATIVE\s+DIAGNOS[EI]S|DIAGNOS[EI]S\s+ON\s+DISCHARGE|DIAGNOSIS)"),
    ("PRINCIPAL_DIAGNOSIS", r"(?:PRINCIPAL\s+DIAGNOS[EI]S|PRIMARY\s+DIAGNOS[EI]S|ADMITTING\s+DIAGNOS[EI]S|ADMISSION\s+DIAGNOS[EI]S)"),
    ("SECONDARY_DIAGNOSES", r"(?:SECONDARY\s+DIAGNOS[EI]S|ADDITIONAL\s+DIAGNOS[EI]S|CO-?MORBIDITIES|OTHER\s+DIAGNOS[EI]S)"),
    ("CHIEF_COMPLAINT", r"(?:CHIEF\s+COMPLAINT|REASON\s+FOR\s+ADMISSION|PRESENTING\s+COMPLAINT|ADMITTED\s+FOR)"),
    ("HOSPITAL_COURSE", r"(?:HOSPITAL\s+COURSE|SUMMARY\s+OF\s+HOSPITAL\s+STAY|BRIEF\s+SUMMARY\s+OF\s+HOSPITAL\s+COURSE|COURSE\s+IN\s+HOSPITAL|CLINICAL\s+COURSE)"),
    ("PROCEDURES", r"(?:PROCEDURES\s+PERFORMED|OPERATIVE\s+PROCEDURES|SURGICAL\s+PROCEDURES|MAJOR\s+PROCEDURES|PROCEDURES)"),
    ("PAST_MEDICAL_HISTORY", r"(?:PAST\s+MEDICAL\s+HISTORY|PMH|MEDICAL\s+HISTORY|BACKGROUND\s+HISTORY|PAST\s+HISTORY)"),
    ("PAST_SURGICAL_HISTORY", r"(?:PAST\s+SURGICAL\s+HISTORY|PSH|SURGICAL\s+HISTORY)"),
    ("INVESTIGATIONS", r"(?:RELEVANT\s+INVESTIGATIONS|INVESTIGATIONS|LABORATORY\s+DATA|PERTINENT\s+LABS|DIAGNOSTIC\s+STUDIES|IMAGING|RADIOLOGY|ECHOCARDIOGRAM|CT\s+SCAN|ULTRASOUND|MRI)"),
    ("MEDICATIONS", r"(?:DISCHARGE\s+MEDICATIONS|MEDICATIONS\s+ON\s+DISCHARGE|ACTIVE\s+MEDICATIONS|MEDICATIONS|CURRENT\s+MEDICATIONS)"),
    ("ASSESSMENT_PLAN", r"(?:ASSESSMENT\s+AND\s+PLAN|ASSESSMENT|PLAN|IMPRESSION)"),
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
    r"^\s*\?",
]

ACUITY_ACUTE_CUES = [r"\bacute\b", r"\bdecompensated\b", r"\bexacerbation\b", r"\bemergent\b", r"\bsevere\s+acute\b"]
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
            regex = re.compile(rf"(?:^|\n)\s*({pat_str})\s*(?::|--|\n)", re.IGNORECASE)
            for m in regex.finditer(text):
                matches.append({
                    "section_name": sec_name,
                    "header_text": m.group(1).strip(),
                    "start_pos": m.start(),
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

    def extract_clinical_state(
        self,
        text: str,
        document_id: str,
    ) -> ClinicalDiagnosisState:
        """Extract all clinical diagnosis candidates, score them, and determine roles."""
        sections = SectionSegmenter.segment(text)
        candidates = self._extract_candidates_from_sections(sections, text)

        # Classify candidates using strict evidence hierarchy
        state = self._classify_and_structure_state(candidates, sections, document_id)
        return state

    def _extract_candidates_from_sections(
        self,
        sections: list[ClinicalDocumentSection],
        full_text: str,
    ) -> list[ClinicalDiagnosisCandidate]:
        candidates: list[ClinicalDiagnosisCandidate] = []
        seen_canonical: set[str] = set()

        # Phase 1: High-yield diagnosis sections
        for sec in sections:
            sec_name = sec.section_name
            if sec_name in ("DISCHARGE_DIAGNOSES", "PRINCIPAL_DIAGNOSIS", "SECONDARY_DIAGNOSES", "ASSESSMENT_PLAN"):
                items = self._parse_diagnosis_items(sec.content, sec)
                for item in items:
                    canon = self._canonicalize_term(item.raw_term)
                    if canon and canon.lower() not in seen_canonical:
                        seen_canonical.add(canon.lower())
                        candidates.append(item)

        # Phase 2: Chief Complaint / Reason for Admission
        for sec in sections:
            if sec.section_name == "CHIEF_COMPLAINT":
                items = self._parse_complaint_items(sec.content, sec)
                for item in items:
                    canon = self._canonicalize_term(item.raw_term)
                    if canon and canon.lower() not in seen_canonical:
                        seen_canonical.add(canon.lower())
                        candidates.append(item)

        # Phase 3: Past Medical History
        for sec in sections:
            if sec.section_name == "PAST_MEDICAL_HISTORY":
                items = self._parse_pmh_items(sec.content, sec)
                for item in items:
                    canon = self._canonicalize_term(item.raw_term)
                    if canon and canon.lower() not in seen_canonical:
                        seen_canonical.add(canon.lower())
                        candidates.append(item)

        # Phase 4: Unstructured Narrative Fallback
        # If no candidates extracted from high-yield diagnosis sections, parse narrative sections
        has_formal_diags = any(
            c.evidence and c.evidence[0].section in ("DISCHARGE_DIAGNOSES", "PRINCIPAL_DIAGNOSIS", "SECONDARY_DIAGNOSES")
            for c in candidates
        )
        if not has_formal_diags:
            for sec in sections:
                if sec.section_name in ("GENERAL", "DISCHARGE_SUMMARY", "HOSPITAL_COURSE", "HISTORY_OF_PRESENT_ILLNESS"):
                    narrative_items = self._parse_narrative_items(sec.content, sec)
                    for item in narrative_items:
                        canon = self._canonicalize_term(item.raw_term)
                        if canon and canon.lower() not in seen_canonical:
                            seen_canonical.add(canon.lower())
                            candidates.append(item)

        # Corroborate candidates with Hospital Course and Procedures
        self._corroborate_with_hospital_course_and_procedures(candidates, sections, full_text)

        return candidates

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

            # Pattern 2: Explicit admission / diagnosis / condition clauses
            adm_match = re.search(
                r"(?:patient\s+(?:was\s+)?(?:admitted\s+(?:with|for|in)|presented\s+with|diagnosed\s+with|treated\s+for|has|had)|admitted\s+(?:with|for|in)|impression\s*:\s*)\s+([^.]+)",
                sent_clean,
                re.IGNORECASE,
            )
            if adm_match:
                clause = adm_match.group(1).strip()
                for tail_delim in [" on ", " treated with ", " managed with ", " secondary to ", " requiring ", " with good response", " with symptom"]:
                    if tail_delim in clause.lower():
                        idx = clause.lower().find(tail_delim)
                        clause = clause[:idx].strip()

                sub_clauses = re.split(r"\s+and\s+|\s*,\s*", clause)
                for sc in sub_clauses:
                    sc = sc.strip()
                    sc = re.sub(r"^(?:a|an|the)\s+", "", sc, flags=re.IGNORECASE).strip()
                    if len(sc) < 3 or sc.lower() in ("stable condition", "discharge", "home", "baseline"):
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

        for line in lines:
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

            # Split diagnosis from attached narrative / treatment
            diag_term, narrative = self._split_term_and_narrative(cleaned_line)
            if not diag_term or len(diag_term) < 3:
                continue

            # Check negation
            polarity, certainty = self._assess_polarity_and_certainty(cleaned_line)

            # Acuity & Laterality
            acuity = self._detect_acuity(diag_term)
            laterality = self._detect_laterality(diag_term)

            # Check procedure / source control intervention
            proc_rel = 1.0 if any(re.search(pat, cleaned_line, re.IGNORECASE) for pat in PROCEDURE_SOURCE_CONTROL_CUES) else 0.0
            treat_rel = 1.0 if any(w in cleaned_line.lower() for w in ["treated", "iv", "antibiotic", "diuresis", "insulin", "surgery", "stent", "infusion"]) else 0.0

            evidence_obj = StructuredEvidence(
                text=line,
                section=sec.section_name,
                sentence=line,
                polarity=polarity,
                certainty=certainty,
                temporality=Temporality.CURRENT,
                evidence_type=EvidenceType.DISCHARGE_SUMMARY,
                clinical_relevance=1.0,
            )

            scores = MultiDimensionalScore(
                evidence_score=1.0,
                diagnostic_certainty=1.0 if certainty == Certainty.CONFIRMED else 0.8 if certainty == Certainty.SUPPORTED else 0.5 if certainty in (Certainty.SUSPECTED, Certainty.POSSIBLE) else 0.0,
                encounter_relevance=1.0,
                role_confidence=1.0,
                semantic_match=1.0,
                specificity_match=1.0,
                contradiction_penalty=1.0 if polarity == NegationStatus.NEGATED else 0.0,
                admitting_score=5.0 + (proc_rel * 2.0) + (1.0 if acuity == Acuity.ACUTE else 0.0),
            )

            cand = ClinicalDiagnosisCandidate(
                raw_term=diag_term,
                normalized_diagnosis=diag_term,
                role=DiagnosisRole.PRIMARY if sec.section_name in ("DISCHARGE_DIAGNOSES", "PRINCIPAL_DIAGNOSIS") else DiagnosisRole.SECONDARY,
                certainty=certainty,
                temporality=Temporality.CURRENT,
                evidence=[evidence_obj],
                evidence_strength=1.0,
                clinical_relevance=1.0,
                encounter_relevance=1.0,
                treatment_relevance=treat_rel,
                procedure_relevance=proc_rel,
                scores=scores,
                is_authorized=polarity != NegationStatus.NEGATED,
                classification_reason=f"Extracted from {sec.section_name}",
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

        # Split sentences
        sentences = [s.strip() for s in re.split(r"[\.\n;]+", cleaned) if s.strip()]
        for s in sentences:
            if len(s) < 4:
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
                clinical_relevance=0.9,
            )
            scores = MultiDimensionalScore(
                evidence_score=0.9,
                diagnostic_certainty=0.9,
                encounter_relevance=1.0,
                role_confidence=0.8,
                admitting_score=4.0,
            )
            cand = ClinicalDiagnosisCandidate(
                raw_term=s,
                normalized_diagnosis=s,
                role=DiagnosisRole.PRIMARY,
                certainty=certainty,
                temporality=Temporality.CURRENT,
                evidence=[ev],
                scores=scores,
                is_authorized=polarity != NegationStatus.NEGATED,
                classification_reason="Documented as chief presenting complaint",
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
            term, _ = self._split_term_and_narrative(cleaned)
            polarity, certainty = self._assess_polarity_and_certainty(cleaned)
            is_resolved = "resolved" in cleaned.lower() or "remote" in cleaned.lower()

            ev = StructuredEvidence(
                text=line,
                section="PAST_MEDICAL_HISTORY",
                sentence=line,
                polarity=polarity,
                certainty=certainty,
                temporality=Temporality.RESOLVED if is_resolved else Temporality.HISTORICAL,
                evidence_type=EvidenceType.HISTORY,
                clinical_relevance=0.2,
            )
            scores = MultiDimensionalScore(
                evidence_score=0.8,
                diagnostic_certainty=1.0,
                encounter_relevance=0.0,
                role_confidence=0.0,
                admitting_score=-10.0,
            )
            cand = ClinicalDiagnosisCandidate(
                raw_term=term,
                normalized_diagnosis=term,
                role=DiagnosisRole.HISTORICAL,
                certainty=certainty,
                temporality=Temporality.RESOLVED if is_resolved else Temporality.HISTORICAL,
                evidence=[ev],
                scores=scores,
                is_authorized=False,
                classification_reason="Past medical history condition without documented active inpatient care",
            )
            candidates.append(cand)

        return candidates

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

        # Look for sentence period
        if ". " in line:
            parts = line.split(". ", 1)
            return parts[0].strip(), parts[1].strip()

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
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip(".,;: ")

    def _corroborate_with_hospital_course_and_procedures(
        self,
        candidates: list[ClinicalDiagnosisCandidate],
        sections: list[ClinicalDocumentSection],
        full_text: str,
    ) -> None:
        """Enrich candidates with corroborating evidence from hospital course, procedures, and objective investigations."""
        hospital_course_text = ""
        investigations_text = ""
        procedure_text = ""

        for sec in sections:
            if sec.section_name == "HOSPITAL_COURSE":
                hospital_course_text = f"{hospital_course_text} {sec.content}"
            elif sec.section_name == "INVESTIGATIONS":
                investigations_text = f"{investigations_text} {sec.content}"
            elif sec.section_name == "PROCEDURES":
                procedure_text = f"{procedure_text} {sec.content}"

        for cand in candidates:
            # Check hospital course
            words = [w for w in cand.normalized_diagnosis.lower().split() if len(w) > 3 and w not in ("with", "without", "acute", "chronic", "type")]
            if words and any(w in hospital_course_text.lower() for w in words):
                cand.encounter_relevance = 1.0
                cand.scores.encounter_relevance = 1.0
                cand.scores.admitting_score += 2.0
                # Add sentence quote as supporting evidence
                for sent in re.split(r"[\.\n]+", hospital_course_text):
                    if any(w in sent.lower() for w in words):
                        cand.evidence.append(
                            StructuredEvidence(
                                text=sent.strip(),
                                section="HOSPITAL_COURSE",
                                sentence=sent.strip(),
                                polarity=NegationStatus.AFFIRMATIVE,
                                certainty=cand.certainty,
                                temporality=Temporality.CURRENT,
                                evidence_type=EvidenceType.HOSPITAL_COURSE,
                                clinical_relevance=1.0,
                            )
                        )
                        break

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
            # Sort active candidates by admitting score descending
            active_candidates.sort(key=lambda c: c.scores.admitting_score, reverse=True)

            # Top candidate becomes PRIMARY
            top = active_candidates[0]
            top.role = DiagnosisRole.PRIMARY
            top.primary_justification = f"Documented chief condition occasioning admission and inpatient treatment: {top.primary_evidence_quote}"
            primary_diag = top

            # Remaining become SECONDARY
            for sec_cand in active_candidates[1:]:
                sec_cand.role = DiagnosisRole.SECONDARY
                sec_cand.classification_reason = f"Active co-existing condition managed during admission: {sec_cand.primary_evidence_quote}"
                secondary_diags.append(sec_cand)

        return ClinicalDiagnosisState(
            document_id=document_id,
            primary_diagnosis=primary_diag,
            secondary_diagnoses=secondary_diags,
            historical_conditions=historical_conditions,
            ruled_out_conditions=ruled_out_conditions,
            uncertain_conditions=uncertain_conditions,
            all_candidates=candidates,
            has_unique_primary=primary_diag is not None,
            audit_notes=[f"Classified {len(candidates)} total entities: 1 Primary, {len(secondary_diags)} Secondary, {len(historical_conditions)} Historical, {len(ruled_out_conditions)} Ruled out."],
        )
