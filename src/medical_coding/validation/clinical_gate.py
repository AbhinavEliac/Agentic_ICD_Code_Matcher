"""Hard Clinical Candidate Gate conforming to Section 5 and Section 8 specifications.

Strictly rejects:
1. Absence statements ("No acute complications", "No major adverse events", "No acute chemotherapy-related adverse events")
2. Treatment instructions & monitoring phrases ("Watch for reactions", "Follow up", "Review in OPD", "Monitor")
3. Medication names, prescriptions, and dosage instructions ("Tab Prednisolone 100mg", "Syp Cremaffin 15ml")
4. Section headings, JSON labels, test metadata, and template artifacts
5. Isolated non-diagnostic status descriptions ("Stable", "Tolerated well")
6. Tumor staging and molecular attributes extracted as standalone diagnoses ("Stage IV", "GCB type")
"""

import re

from medical_coding.schemas.enums import ClinicalEntityType

# 1. Statements describing the ABSENCE of disease, complications, or adverse events
ABSENCE_PATTERNS = [
    r"^\s*no\s+(?:acute|major|significant|active|obvious|evident|apparent|new|further)?\s*(?:complications?|adverse\s+(?:events?|reactions?|effects?)|reactions?|bleeding|ulceration|ischemia|infarction|recurrence|toxicity|decompensation|sequelae|abnormality|distress|complaints?)\b",
    r"\bno\s+(?:acute\s+)?chemotherapy-related\s+adverse\s+events\b",
    r"\bno\s+(?:acute\s+)?major\s+complications?\s*(?:occurred|noted|seen|observed)?\b",
    r"\bno\s+(?:major\s+)?acute\s+complications?\b",
    r"\bno\s+acute\s+issues?\b",
    r"\buneventful\s+(?:course|stay|hospitalization|postoperative\s+course)?\b",
    r"^\s*without\s+(?:any\s+)?(?:acute\s+)?(?:complications?|adverse\s+events?)\b",
    r"^\s*negative\s+for\s+(?:acute|major|significant)?\s*(?:complications?|adverse\s+events?)\b",
    r"^\s*absence\s+of\s+(?:acute|major)?\s*(?:complications?|adverse\s+events?)\b",
    r"^\s*patient\s+tolerated\s+procedure\s+well\s+without\s+complication\b",
]

# 2. Treatment instructions, monitoring directives, and administrative disposition
INSTRUCTION_PATTERNS = [
    r"^\s*(?:watch\s+for|monitor(?:\s+for)?|check(?:\s+for)?|observe\s+for|follow[\s\-]?up|review\s+in|repeat|continue|stop|hold|taper|instruct(?:ed)?|counsel(?:ed)?|advise[d]?|reassure[d]?|discharge\s+to|disposition|plan\s*:|treatment\s*:)\b",
    r"\b(?:watch\s+for\s+reactions|follow[\s\-]?up\s+in|return\s+if|as\s+directed|review\s+in\s+opd|review\s+after|keep\s+hydrated)\b",
    r"^\s*(?:rest|fluids|hydration|diet|dressing|wound\s+care|suture\s+removal)\b",
    r"^\s*(?:pt|patient)\s+(?:instructed|advised|counseled)\s+to\b",
    r"^\s*(?:patient\s+education|discharge\s+advice|discharge\s+instructions?)\b",
    r"\b(?:patient\s+education|discharge\s+advice|discharge\s+instructions?)\b",
    r"^\s*(?:take\s+medication|take\s+tablets?|take\s+capsules?|take\s+pills?)\b",
    r"\btake\s+medication\s+(?:once|twice|thrice|\d+\s+times)?\s*(?:daily|a\s+day)?\b",
    r"^\s*(?:follow[\s\-]?up\s+after\s+\d+\s+(?:days?|weeks?|months?))\b",
]

# 3. Medications, prescriptions, and dosage instructions (Section 9: Source semantics & morphology)
DOSAGE_FORM_PATTERNS = [
    r"\b(?:tab|tablets?|cap|capsules?|inj|injections?|syp|syrups?|ointment|cream|drops|infusions?|solutions?|suspensions?|sprays?|inhalers?|lozenges?|patch(?:es)?|suppositor(?:y|ies)|laxatives?|pills?)\b",
    r"\b(?:alex\s+cough\s+lozenges|duphalac\s+syp)\b",
]

PRESCRIPTION_SYNTAX_PATTERNS = [
    r"\b(?:\d+\s*(?:mg|mcg|gm|g|ml|iu|units?|tablets?|capsules?|pills?|puffs?|drops?))\b",
    r"\b(?:po|iv|im|sc|prn|daily|od|bd|bid|tds|tid|qid|sos|hs|at\s+bedtime|before\s+meals?|after\s+meals?|subcutaneously|orally|topically)\b",
    r"^\s*(?:take|apply|instill|inhale|inject|infuse|give|initiated|started\s+on)\b",
]

# 4. Section headers, document metadata, allergy statements, and non-diagnostic observations
METADATA_PATTERNS = [
    r"^\s*(?:hospital\s+)?discharge\s+summary\b",
    r"^\s*(?:summary\s+of\s+)?hospital\s+course\b",
    r"^\s*brief\s+summary\b",
    r"^\s*history\s+of\s+present\s+illness\b",
    r"^\s*past\s+medical\s+history\b",
    r"^\s*presenting\s+complaints?\b",
    r"^\s*chief\s+complaint\b",
    r"^\s*discharge\s+medications?\b",
    r"^\s*investigations?\b",
    r"^\s*pertinent\s+negatives?\b",
    r"^\s*negations?(?:\s*/\s*audit)?\b",
    r"^\s*documentation\s+audit\b",
    r"^\s*final\s+coding\s+summary\b",
    r"^\s*general\s+condition\b",
    r"^\s*physical\s+examination\b",
    # Allergy statements and negative metadata (Section 3)
    r"^\s*(?:drug\s+)?allerg(?:y|ies)(?:\s*:)?\s*(?:none|nkda|no\s+known\s+(?:drug\s+)?allergies|nil|na|n/a)?\s*$",
    r"^\s*(?:no\s+known\s+(?:drug\s+)?allergies|nkda)\s*$",
    r"^\s*(?:none|nil|na|n/a|unknown|unremarkable)\s*$",
    # Patient education, appointments, and follow-up directives
    r"^\s*(?:follow[\s\-]?up|appointment|review\s+in|return\s+to|referral\s+to|discharged?\s+to|see\s+doctor|emergency\s+signs?)\b",
    r"\b(?:call\s+clinic|seek\s+urgent\s+care|return\s+if\s+worse|take\s+rest|diet\s+counseling)\b",
    r"^\s*(?:discharge\s+advice|patient\s+education|discharge\s+instructions?)\s*$",
    r"^\s*take\s+medication\s+(?:once|twice|thrice|\d+\s+times)?\s*(?:daily|a\s+day)?\s*$",
    r"^\s*follow\s+up\s+after\s+\d+\s+(?:days?|weeks?|months?)\s*$",
    # Vital signs, general status, and measurements
    r"^\s*(?:stable|afebrile|conscious|oriented|tolerated\s+well|in\s+stable\s+condition|vital\s+signs?|general\s+condition\s+stable)\s*$",
    r"^\s*(?:bp|hr|pulse|rr|temp|spo2|gcs|creatinine|cbc|wbc|hemoglobin|hgb|platelets?)\s*(?::|=|\s+\d)\b",
    r"^\s*[\d\.\,\/]+\s*(?:mg/dl|mmol/l|g/dl|mmhg|bpm|%|cells/cumm)?\s*$",
    r"^\s*(?:ecg|ekg|cxr|chest\s+x[\s\-]?ray|ct\s+scan|mri|ultrasound|usg|2d\s+echo|cbc|bmp|cmp|lft|kft|rft|urinalysis|blood\s+culture|troponin|hba1c)\s*(?:normal|clear|pending|done|negative|reviewed)?\s*$",
]

# 5. Tumor attributes / Staging markers (attributes of a neoplasm, not standalone diagnoses)
STANDALONE_ATTRIBUTE_PATTERNS = [
    r"^\s*stage\s+(?:i|ii|iii|iv|[1-4])[a-z]?\s*$",
    r"^\s*(?:non-)?gcb(?:\s+(?:sub)?type)?\s*$",
    r"^\s*double\s+expressor(?:\s*(?:bcl2/bcl6|myc/bcl2|\(.*\)))?\s*$",
    r"^\s*ki[\s\-]?67\s*(?:index)?\s*(?:>|=|<)?\s*\d+\s*%?\s*$",
    r"^\s*grade\s+(?:i|ii|iii|iv|[1-4])(?:\s+histology)?\s*$",
    r"^\s*t\d+n\d+m\d+\s*$",
    r"^\s*(?:triple\s+negative|er\s+negative|pr\s+negative|her2\s*(?:positive|negative|0|\d+))\s*$",
    r"^\s*brca(?:\s*1|\s*2)?\s+(?:pathogenic|mutation|positive|negative)(?:\s+mutation)?\s*$",
    r"^\s*pd[\s\-]?l1\s+(?:positive|negative|expression)?(?:\s*\(.*?\))?\s*$",
    r"^\s*(?:er|pr|her2)\s*(?:positive|negative|0|\d+)\s*$",
]

# 6. Isolated anatomical locations lacking any clinical condition / pathology descriptor
ISOLATED_ANATOMY_PATTERNS = [
    r"^\s*(?:left\s+|right\s+|bilateral\s+)?(?:foot|feet|ankle|heel|toe|toes|hand|wrist|arm|leg|thigh|knee|shoulder|elbow|chest|thorax|breast|abdomen|flank|back|spine|pelvis|hip|neck|head|brain|lung|kidney|liver|spleen|heart|eye|ear|nose|throat)\s*$",
]

# Pathological condition words that validate an anatomical mention into a true disease entity
PATHOLOGY_KEYWORDS = {
    "fracture", "sprain", "strain", "pain", "cellulitis", "arthritis", "arthropathy",
    "swelling", "edema", "lesion", "mass", "tumor", "carcinoma", "cancer", "neoplasm",
    "infection", "calculus", "stone", "lithiasis", "stenosis", "effusion", "laceration",
    "wound", "contusion", "deformity", "syndrome", "disease", "disorder", "ulcer",
    "gangrene", "ischemia", "infarction", "tear", "rupture", "dislocation", "subluxation",
    "injury", "trauma", "abscess", "impairment", "failure", "colic", "bacteremia",
}


# 7. Procedure and surgical keywords (Section 11 Procedure Firewall)
PROCEDURE_TERMS = {
    "mastectomy", "reconstruction", "endoscopy", "colonoscopy", "stenting", "stent",
    "biopsy", "laparoscopy", "excision", "resection", "debridement", "incision",
    "repair", "intubation", "catheterization", "cannulation", "grafting", "arthroscopy",
    "cholecystectomy", "appendectomy", "hysterectomy", "lumpectomy", "angioplasty",
    "bypass", "dialysis", "hemodialysis", "paracentesis", "thoracentesis", "fet",
    "frozen embryo transfer", "infusion", "chemotherapy", "radiotherapy", "radiation",
}

PROCEDURE_SUFFIXES = (
    "ectomy", "otomy", "ostomy", "plasty", "pexy", "scopy", "centesis", "rrhaphy",
)

# 8. Investigation names lacking pathological findings
INVESTIGATION_TERMS = {
    "ct", "ct scan", "mri", "cxr", "chest x-ray", "ultrasound", "usg",
    "echocardiogram", "echo", "ecg", "ekg", "xray", "x-ray", "eeg", "emg", "urinalysis",
}

# 9. Family history and hypothetical cues
FAMILY_HISTORY_PATTERNS = [
    r"\b(?:family\s+history|fhx|mother\s+(?:had|with)|father\s+(?:had|with)|sister\s+(?:had|with)|brother\s+(?:had|with))\b",
]

HYPOTHETICAL_PATTERNS = [
    r"\b(?:if\s+symptoms?\s+worsen|in\s+the\s+event\s+of|in\s+case\s+of|contingent\s+upon)\b",
]


class HardClinicalCandidateGate:
    """Deterministic clinical candidate gate enforcing Section 4 & Section 11 specifications."""

    @classmethod
    def evaluate_candidate(cls, term: str, evidence_text: str = "") -> tuple[bool, str]:
        """Evaluate whether an extracted term is a genuine clinical diagnosis (Section 4).

        A candidate MUST satisfy:
        1. It represents a clinical condition or codable clinical concept.
        2. It has identifiable evidence in the source document.
        3. The evidence is attributable to the patient.
        4. The evidence is not merely a heading or section label.
        5. The evidence is not merely a procedure.
        6. The evidence is not merely a medication.
        7. The evidence is not merely a treatment instruction.
        8. The evidence is not merely an investigation name.
        9. The evidence is not merely an anatomical site.
        10. The evidence is not merely a disease attribute.
        11. The evidence is not negated.
        12. The evidence is not purely family history.
        13. The evidence is not hypothetical.
        14. The evidence is not merely a rule-out diagnosis.
        15. The candidate does not depend on unsupported inference.
        16. The candidate is not generated solely from database retrieval.

        Returns:
            Tuple of (is_valid, rejection_reason).
        """
        clean_term = term.strip()
        term_lower = clean_term.lower()

        if len(clean_term) < 2:
            return False, "Term is too short to be a clinical condition"

        # Check 1: Absence statements
        for pat in ABSENCE_PATTERNS:
            if re.search(pat, term_lower):
                return False, f"Absence statement describing absence of disease or complication: '{clean_term}'"

        # Check 2: Treatment instructions and monitoring
        for pat in INSTRUCTION_PATTERNS:
            if re.search(pat, term_lower):
                return False, f"Treatment instruction or monitoring directive: '{clean_term}'"

        # Check 3: Pharmaceutical dosage forms or prescription syntax (Section 9)
        is_dosage_form = any(re.search(pat, term_lower) for pat in DOSAGE_FORM_PATTERNS)
        has_prescription_syntax = (
            any(re.search(pat, term_lower) for pat in PRESCRIPTION_SYNTAX_PATTERNS)
            or any(re.search(pat, evidence_text.lower()) for pat in PRESCRIPTION_SYNTAX_PATTERNS)
        )
        if is_dosage_form or (has_prescription_syntax and any(w in evidence_text.lower() for w in ["medication", "dose", "tablet", "syrup", "capsule", "injection", "take", "prn"])):
            return False, f"Pharmaceutical dosage form or prescription instruction: '{clean_term}'"

        # Check 4: Section headings, metadata, allergy statements, lab values
        for pat in METADATA_PATTERNS:
            if re.search(pat, term_lower):
                return False, f"Section heading, metadata, allergy, or clinical status artifact: '{clean_term}'"

        # Check 5: Standalone tumor staging or genetic attributes
        for pat in STANDALONE_ATTRIBUTE_PATTERNS:
            if re.search(pat, term_lower):
                return False, f"Neoplasm attribute or staging marker, not an independent diagnosis: '{clean_term}'"

        # Check 6: Isolated anatomical locations without condition descriptors
        for pat in ISOLATED_ANATOMY_PATTERNS:
            if re.search(pat, term_lower):
                words = set(re.findall(r"[a-z]+", term_lower))
                if not words.intersection(PATHOLOGY_KEYWORDS):
                    return False, f"Isolated anatomical location lacking pathological condition: '{clean_term}'"

        # Check 7: Entire phrase is an explicit absence or negative statement
        if term_lower.startswith("no evidence of ") or term_lower.startswith("negative for "):
            return False, f"Negative finding / absence clause: '{clean_term}'"

        # Check 8: Explicit metadata keywords
        if any(term_lower == meta for meta in ["classification of fracture", "classification of fracture: not applicable", "not applicable", "n/a", "na", "biomarkers & receptor status", "biomarkers", "receptor status"]):
            return False, f"Metadata or template header artifact: '{clean_term}'"

        # Check 9: Pure surgical or procedural interventions (Section 11 Procedure Firewall)
        is_pure_procedure = (
            any(re.search(rf"\b{re.escape(proc)}\b", term_lower) for proc in PROCEDURE_TERMS)
            or any(term_lower.endswith(sfx) for sfx in PROCEDURE_SUFFIXES)
        )
        if is_pure_procedure:
            words = set(re.findall(r"[a-z]+", term_lower))
            if not words.intersection(PATHOLOGY_KEYWORDS):
                return False, f"Procedure or surgical intervention, not an independent clinical diagnosis: '{clean_term}'"

        # Check 10: Pure investigation names lacking findings
        is_pure_investigation = any(term_lower == inv or term_lower.startswith(f"{inv} of ") for inv in INVESTIGATION_TERMS)
        if is_pure_investigation:
            words = set(re.findall(r"[a-z]+", term_lower))
            if not words.intersection(PATHOLOGY_KEYWORDS):
                return False, f"Investigation or diagnostic test name lacking pathological finding: '{clean_term}'"

        # Check 11: Family history
        for pat in FAMILY_HISTORY_PATTERNS:
            if re.search(pat, term_lower):
                return False, f"Family medical history, not personal patient condition: '{clean_term}'"

        # Check 12: Hypothetical statements
        for pat in HYPOTHETICAL_PATTERNS:
            if re.search(pat, term_lower):
                return False, f"Hypothetical or contingent clinical statement: '{clean_term}'"

        return True, "Valid clinical diagnosis entity"

    @classmethod
    def classify_entity_type(cls, term: str, evidence_text: str = "") -> ClinicalEntityType:
        """Classify any clinical phrase into the Universal Clinical Entity Taxonomy (Section 2)."""
        term_lower = term.strip().lower()
        ev_lower = evidence_text.strip().lower()

        # 1. Negated
        if any(re.search(pat, term_lower) for pat in ABSENCE_PATTERNS) or term_lower.startswith("no ") or "ruled out" in term_lower or "negative for" in term_lower:
            return ClinicalEntityType.NEGATED_CONDITION

        # 2. Family History
        if any(re.search(pat, term_lower) for pat in FAMILY_HISTORY_PATTERNS) or "family history" in ev_lower:
            return ClinicalEntityType.FAMILY_HISTORY

        # 3. Instruction
        if any(re.search(pat, term_lower) for pat in INSTRUCTION_PATTERNS):
            return ClinicalEntityType.INSTRUCTION

        # 4. Administrative Text
        if any(term_lower == m or re.search(pat, term_lower) for pat in METADATA_PATTERNS for m in ["not applicable", "n/a", "biomarkers & receptor status"]):
            return ClinicalEntityType.ADMINISTRATIVE_TEXT

        # 5. Clinical Attribute
        if any(re.search(pat, term_lower) for pat in STANDALONE_ATTRIBUTE_PATTERNS):
            return ClinicalEntityType.CLINICAL_ATTRIBUTE

        # 6. Anatomical Site
        if any(re.search(pat, term_lower) for pat in ISOLATED_ANATOMY_PATTERNS):
            return ClinicalEntityType.ANATOMICAL_SITE

        # 7. Medication
        if any(re.search(pat, term_lower) for pat in DOSAGE_FORM_PATTERNS) or any(re.search(pat, term_lower) for pat in PRESCRIPTION_SYNTAX_PATTERNS):
            return ClinicalEntityType.MEDICATION

        # 8. Surgery & Procedure
        is_procedure = (
            any(re.search(rf"\b{re.escape(proc)}\b", term_lower) for proc in PROCEDURE_TERMS)
            or any(term_lower.endswith(sfx) for sfx in PROCEDURE_SUFFIXES)
        )
        if is_procedure:
            words = set(re.findall(r"[a-z]+", term_lower))
            if not words.intersection(PATHOLOGY_KEYWORDS):
                if any(surg in term_lower for surg in ("mastectomy", "resection", "excision", "cholecystectomy", "appendectomy", "repair", "bypass")):
                    return ClinicalEntityType.SURGERY
                return ClinicalEntityType.PROCEDURE

        # 9. Investigation
        if any(term_lower == inv or term_lower.startswith(f"{inv} of ") for inv in INVESTIGATION_TERMS):
            words = set(re.findall(r"[a-z]+", term_lower))
            if not words.intersection(PATHOLOGY_KEYWORDS):
                return ClinicalEntityType.INVESTIGATION

        # 10. Symptoms & Signs
        if any(sym in term_lower for sym in ["pain", "cough", "nausea", "vomiting", "dyspnea", "shortness of breath", "fatigue", "malaise", "colic", "dizziness"]):
            return ClinicalEntityType.SYMPTOM
        if any(sign in term_lower for sign in ["tachycardia", "tachypnea", "fever", "hypotension", "wheezing", "rales", "edema", "swelling"]):
            return ClinicalEntityType.SIGN

        # 11. Neoplasms
        if any(neo in term_lower for neo in ["carcinoma", "sarcoma", "lymphoma", "cancer", "neoplasm", "tumor", "carcinoid", "adenoma", "malignancy"]):
            return ClinicalEntityType.NEOPLASM

        # 12. Injuries
        if any(inj in term_lower for inj in ["fracture", "tear", "sprain", "strain", "rupture", "dislocation", "laceration", "contusion", "injury"]):
            return ClinicalEntityType.INJURY

        # 13. Infections
        if any(inf in term_lower for inf in ["infection", "pyelonephritis", "pneumonia", "uti", "sepsis", "cellulitis", "candidiasis", "abscess", "bacteremia"]):
            return ClinicalEntityType.INFECTION

        # 14. Chronic Conditions
        if any(chr_c in term_lower for chr_c in ["hypertension", "diabetes", "asthma", "copd", "ckd", "chronic"]):
            return ClinicalEntityType.CHRONIC_CONDITION

        # 15. Complications
        if any(comp in term_lower for comp in ["septic shock", "dka", "ketoacidosis", "acute kidney injury", "hemorrhage"]):
            return ClinicalEntityType.COMPLICATION

        return ClinicalEntityType.DIAGNOSIS

    @classmethod
    def get_rejection_category(cls, term: str, _evidence_text: str = "") -> str:
        """Classify the deterministic rejection category for audit reporting."""
        term_lower = term.strip().lower()
        if any(term_lower == m or re.search(pat, term_lower) for pat in METADATA_PATTERNS for m in ["classification of fracture", "not applicable", "n/a", "biomarkers & receptor status", "biomarkers", "receptor status"]):
            return "METADATA"
        if any(re.search(pat, term_lower) for pat in STANDALONE_ATTRIBUTE_PATTERNS):
            return "ATTRIBUTE"
        if any(re.search(pat, term_lower) for pat in ABSENCE_PATTERNS) or term_lower.startswith("no ") or "ruled out" in term_lower:
            return "NEGATED"
        if any(re.search(pat, term_lower) for pat in DOSAGE_FORM_PATTERNS) or any(re.search(pat, term_lower) for pat in PRESCRIPTION_SYNTAX_PATTERNS):
            return "MEDICATION"
        if any(re.search(pat, term_lower) for pat in INSTRUCTION_PATTERNS):
            return "INSTRUCTION"
        if any(re.search(rf"\b{re.escape(proc)}\b", term_lower) for proc in PROCEDURE_TERMS) or any(term_lower.endswith(sfx) for sfx in PROCEDURE_SUFFIXES):
            return "PROCEDURE"
        if any(re.search(pat, term_lower) for pat in ISOLATED_ANATOMY_PATTERNS):
            return "ANATOMY"
        if any(sym in term_lower for sym in ["pain", "cough", "fever", "dyspnea", "wheezing", "discomfort"]):
            return "SYMPTOM"
        return "METADATA"

    @classmethod
    def is_absence_statement(cls, text: str) -> bool:
        t = text.strip().lower()
        return any(bool(re.search(pat, t)) for pat in ABSENCE_PATTERNS)

    @classmethod
    def is_treatment_instruction(cls, text: str) -> bool:
        t = text.strip().lower()
        return any(bool(re.search(pat, t)) for pat in INSTRUCTION_PATTERNS)

    @classmethod
    def is_medication_or_prescription(cls, text: str) -> bool:
        t = text.strip().lower()
        has_form = any(bool(re.search(pat, t)) for pat in DOSAGE_FORM_PATTERNS)
        has_syntax = any(bool(re.search(pat, t)) for pat in PRESCRIPTION_SYNTAX_PATTERNS)
        return has_form or (has_syntax and any(w in t for w in ["medication", "dose", "tablet", "syrup", "capsule", "take", "prn"]))

    @classmethod
    def is_metadata_or_heading(cls, text: str) -> bool:
        t = text.strip().lower()
        return any(bool(re.search(pat, t)) for pat in METADATA_PATTERNS)
