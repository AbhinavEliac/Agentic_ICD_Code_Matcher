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
    r"^\s*(?:clinical|general|systemic)\s+examination\b",
    r"^\s*course\s+in\s+(?:the\s+)?hospital(?:\s+and\s+discussion)?\b",
    r"^\s*personal\s+history\b",
    r"^\s*discussion\b",
    # Physical exam findings, vitals, and normal systemic observations
    r"\bpatient\s+alert\s+and\s+responsive\b",
    r"\b(?:alert\s+and\s+responsive|conscious\s+and\s+oriented)\b",
    r"\b(?:rs\s*:\s*aebe|cvs\s*:\s*s1|cns\s*:\s*nad|pa\s*:\s*soft)\b",
    r"\b(?:temperature\s*:|pulse\s*:|respiratory\s+rate\s*:|bp\s*:|spo2\s*:)\b",
    r"\b(?:soft[,\s]+non[\s\-]?tender|heard\s+well)\b",
    # Administrative phrases and non-disease encounter boilerplate
    r"\b(?:patient\s+was\s+)?admitted\s+with\s+above\s+mentioned\s+complaints\b",
    r"\b(?:admitted\s+for\s+)?further\s+management\b",
    r"\babove\s+mentioned\s+complaints\b",
    r"\bfeared\s+health\s+complaint\b",
    r"\bcomplaint\s+in\s+whom\s+no\s+diagnosis\s+is\s+made\b",
    # Cross-references, section references, and meta-narrative boilerplate
    r"\b(?:symptoms?|complaints?|conditions?|diagnos[ei]s)\s+(?:described|mentioned|listed|noted|specified)\s+(?:in|under|above|below)\b",
    r"\b(?:described|mentioned|listed|noted|specified)\s+(?:in|under)\s+(?:the\s+)?(?:final|discharge|admitting|admission)?\s*diagnos[ei]s\b",
    r"\b(?:symptoms?|complaints?)\s+as\s+(?:above|below|mentioned)\b",
    r"\b(?:as\s+described\s+above|as\s+mentioned\s+above|described\s+above|as\s+above)\b",
    r"\babove\s+mentioned\s+(?:complaints?|symptoms?|conditions?)\b",
    r"\bseveral\s+days\s+of\s+(?:the\s+)?symptoms\b",
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
    r"^\s*(?:patient\s+name|mrn|admission\s+date|discharge\s+date|admit\s+date|date\s+of\s+admission|date\s+of\s+discharge|dob|age|sex|gender)\s*[:\-–—]",
    r"^\s*metastatic\s+sites?\s*(?::\s*)?$",
    r"^\s*(?:oncology\s+attributes?(?:\s*(?:&|and)?\s*biomarkers?)?|biomarkers?\s*(?:&|and)?\s*receptor\s+status)\s*(?::\s*)?$",
    r"^\s*castration[- ]resistant\s+status(?:\s+on.*)?$",
    r"^\s*(?:androgen\s+deprivation\s+therapy|adt|palliative\s+radiotherapy)\s*$",
]

# 5. Tumor attributes / Staging markers (attributes of a neoplasm, not standalone diagnoses)
STANDALONE_ATTRIBUTE_PATTERNS = [
    r"^\s*stage\s+(?:i|ii|iii|iv|[1-4])[a-z]?\s*$",
    r"^\s*(?:non-)?gcb(?:\s+(?:sub)?type)?\s*$",
    r"^\s*double\s+expressor(?:\s*(?:bcl2/bcl6|myc/bcl2|\(.*\)))?\s*$",
    r"^\s*ki[\s\-]?67\s*(?:index)?\s*(?:>|=|<)?\s*\d+\s*%?\s*$",
    r"^\s*grade\s+(?:i|ii|iii|iv|[1-4])(?:\s+histology)?\s*$",
    r"^\s*grade\s+group\s*[1-5]\s*$",
    r"^\s*gleason(?:\s+score)?\s*(?::|=|\b)\s*\d+(?:\s*(?:\+|\/|\=|\(.*?\))\s*\d+)?\s*$",
    r"^\s*psa(?:\s+level)?\s*(?::|=|\b)\s*[\d\.\,]+(?:\s*ng\s*/\s*m[lL])?\s*$",
    r"^\s*(?:metastatic\s+)?castrat(?:ion|e)[- ]resistant(?:\s+(?:status|state|prostate\s+cancer))?\s*$",
    r"^\s*(?:mcrpc|crpc|nmcrpc|mhspc)\s*$",
    r"^\s*metastatic\s+sites?\s*$",
    r"^\s*t\d+n\d+m\d+\s*$",
    r"^\s*(?:triple\s+negative|er\s+negative|pr\s+negative|her2\s*(?:positive|negative|0|\d+))\s*$",
    r"^\s*brca(?:\s*1|\s*2)?\s+(?:pathogenic|mutation|positive|negative)(?:\s+mutation)?\s*$",
    r"^\s*pd[\s\-]?l1\s+(?:positive|negative|expression)?(?:\s*\(.*?\))?\s*$",
    r"^\s*(?:er|pr|her2)\s*(?:positive|negative|0|\d+)\s*$",
]

# 6. Isolated anatomical locations lacking any clinical condition / pathology descriptor
ISOLATED_ANATOMY_PATTERNS = [
    r"^\s*(?:left\s+|right\s+|bilateral\s+)?(?:foot|feet|ankle|heel|toe|toes|hand|wrist|arm|leg|thigh|knee|shoulder|elbow|chest|thorax|breast|abdomen|flank|back|spine|pelvis|hip|neck|head|brain|lung|kidney|liver|spleen|heart|eye|ear|nose|throat|retroperitoneal|intra-abdominal|pelvic|axial|skeletal|osseous)\s*$",
    r"^\s*(?:intra-abdominal|retroperitoneal|pelvic|axial|skeletal|osseous)\s*$",
]

# Pathological condition words that validate an anatomical mention into a true disease entity
PATHOLOGY_KEYWORDS = {
    "fracture", "sprain", "strain", "pain", "cellulitis", "arthritis", "arthropathy",
    "swelling", "edema", "lesion", "mass", "tumor", "tumour", "carcinoma", "cancer", "neoplasm",
    "infection", "calculus", "stone", "lithiasis", "stenosis", "effusion", "laceration",
    "wound", "contusion", "deformity", "syndrome", "disease", "disorder", "ulcer",
    "gangrene", "ischemia", "infarction", "tear", "rupture", "dislocation", "subluxation",
    "injury", "trauma", "abscess", "impairment", "failure", "colic", "bacteremia",
    "hypertension", "diabetes", "asthma", "copd", "cirrhosis", "hepatitis", "pancreatitis",
    "gastritis", "colitis", "enteritis", "appendicitis", "cholecystitis", "pyelonephritis",
    "nephritis", "sepsis", "shock", "pneumonia", "bronchitis", "anemia", "acidosis",
    "alkalosis", "ischemia", "stenosis", "thrombosis", "embolism", "atherosclerosis",
    "cardiomyopathy", "arrhythmia", "fibrillation", "leukemia", "lymphoma", "sarcoma",
    "melanoma", "adenoma", "polyp", "cyst", "perforation", "obstruction", "ileus",
    "fistula", "diverticulitis", "diverticulosis", "fibrosis", "osteomyelitis",
    "gout", "neuropathy", "radiculopathy", "dementia", "adenitis", "spasm",
    "metastasis", "metastases", "metastatic", "malignancy", "malignant", "carcinomatosis",
    "effusion", "atelectasis", "pneumothorax", "hemothorax", "consolidation", "ascites",
    "hernia", "herniation", "hernias", "dehydration", "hypovolemia", "hypervolemia",
    "urticaria", "urticarial", "dermatitis", "eczema", "psoriasis", "anaphylaxis", "angioedema",
    "hypothyroidism", "hyperthyroidism", "thyroiditis", "goiter", "epilepsy", "seizure",
    "convulsion", "parkinson", "alzheimer", "dementia", "reflux", "gerd", "esophagitis",
    "regurgitation", "insufficiency", "prolapse", "incompetence", "lupus", "erythematosus",
    "sjogren", "scleroderma", "sarcoidosis", "amyloidosis", "hemophilia", "thalassemia",
    "cataract", "glaucoma", "retinopathy", "tuberculosis", "malaria", "dengue", "typhoid",
    "depression", "anxiety", "bipolar", "schizophrenia", "psychosis", "malformation", "anomaly",
    "deviation", "deviated", "hypertrophy", "hypertrophic", "hyperplasia", "atrophy", "atrophic",
    "hypoplasia", "dysplasia", "thickening", "narrowing", "collapse", "perforation", "calcification",
    "sclerosis", "fibrosis", "adhesions", "dns", "mastoiditis",
}

# Clinical symptoms, signs, and patient complaints
SYMPTOM_KEYWORDS = {
    "pain", "colic", "discomfort", "ache", "aching", "soreness", "tenderness",
    "fever", "pyrexia", "chill", "chills", "rigor", "rigors", "sweating", "diaphoresis",
    "cough", "coughing", "hemoptysis", "dyspnea", "breathlessness", "tachypnea",
    "wheeze", "wheezing", "stridor", "rhonchi", "rales", "crackles",
    "nausea", "vomiting", "emesis", "dyspepsia", "indigestion", "heartburn",
    "dysphagia", "odynophagia", "diarrhea", "constipation", "obstipation",
    "hematochezia", "melena", "hematemesis", "jaundice", "icterus",
    "dizziness", "vertigo", "syncope", "presyncope", "lightheadedness",
    "headache", "migraine", "cephalea", "fatigue", "malaise", "lethargy",
    "weakness", "asthenia", "somnolence", "drowsiness", "insomnia",
    "edema", "swelling", "swollen", "distension", "bloating", "ascites",
    "rash", "erythema", "pruritus", "itching", "lesion", "lesions",
    "cyanosis", "pallor", "flushing", "palpitation", "palpitations",
    "tachycardia", "bradycardia", "arrhythmia", "hypotension", "hypertension",
    "oliguria", "anuria", "polyuria", "hematuria", "dysuria", "nocturia",
    "incontinence", "retention", "urgency", "frequency",
    "confusion", "delirium", "altered", "encephalopathy", "seizure", "seizures", "convulsion",
    "tremor", "ataxia", "numbness", "paresthesia", "tingling", "hypesthesia",
    "paralysis", "paresis", "plegia", "spasticity", "rigidity", "stiffness",
    "anorexia", "cachexia", "weight loss", "weight gain",
}

# Recognized medical pathology, morphology, and clinical suffixes
MEDICAL_CONDITION_SUFFIXES = (
    "itis", "oma", "osis", "emia", "aemia", "pathy", "algia", "dynia",
    "pnea", "uria", "stenosis", "megaly", "penia", "cytosis", "iasis",
    "spasm", "derma", "ectasis", "malacia", "necrosis", "rrhea", "rrhagia",
    "cele", "ptosis", "lysis", "plegia", "paresis", "sclerosis",
    "ism", "ia", "ence", "ency", "lapse",
)

# 7. Procedure and surgical keywords (Section 11 Procedure Firewall)
PROCEDURE_TERMS = {
    "mastectomy", "reconstruction", "endoscopy", "colonoscopy", "stenting", "stent",
    "biopsy", "laparoscopy", "excision", "resection", "debridement", "incision",
    "repair", "intubation", "catheterization", "cannulation", "grafting", "arthroscopy",
    "cholecystectomy", "appendectomy", "hysterectomy", "lumpectomy", "angioplasty",
    "bypass", "dialysis", "hemodialysis", "paracentesis", "thoracentesis", "fet",
    "frozen embryo transfer", "infusion", "chemotherapy", "radiotherapy", "radiation",
    "salpingo-oophorectomy", "salpingooophorectomy", "oophorectomy", "salpingectomy",
    "lymphadenectomy", "axillary dissection", "lymph node dissection",
    "sentinel node biopsy", "sentinel lymph node biopsy", "cystectomy",
    "adhesiolysis", "lysis of adhesions", "ablation", "cauterization",
    "procedure", "surgery", "operation", "intervention", "surgical procedure",
    "surgical intervention", "procedure done", "surgery done", "operation done",
}

PROCEDURE_SUFFIXES = (
    "ectomy", "otomy", "ostomy", "plasty", "pexy", "scopy", "centesis", "rrhaphy",
)

# Generic procedural completion / performance notices lacking underlying disease
GENERIC_PROCEDURE_PATTERNS = [
    r"^\s*(?:elective\s+|major\s+|minor\s+|surgical\s+|operative\s+|bedside\s+|diagnostic\s+)?(?:procedure|surgery|operation|intervention)\s*(?:done|performed|completed|tolerated|undertaken|successful|finished|noted)?\s*$",
    r"\b(?:procedure\s+done|surgery\s+done|operation\s+done|procedure\s+performed|surgery\s+performed|operation\s+performed|intervention\s+done)\b",
    r"^\s*(?:procedure|surgery|operation|intervention)\s*$",
]

# Non-clinical activities, everyday events, personal actions, and bodily functions without pathology
NON_CLINICAL_ACTIVITY_PATTERNS = [
    r"\b(?:towel|blanket|bedsheet|pillow|napkin|tissue|clothes|linen|tray|remote|phone|call\s*bell)\b",
    r"\b(?:picked\s*up|picking\s*up|pick\s*up|put\s*down|dropped|wiped|cleaned|folded|reached\s*for)\b",
    r"\b(?:i\s+)?(?:pooped|pooping|bowel\s*movement|peed|peeing|urinated|urination|voided|defecated)\b",
    r"\b(?:i\s+pooped|patient\s+pooped|i\s+peed|patient\s+peed|towel\s+picked\s+up|picked\s+up\s+towel)\b",
    r"^\s*(?:i|we|he|she|pt|patient)?\s*(?:ate|eating|eats|drank|drinking|drinks|slept|sleeping|sleeps|walked|walking|walks|ambulated|ambulating|sat|sitting|sits|stood|standing|stands|woke|waking|bathed|bathing|showered|showering|brushed|brushing|dressed|dressing)\b",
    r"^\s*(?:i|we)\s+[a-z]+",
]

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

        # Check 0a: Non-clinical activities, daily living actions, and personal events
        for pat in NON_CLINICAL_ACTIVITY_PATTERNS:
            if re.search(pat, term_lower):
                return False, f"Non-clinical activity, everyday personal action, or household event: '{clean_term}'"

        # Check 0b: Generic procedural completion or intervention statements
        for pat in GENERIC_PROCEDURE_PATTERNS:
            if re.search(pat, term_lower):
                return False, f"Generic procedure statement or completion notice: '{clean_term}'"

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
        has_prescription_syntax = any(re.search(pat, term_lower) for pat in PRESCRIPTION_SYNTAX_PATTERNS)
        if is_dosage_form or has_prescription_syntax:
            return False, f"Pharmaceutical dosage form or prescription instruction: '{clean_term}'"
        if any(w in term_lower for w in ["tablet", "syrup", "capsule", "injection", "mg po", "mg iv", "mcg", "tablets", "capsules", "lozenges"]):
            return False, f"Pharmaceutical dosage form or prescription instruction: '{clean_term}'"

        # Check 4: Section headings, metadata, allergy statements, lab values
        for pat in METADATA_PATTERNS:
            if re.search(pat, term_lower):
                return False, f"Section heading, metadata, allergy, or clinical status artifact: '{clean_term}'"

        # Check 4b: Physical exam, vital signs, systemic examination noise, and meta-narrative boilerplate
        if any(w in term_lower for w in [
            "patient alert", "responsive temperature", "temperature :", "pulse :",
            "respiratory rate :", "bp :", "spo2 :", "rs :", "cvs :", "cns :", "pa :",
            "aebe", "s1, s2 heard well", "soft, non tender", "further management",
            "above mentioned complaints", "symptoms described", "described in the final diagnosis",
            "described in final diagnosis", "symptoms described in", "several days of the symptoms",
        ]):
            return False, f"Physical exam, vital sign, or administrative boilerplate artifact: '{clean_term}'"

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
        term_no_punct = re.sub(r"[:\s]+$", "", term_lower).strip()
        if (
            term_no_punct.startswith("classification of")
            or term_no_punct.startswith("type of fracture")
            or any(term_no_punct == meta for meta in ["classification", "not applicable", "n/a", "na", "biomarkers & receptor status", "biomarkers", "receptor status"])
        ):
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

        # Check 13: Clinical Medical Entity Validation (Strict Disease or Symptom Requirement)
        # Every candidate MUST carry a disease, disorder, pathology, infection, neoplasm, injury, or clinical symptom/sign.
        term_words = set(re.findall(r"[a-z0-9]+", term_lower))
        has_pathology = bool(term_words.intersection(PATHOLOGY_KEYWORDS))
        has_symptom = bool(term_words.intersection(SYMPTOM_KEYWORDS))
        has_med_suffix = any(any(w.endswith(sfx) for sfx in MEDICAL_CONDITION_SUFFIXES) for w in term_words if len(w) > 4)
        has_microbiology = any(org in term_lower for org in ["e. coli", "escherichia coli", "klebsiella", "pseudomonas", "staphylococcus", "streptococcus", "candida", "h. pylori", "helicobacter", "rhinovirus", "influenza", "coronavirus", "bacteremia"])
        has_lab_abnormality = any(lab in term_lower for lab in ["leukocytosis", "thrombocytopenia", "hypokalemia", "hyperkalemia", "hyponatremia", "acidosis", "alkalosis", "anemia", "creatinine", "troponin", "hyperglycemia", "hypoglycemia", "elevated"])

        CLINICAL_ROOTS = (
            "stone", "calcul", "lithias", "itis", "oma", "osis", "emia", "pathy", "algia",
            "pnea", "uria", "stenos", "megaly", "penia", "cytos", "spasm", "infect",
            "fractur", "sprain", "strain", "ulcer", "injur", "trauma", "wound", "syndrom",
            "diseas", "disorder", "failur", "ischemi", "infarct", "cancer", "carcinom",
            "tumor", "tumour", "neoplasm", "abscess", "cellulit", "aneurysm", "thrombos",
            "embol", "anemi", "diabet", "asthma", "copd", "cirrhos", "fibros", "gout",
            "neuropath", "pain", "fever", "cough", "nausea", "vomit", "dyspnea", "headache",
            "edema", "swelling", "rash", "bleed", "hemorrhag", "diarrhea", "constipat",
            "colic", "jaundic", "syncope", "dizzin", "vertigo", "fatigue", "malaise",
            "seizur", "convuls", "tremor", "ataxia", "paralys", "paresis", "hypertens",
            "hypotens", "arrhythm", "tachycard", "bradycard", "fibrillat", "cardiomyopath",
            "adenit", "gangren", "necros", "perforat", "obstruct", "ileus", "fistul",
            "deform", "impair", "calculi", "polyp", "cyst", "adenom", "sarcom", "lymphom",
            "leukem", "melanom", "dementi", "deliri", "encephalopath", "acidosis", "alkalosis",
            "shock", "sepsis", "bacteremi", "pyelonephrit", "cholecystit", "pancreatit",
            "gastrit", "colit", "enterit", "hepatit", "appendicit", "pneumon", "spasm",
            "lacerat", "contus", "ruptur", "dislocat", "subluxat", "metasta", "malignan",
            "effus", "atelecta", "pneumothor", "hemothor", "ascit", "herni", "dehydrat", "hypovolem",
            "urticar", "dermatit", "eczema", "psorias", "anaphyla", "angioedem",
            "thyroid", "goiter", "epilep", "parkinson", "alzheimer", "reflux", "esophag",
            "regurgitat", "insufficien", "prolaps", "incompeten", "lupus", "erythemat",
            "hemophil", "thalassem", "cataract", "glaucom", "retinopath", "tuberculos",
            "malari", "dengue", "typhoid", "depress", "anxiet", "bipolar", "schizophren",
            "psychos", "malformat", "anomal", "dysplas", "deviat", "hypertroph", "hyperplas",
            "atroph", "thicken", "narrow", "calcif",
        )
        has_clinical_root = any(r in term_lower for r in CLINICAL_ROOTS)

        if not (has_pathology or has_symptom or has_med_suffix or has_microbiology or has_lab_abnormality or has_clinical_root):
            # If the phrase has no clinical terms at all, reject it as a non-diagnostic statement
            return False, f"Phrase does not describe a clinical disease, disorder, pathology, or symptom: '{clean_term}'"

        # Check: Incomplete adjectival phrases lacking a clinical condition noun
        if re.search(r"^\s*(?:metastatic\s+)?castrat(?:ion|e)[- ]resistant(?:\s+(?:status|state))?\s*$", term_lower) or term_lower in (
            "metastatic castration-resistant", "castration-resistant", "metastatic sites",
            "metastatic disease sites", "sites of metastasis", "metastatic status",
            "intra-abdominal", "retroperitoneal", "pelvic", "axial", "skeletal", "osseous",
        ):
            return False, f"Adjectival modifier or attribute phrase without clinical entity: '{clean_term}'"

        return True, "Valid clinical diagnosis entity"

    @classmethod
    def is_symptom_integral_to_primary(
        cls, symptom_term: str, primary_term: str, evidence: str = ""
    ) -> bool:
        """Evaluate whether a clinical symptom is an expected integral manifestation of a primary diagnosis under CMS Guideline I.B.4.

        Principles:
        1. A symptom is ONLY integral if it is a standard physiological manifestation within the SAME organ system as the primary disease.
        2. Cross-system conditions (e.g. chest pain in skin disease, epigastric pain in urticaria, rash in heart disease) are NOT integral.
        3. If a symptom was explicitly evaluated, investigated (e.g. ECG changes, endoscopy), or diagnosed, it is not integral.
        """
        sym = symptom_term.lower().strip()
        pri = primary_term.lower().strip()

        # Check if sym is even a symptom
        symptom_indicators = [
            "pain", "colic", "fever", "cough", "dyspnea", "shortness of breath",
            "nausea", "vomiting", "dyspepsia", "indigestion", "heartburn",
            "discomfort", "wheezing", "fatigue", "malaise", "respiratory distress",
            "rash", "pruritus", "itching", "edema", "swelling", "cyanosis",
        ]
        is_sym = any(s in sym for s in symptom_indicators)
        has_definitive_dx = (
            any(sym.endswith(sfx) or f"{sfx} " in sym for sfx in ("itis", "oma", "osis", "emia", "pathy"))
            or any(dx in sym for dx in ["syndrome", "failure", "infarction", "disease", "disorder", "calculus", "lithiasis", "ulcer", "pneumonia", "dermatitis"])
        )
        if not is_sym or has_definitive_dx:
            return False

        # Organ system mapping
        # 1. Skin
        is_skin_sym = any(w in sym for w in ["rash", "pruritus", "itching", "erythema", "hive", "wheal"])
        is_skin_pri = any(w in pri for w in ["urticaria", "dermatitis", "eczema", "psoriasis", "cellulitis", "impetigo"])
        if is_skin_sym and is_skin_pri:
            return True

        # 2. Heart failure (dyspnea, shortness of breath, fluid overload, peripheral edema)
        is_hf_sym = any(w in sym for w in ["dyspnea", "shortness of breath", "edema", "peripheral edema", "fluid overload", "orthopnea"])
        is_hf_pri = any(w in pri for w in ["heart failure", "chf", "congestive heart", "systolic heart", "diastolic heart"])
        if is_hf_sym and is_hf_pri:
            return True

        # 3. Gastrointestinal & Acute Abdominal Inflammation
        is_gi_sym = any(w in sym for w in ["abdominal pain", "rlq pain", "llq pain", "ruq pain", "luq pain", "pain", "colic", "epigastric pain", "epigastric discomfort", "dyspepsia", "nausea", "vomiting", "heartburn", "indigestion", "diarrhea"])
        is_gi_pri = any(w in pri for w in ["gastritis", "appendicitis", "cholecystitis", "pancreatitis", "ulcer", "diverticulitis", "gastroenteritis", "colitis", "crohn", "ileus"])
        if is_gi_sym and is_gi_pri:
            return True
        if "fever" in sym and any(w in pri for w in ["appendicitis", "cholecystitis", "diverticulitis", "pyelonephritis", "abscess"]):
            return True

        # 4. Respiratory
        is_resp_sym = any(w in sym for w in ["cough", "wheezing", "dyspnea", "shortness of breath", "stridor", "sputum", "hemoptysis", "respiratory distress"])
        is_resp_pri = any(w in pri for w in ["pneumonia", "asthma", "copd", "bronchitis", "pneumothorax", "pulmonary edema", "respiratory failure"])
        if is_resp_sym and is_resp_pri:
            return True

        # 5. Cardiovascular
        is_cardiac_sym = any(w in sym for w in ["chest pain", "angina", "palpitation", "palpitations"])
        is_cardiac_pri = any(w in pri for w in ["infarction", "stemi", "nstemi", "angina", "coronary", "pericarditis", "myocarditis", "ischemic heart"])
        if is_cardiac_sym and is_cardiac_pri:
            return True

        # 6. Genitourinary
        is_gu_sym = any(w in sym for w in ["dysuria", "hematuria", "flank pain", "frequency", "urgency"])
        is_gu_pri = any(w in pri for w in ["pyelonephritis", "cystitis", "calculus", "lithiasis", "uti", "urinary tract infection", "nephritis"])
        if is_gu_sym and is_gu_pri:
            return True

        return False

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

        # 10. Organisms and Microbiological Findings
        if any(org in term_lower for org in ["e. coli", "escherichia coli", "klebsiella", "pseudomonas", "staphylococcus", "streptococcus", "candida", "h. pylori", "helicobacter"]):
            return ClinicalEntityType.CLINICAL_FINDING

        # 10b. Laboratory Results & Abnormal Findings
        if any(lab in term_lower for lab in ["leukocytosis", "thrombocytopenia", "elevated troponin", "elevated creatinine", "hypokalemia", "hyperkalemia", "hyponatremia", "acidosis", "alkalosis"]):
            return ClinicalEntityType.LAB_RESULT

        # 10c. Symptoms & Signs
        if any(sym in term_lower for sym in ["pain", "cough", "nausea", "vomiting", "dyspnea", "shortness of breath", "fatigue", "malaise", "colic", "dizziness", "dyspepsia", "indigestion", "heartburn", "headache", "discomfort", "orthopnea", "hematuria"]):
            return ClinicalEntityType.SYMPTOM
        if any(sign in term_lower for sign in ["tachycardia", "tachypnea", "fever", "hypotension", "wheezing", "rales", "edema", "swelling", "cyanosis", "jaundice", "pallor"]):
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
        if any(re.search(pat, term_lower) for pat in NON_CLINICAL_ACTIVITY_PATTERNS):
            return "NON_CLINICAL"
        if any(re.search(pat, term_lower) for pat in GENERIC_PROCEDURE_PATTERNS) or any(re.search(rf"\b{re.escape(proc)}\b", term_lower) for proc in PROCEDURE_TERMS) or any(term_lower.endswith(sfx) for sfx in PROCEDURE_SUFFIXES):
            return "PROCEDURE"
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
        if any(re.search(pat, term_lower) for pat in ISOLATED_ANATOMY_PATTERNS):
            return "ANATOMY"
        if any(sym in term_lower for sym in ["pain", "cough", "fever", "dyspnea", "wheezing", "discomfort"]):
            return "SYMPTOM"
        return "NON_CLINICAL"

    @classmethod
    def is_non_clinical_activity(cls, text: str) -> bool:
        t = text.strip().lower()
        return any(bool(re.search(pat, t)) for pat in NON_CLINICAL_ACTIVITY_PATTERNS)

    @classmethod
    def is_generic_procedure(cls, text: str) -> bool:
        t = text.strip().lower()
        return any(bool(re.search(pat, t)) for pat in GENERIC_PROCEDURE_PATTERNS)

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
