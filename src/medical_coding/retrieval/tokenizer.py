"""Clinical text tokenization, normalization, and medical abbreviation expansion."""

import re

# Standard clinical abbreviations mapped to expanded medical concepts
CLINICAL_ABBREVIATIONS: dict[str, str] = {
    "hf": "heart failure",
    "chf": "congestive heart failure",
    "hfref": "heart failure reduced ejection fraction systolic",
    "hfpef": "heart failure preserved ejection fraction diastolic",
    "dm": "diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "htn": "hypertension",
    "ckd": "chronic kidney disease",
    "esrd": "end stage renal disease",
    "copd": "chronic obstructive pulmonary disease",
    "afib": "atrial fibrillation",
    "af": "atrial fibrillation",
    "cad": "coronary artery disease",
    "mi": "myocardial infarction",
    "ami": "acute myocardial infarction",
    "stemi": "st elevation myocardial infarction",
    "nstemi": "non st elevation myocardial infarction",
    "aki": "acute kidney injury",
    "arf": "acute renal failure",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "gerd": "gastroesophageal reflux disease",
    "cva": "cerebrovascular accident stroke",
    "tia": "transient ischemic attack",
    "uti": "urinary tract infection",
    "pna": "pneumonia",
    "cap": "pneumonia",
    "pud": "peptic ulcer disease",
    "sob": "shortness of breath dyspnea",
    "bph": "benign prostatic hyperplasia",
    "osa": "obstructive sleep apnea",
    "ra": "rheumatoid arthritis",
    "oa": "osteoarthritis",
    "dlbcl": "diffuse large b cell lymphoma",
    "dka": "diabetic ketoacidosis",
    "hiv": "human immunodeficiency virus",
    "hbv": "hepatitis b virus viral hepatitis b",
    "hcv": "hepatitis c virus viral hepatitis c",
    "hav": "hepatitis a virus viral hepatitis a",
    "ks": "kaposi sarcoma",
    "acl": "anterior cruciate ligament",
    "pcl": "posterior cruciate ligament",
    "net": "neuroendocrine tumor carcinoid",
    "fet": "frozen embryo transfer",
}

# Non-informative general words (keeping critical medical distinctions: acute, chronic, type, 1, 2, without, etc.)
GENERAL_STOPWORDS: set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def expand_medical_abbreviations(text: str) -> str:
    """Expand recognized clinical acronyms into full diagnostic terms.

    Example:
        'acute systolic HF' -> 'acute systolic heart failure'
    """
    if not text:
        return ""

    tokens = re.findall(r"\b[A-Za-z0-9_]+\b", text)
    expanded: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in CLINICAL_ABBREVIATIONS:
            expanded.append(CLINICAL_ABBREVIATIONS[lower])
        else:
            expanded.append(token)

    return " ".join(expanded)


def tokenize_clinical_text(
    text: str,
    expand_abbreviations: bool = True,
    remove_stopwords: bool = True,
) -> list[str]:
    """Tokenize clinical text into normalized lowercase tokens for BM25 lexical search.

    Args:
        text: Clinical text or diagnosis description.
        expand_abbreviations: Whether to expand medical abbreviations.
        remove_stopwords: Whether to filter general non-informative stopwords.

    Returns:
        List of cleaned token strings.
    """
    if not text:
        return []

    processed = text
    if expand_abbreviations:
        processed = expand_medical_abbreviations(processed)

    # Extract alphanumeric words and decimal codes
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?", processed.lower())

    if remove_stopwords:
        tokens = [
            t
            for t in raw_tokens
            if t not in GENERAL_STOPWORDS and (len(t) > 1 or t.isdigit() or t in {"b", "c", "d", "k", "t"})
        ]
    else:
        tokens = [t for t in raw_tokens if len(t) > 0]

    return tokens


def normalize_clinical_query(text: str) -> str:
    """Prepare a normalized string query for retrieval with abbreviations expanded."""
    expanded = expand_medical_abbreviations(text.strip())
    # Clean redundant spaces
    return re.sub(r"\s+", " ", expanded)


CLINICAL_MORPHOLOGY: dict[str, list[str]] = {
    "ureter": ["ureteric", "ureteral"],
    "ureteric": ["ureter", "ureteral"],
    "ureteral": ["ureter", "ureteric"],
    "calculus": ["stone", "calculi", "stones", "cholelithiasis", "gallstone", "gallstones", "lithiasis", "nephrolithiasis"],
    "calculi": ["calculus", "stone", "stones", "cholelithiasis", "gallstone", "gallstones"],
    "stone": ["calculus", "calculi", "stones", "cholelithiasis", "gallstone", "gallstones"],
    "stones": ["calculus", "calculi", "stone", "cholelithiasis", "gallstone", "gallstones"],
    "renal": ["kidney", "renal"],
    "kidney": ["renal", "kidney"],
    "gastric": ["stomach", "gastric"],
    "stomach": ["gastric", "stomach"],
    "cardiac": ["heart", "cardiac"],
    "heart": ["cardiac", "heart"],
    "pulmonary": ["lung", "pulmonary"],
    "lung": ["pulmonary", "lung"],
    "hepatic": ["liver", "hepatic"],
    "liver": ["hepatic", "liver"],
    "colonic": ["colon", "colonic"],
    "colon": ["colonic", "colon"],
    "appendiceal": ["appendix", "appendiceal"],
    "appendix": ["appendiceal", "appendix"],
    "splenic": ["spleen", "splenic"],
    "spleen": ["splenic", "spleen"],
    "cerebral": ["brain", "cerebral"],
    "brain": ["cerebral", "brain"],
    "ocular": ["eye", "ocular"],
    "eye": ["ocular", "eye"],
    "pneumonia": ["pneumonitis", "pna"],
    "hypertension": ["htn", "high blood pressure"],
    "diabetes": ["dm", "diabetic"],
    "diabetic": ["diabetes", "dm"],
    "colic": ["colicky"],
    "chf": ["heart failure", "congestive heart failure"],
    "pain": ["discomfort", "ache"],
    "discomfort": ["pain", "ache"],
    "ache": ["pain", "discomfort"],
    "community": ["pneumonia"],
    "dlbcl": ["lymphoma", "diffuse", "large"],
    "lymphoma": ["dlbcl"],
    "bronchitis": ["bronchial"],
    "ckd": ["chronic", "kidney", "disease"],
    "candida": ["candidiasis", "candidal"],
    "candidiasis": ["candida", "candidal"],
    "candidal": ["candida", "candidiasis"],
    "sepsis": ["septic", "septicemia"],
    "septic": ["sepsis", "septicemia"],
    "septicemia": ["sepsis", "septic"],
    "urinary": ["urogenital"],
    "urogenital": ["urinary", "genital"],
    "urosepsis": ["urinary", "sepsis"],
    "hbv": ["hepatitis", "viral"],
    "hepatitis": ["hbv", "hcv", "viral"],
    "sarcoma": ["kaposi", "sarcomas", "malignant", "neoplasm", "cancer"],
    "viral": ["virus", "hepatitis", "hiv"],
    "virus": ["viral", "hepatitis", "hiv"],
    "cancer": ["malignant", "neoplasm", "carcinoma"],
    "carcinoma": ["cancer", "malignant", "neoplasm"],
    "malignant": ["cancer", "carcinoma", "neoplasm", "metastasis", "metastatic", "metastases", "malignancy"],
    "neoplasm": ["cancer", "carcinoma", "malignant", "metastasis", "metastatic", "metastases", "tumor", "tumour"],
    "metastasis": ["metastatic", "metastases", "secondary", "malignant", "neoplasm"],
    "metastases": ["metastasis", "metastatic", "secondary", "malignant", "neoplasm"],
    "metastatic": ["metastasis", "metastases", "secondary", "malignant", "neoplasm"],
    "secondary": ["metastasis", "metastases", "metastatic", "secondaries"],
    "carcinoid": ["neuroendocrine", "net"],
    "neuroendocrine": ["carcinoid", "net", "endocrine", "tumor", "neoplasm"],
    "male": ["man", "men", "boy", "gentleman"],
    "subsequent": ["follow-up", "followup", "subsequent", "routine healing"],
    "pleural": ["pleura"],
    "pleura": ["pleural"],
    "malleolus": ["malleolar", "fibula", "fibular"],
    "malleolar": ["malleolus", "fibula", "fibular"],
    "fibula": ["malleolus", "malleolar", "fibular"],
    "fibular": ["malleolus", "malleolar", "fibula"],
    "tarsal": ["tarsus", "tarsals", "foot"],
    "tarsals": ["tarsal", "foot"],
    "metatarsal": ["metatarsals", "foot"],
    "metatarsals": ["metatarsal", "foot"],
    "foot": ["tarsal", "tarsals", "metatarsal", "metatarsals", "ankle"],
    "ankle": ["deltoid", "calcaneofibular", "malleolus", "malleolar"],
    "ligament": ["deltoid", "calcaneofibular", "sprain", "tear"],
    "sprain": ["sprains", "injury", "ligament", "tear", "rupture"],
    "sprains": ["sprain", "injury", "ligament", "tear", "rupture"],
    "tear": ["tears", "rupture", "sprain", "sprains", "disruption", "injury"],
    "tears": ["tear", "rupture", "sprain", "sprains", "disruption"],
    "cruciate": ["acl", "pcl", "ligament"],
    "sarcomas": ["sarcoma", "malignant", "neoplasm", "cancer"],
    "pyelonephritis": ["pyelonephritic", "nephritis", "renal", "kidney"],
    "nondisplaced": ["non-displaced"],
    "displaced": ["displacement"],
    "fracture": ["fractures"],
    "fractures": ["fracture"],
    "cholecystitis": ["gallbladder"],
    "cholelithiasis": ["calculus", "gallbladder", "gallstone", "gallstones", "calculi"],
    "gallstones": ["cholelithiasis", "calculus", "gallbladder", "gallstone", "stone"],
    "gallstone": ["cholelithiasis", "calculus", "gallbladder", "gallstones", "stone"],
    "gallbladder": ["cholecystitis", "cholelithiasis", "biliary"],
    "myocardial": ["infarction", "nstemi", "stemi", "heart"],
    "infarction": ["myocardial", "nstemi", "stemi"],
    "nstemi": ["subendocardial", "myocardial", "infarction"],
    "stemi": ["myocardial", "infarction"],
    "subendocardial": ["nstemi", "infarction", "myocardial"],
    "hyperlipidemia": ["hypercholesterolemia", "dyslipidemia", "lipids"],
    "dyslipidemia": ["hyperlipidemia", "hypercholesterolemia", "lipids"],
    "hypercholesterolemia": ["hyperlipidemia", "dyslipidemia", "cholesterol"],
    "hypokalemia": ["potassium", "electrolyte"],
    "hyperkalemia": ["potassium", "electrolyte"],
    "hyponatremia": ["sodium", "electrolyte"],
    "potassium": ["hypokalemia", "hyperkalemia"],
    "sodium": ["hyponatremia"],
    "anemia": ["iron", "deficiency", "blood"],
    "appendicitis": ["appendix"],
    "diverticulitis": ["diverticular", "colon"],
    "pancreatitis": ["pancreas", "pancreatic"],
    "pancreas": ["pancreatitis", "pancreatic"],
    "pancreatic": ["pancreas", "pancreatitis"],
    "fibrillation": ["atrial", "afib"],
    "atherosclerosis": ["coronary", "artery", "disease", "cad"],
    "hypothyroidism": ["thyroid"],
    "thyroid": ["hypothyroidism"],
}


def expand_clinical_morphology(tokens: list[str]) -> list[str]:
    """Expand tokens with morphological clinical equivalents and singular forms."""
    result: list[str] = list(tokens)
    seen = set(tokens)
    for t in tokens:
        lower = t.lower()
        if lower in CLINICAL_MORPHOLOGY:
            for syn in CLINICAL_MORPHOLOGY[lower]:
                for word in syn.split():
                    if word not in seen:
                        result.append(word)
                        seen.add(word)
        # De-pluralize simple medical plurals if > 4 chars
        if lower.endswith("s") and len(lower) > 4 and not lower.endswith("ss"):
            singular = lower[:-1]
            if singular not in seen:
                result.append(singular)
                seen.add(singular)
    return result


def get_expanded_query_tokens(text: str) -> list[str]:
    """Tokenize and expand clinical query terms for high-recall lexical retrieval."""
    tokens = tokenize_clinical_text(text, expand_abbreviations=True, remove_stopwords=True)
    return expand_clinical_morphology(tokens)

