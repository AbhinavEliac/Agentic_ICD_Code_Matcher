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
    "pud": "peptic ulcer disease",
    "sob": "shortness of breath dyspnea",
    "bph": "benign prostatic hyperplasia",
    "osa": "obstructive sleep apnea",
    "ra": "rheumatoid arthritis",
    "oa": "osteoarthritis",
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
        tokens = [t for t in raw_tokens if t not in GENERAL_STOPWORDS and len(t) > 1]
    else:
        tokens = [t for t in raw_tokens if len(t) > 0]

    return tokens


def normalize_clinical_query(text: str) -> str:
    """Prepare a normalized string query for retrieval with abbreviations expanded."""
    expanded = expand_medical_abbreviations(text.strip())
    # Clean redundant spaces
    return re.sub(r"\s+", " ", expanded)
