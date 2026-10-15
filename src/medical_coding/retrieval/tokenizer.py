"""Clinical text tokenization, normalization, and medical abbreviation expansion."""

import re

from medical_coding.utils.clinical_lexicon import (
    CLINICAL_ABBREVIATIONS,
    CLINICAL_MORPHOLOGY,
    GENERAL_STOPWORDS,
)


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

