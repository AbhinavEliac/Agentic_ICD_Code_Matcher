"""Text processing, span extraction, and ICD string normalization utilities."""

import re


def normalize_whitespace(text: str) -> str:
    """Collapse contiguous whitespace sequences into single spaces."""
    return re.sub(r"[ \t]+", " ", text).strip()


def find_verbatim_span(full_text: str, quote: str) -> tuple[int, int] | None:
    """Locate the exact character span [start, end) of quote in full_text.

    Returns:
        (start_char, end_char) tuple if found, otherwise None.
    """
    if not quote or not full_text:
        return None
    start = full_text.find(quote)
    if start != -1:
        return (start, start + len(quote))

    # Case-insensitive fallback
    lower_full = full_text.lower()
    lower_quote = quote.lower()
    start = lower_full.find(lower_quote)
    if start != -1:
        return (start, start + len(quote))

    return None


def format_icd_code(raw_code: str) -> str:
    """Format alphanumeric ICD code with standard decimal (e.g. 'I5021' -> 'I50.21').

    Standard ICD-10-CM codes place a decimal point after the first 3 characters.
    """
    clean = re.sub(r"[^A-Za-z0-9]", "", raw_code).upper()
    if len(clean) > 3 and "." not in clean:
        return f"{clean[:3]}.{clean[3:]}"
    return clean


def unformat_icd_code(formatted_code: str) -> str:
    """Strip punctuation and whitespace from an ICD code (e.g. 'I50.21' -> 'I5021')."""
    return re.sub(r"[^A-Za-z0-9]", "", formatted_code).upper()


def is_valid_icd_format(code: str) -> bool:
    """Check whether a code adheres to the standard ICD-10-CM structural pattern.

    ICD-10-CM codes start with a letter, followed by a digit and 1 to 5 alphanumeric chars
    (with an optional decimal point after the 3rd character).
    """
    if not code:
        return False
    clean = unformat_icd_code(code)
    if len(clean) < 3 or len(clean) > 7:
        return False
    # First char must be a letter, second must be a digit, followed by alphanumeric
    return bool(re.match(r"^[A-Z][0-9][0-9A-Z]{1,5}$", clean))
