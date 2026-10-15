"""Text processing, span extraction, and ICD string normalization utilities."""

import re
from typing import Any


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
    """Format alphanumeric clinical code (ICD-10-CM, ICD-O, or CPT) cleanly.

    - Standard ICD-10-CM codes (e.g. 'I5021' -> 'I50.21') place a decimal point after character 3.
    - CPT codes (e.g. '1404', '99213') remain pure numeric without artificial decimal points.
    - ICD-O morphology codes (e.g. 'M8000.0') preserve their oncology morphology structure.
    - Already formatted codes (e.g. 'I50.21', 'M8000.0') are preserved.
    """
    if not raw_code:
        return ""
    raw = str(raw_code).strip()
    if raw.isdigit():
        return raw
    if re.match(r"^[0-9]{4}[A-Za-z]$", raw):
        return raw.upper()
    if re.match(r"^M[89][0-9]{3}[\./][0-9]$", raw, re.IGNORECASE):
        return raw.upper()
    if "." in raw:
        return raw.upper()
    clean = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    if len(clean) > 3 and clean[0].isalpha() and clean[1].isdigit():
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


CLINICAL_PREAMBLE_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:\d+[\.\)]|\*|\-|\u2022|\([a-z0-9]\))\s*)*"
    r"(?:"
    r"known\s+case\s+of|k/?c/?o\b|kco\b|"
    r"history\s+of|h/?o\b|p/?h/?o\b|past\s+medical\s+history\s+of|past\s+history\s+of|"
    r"status\s+post|s/?p\b|status-post|"
    r"c/?o\b|complaining\s+of|complaints\s+of|chief\s+complaint\s+of|"
    r"admitted\s+(?:for|with|due\s+to)|presented\s+with|presents\s+with|"
    r"diagnosed\s+with|diagnosis\s*:|diagnoses\s*:|impression\s*:|assessment\s*:|"
    r"discharge\s+diagnosis\s*:|final\s+diagnosis\s*:|principal\s+diagnosis\s*:|secondary\s+diagnosis\s*:|"
    r"features\s+(?:suggestive\s+of|of)|findings\s+(?:suggestive\s+of|compatible\s+with|of)|"
    r"evidence\s+of|signs\s+of|s/?o\b|suggestive\s+of|"
    r"likely|probable|possibly|suspected|rule\s+out|r/?o\b|\?|presumed|underlying"
    r")\s*[:\-–—]?\s*",
    re.IGNORECASE,
)

MEDICATION_ATTACHMENT_PATTERN = re.compile(
    r"\s*[-–—:,;]?\s*(?:"
    r"on\s+(?:tab|tablets?|cap|capsules?|inj|injections?|syp|syrups?|medication|rx|treatment|dialysis|insulin|metformin|aspirin|statin|steroids?|antibiotics?)|"
    r"taking\s+|treated\s+with|started\s+on|prescribed|dosage"
    r").*$",
    re.IGNORECASE,
)

QUALIFIERS_TO_STRIP_FOR_CORE = {
    "mild", "moderate", "severe", "stage", "1", "2", "3", "4", "5", "i", "ii", "iii", "iv",
    "compensated", "decompensated", "routine", "unspecified", "uncontrolled", "controlled",
    "status", "post", "acute", "chronic", "subacute", "left", "right", "bilateral",
}


def normalize_diagnosis_name(text: str) -> str:
    """Strengthened clinical diagnosis normalizer.

    1. Strips leading list bullets, indices, and whitespace.
    2. Strips medical preambles (K/C/O, H/O, S/P, C/O, impression of, etc.).
    3. Strips trailing prescription / medication attachments.
    4. Cleans and integrates useful parentheticals (e.g. 'Essential (primary) hypertension').
    5. Expands common clinical abbreviations and acronyms.
    6. Normalizes whitespace and casing.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Step 1: Strip leading bullets, numbering, dashes
    cleaned = re.sub(r"^\s*(?:\d+[\.\)]|\*|\-|\u2022|\([a-z0-9]\))\s*", "", cleaned)

    # Step 2: Strip clinical preambles iteratively (e.g. 'K/C/O H/O HTN')
    for _ in range(3):
        match = CLINICAL_PREAMBLE_PATTERN.match(cleaned)
        if match:
            cleaned = cleaned[match.end():].strip()
        else:
            break

    # Step 3: Strip trailing medication attachments
    cleaned = MEDICATION_ATTACHMENT_PATTERN.sub("", cleaned).strip()

    # Step 4: Handle parentheticals
    def _clean_paren(m: re.Match[str]) -> str:
        inside = m.group(1).strip()
        # Keep clinical entity words like primary, essential, acute, chronic, bilateral, type 1, type 2
        if any(w in inside.lower() for w in ["primary", "essential", "acute", "chronic", "type", "bilateral", "left", "right", "congestive", "systolic", "diastolic", "organism"]):
            return f" {inside} "
        # Strip notes like dates, non-informative parentheticals
        return " "

    cleaned = re.sub(r"\(([^)]+)\)", _clean_paren, cleaned)

    # Step 5: Expand clinical abbreviations
    from medical_coding.retrieval.tokenizer import expand_medical_abbreviations
    cleaned = expand_medical_abbreviations(cleaned)

    # Step 6: Normalize whitespace and trailing punctuation
    cleaned = re.sub(r"[,\-–—:;]+$", "", cleaned).strip()
    cleaned = normalize_whitespace(cleaned)

    return cleaned


def extract_search_query_variations(raw_term: str, evidence: str = "") -> list[str]:
    """Generate prioritized query variations for robust multi-round BM25 and vector search.

    Produces:
    1. Fully normalized diagnosis name.
    2. Standard clinical query with expanded abbreviations.
    3. Core diagnostic entity (stripped of peripheral severity/acuity/laterality qualifiers).
    4. Clinical and morphological synonyms (e.g. primary hypertension -> essential hypertension).
    5. Evidence-anchored diagnostic entities (if evidence quote provided).
    """
    if not raw_term:
        return []

    variations: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        clean = normalize_whitespace(q.strip().lower())
        if clean and clean not in seen and len(clean) >= 2:
            variations.append(clean)
            seen.add(clean)

    # 1. Normalized primary term
    norm_term = normalize_diagnosis_name(raw_term)
    _add(norm_term)
    _add(raw_term)

    # 2. Etiological & Primary Name Specific Variations
    # For diagnoses carrying "Primary", ensure essential/idiopathic forms are checked
    norm_lower = norm_term.lower()
    if "primary hypertension" in norm_lower:
        _add("essential primary hypertension")
        _add("essential hypertension")
        _add("hypertension")
    elif "essential hypertension" in norm_lower:
        _add("essential primary hypertension")
        _add("primary hypertension")
        _add("hypertension")
    elif "primary osteoarthritis" in norm_lower:
        _add("osteoarthritis")
        _add("degenerative joint disease")
        _add("polyosteoarthritis")
    elif "primary biliary" in norm_lower:
        _add("primary biliary cholangitis")
        _add("primary biliary cirrhosis")
        _add("biliary cholangitis")
    elif "primary open-angle" in norm_lower or "primary open angle" in norm_lower:
        _add("primary open angle glaucoma")
        _add("open angle glaucoma")
    elif "primary pulmonary hypertension" in norm_lower:
        _add("idiopathic pulmonary arterial hypertension")
        _add("pulmonary arterial hypertension")
    elif "primary spontaneous pneumothorax" in norm_lower:
        _add("spontaneous pneumothorax")
        _add("pneumothorax")

    # 3. Core diagnostic entity tokens (strip qualifiers like mild, acute, left, etc.)
    words = re.findall(r"[a-z0-9]+", norm_lower)
    core_words = [w for w in words if w not in QUALIFIERS_TO_STRIP_FOR_CORE]
    if core_words and len(core_words) != len(words):
        _add(" ".join(core_words))

    # 4. Morphological synonyms
    from medical_coding.retrieval.tokenizer import CLINICAL_MORPHOLOGY
    syn_words: list[str] = []
    has_syn = False
    for w in (core_words or words):
        if w in CLINICAL_MORPHOLOGY:
            has_syn = True
            syn_words.append(CLINICAL_MORPHOLOGY[w][0])
        else:
            syn_words.append(w)
    if has_syn:
        _add(" ".join(syn_words))

    # 5. Evidence-anchored keywords
    if evidence:
        ev_clean = normalize_diagnosis_name(evidence)
        ev_words = re.findall(r"[a-z0-9]{4,}", ev_clean.lower())
        diag_words_set = set(words)
        matching_ev_words = [w for w in ev_words if w in diag_words_set or any(v in diag_words_set for v in CLINICAL_MORPHOLOGY.get(w, []))]
        if matching_ev_words and len(matching_ev_words) >= 2:
            _add(" ".join(matching_ev_words[:4]))

    return variations


def deduplicate_evidence_spans(
    evidence_items: list[Any] | str | None,
    diagnosis_name: str | None = None,
) -> list[str]:
    """Clean and deduplicate evidence strings, removing identical repeats,
    subsumed substrings, bare diagnosis label headers, and audit boilerplate.

    Preserves unique clinical evidence statements in chronological order.
    """
    if not evidence_items:
        return []

    # 1. Flatten all items into raw candidate text chunks
    raw_chunks: list[str] = []
    if isinstance(evidence_items, str):
        for part in re.split(r"\s*\|\s*|\n+", evidence_items):
            if part.strip():
                raw_chunks.append(part.strip())
    elif isinstance(evidence_items, list):
        for item in evidence_items:
            q = ""
            if isinstance(item, dict):
                q = str(item.get("quote") or item.get("text") or item.get("sentence") or "")
            elif hasattr(item, "quote"):
                q = str(getattr(item, "quote") or "")
            elif hasattr(item, "text"):
                q = str(getattr(item, "text") or "")
            elif hasattr(item, "sentence"):
                q = str(getattr(item, "sentence") or "")
            elif isinstance(item, str):
                q = item
            if q.strip():
                for part in re.split(r"\s*\|\s*|\n+", q):
                    if part.strip():
                        raw_chunks.append(part.strip())
    else:
        q_str = str(evidence_items).strip()
        if q_str:
            raw_chunks.append(q_str)

    # 2. Clean each chunk of audit boilerplate and wrappers
    cleaned_chunks: list[str] = []
    for chunk in raw_chunks:
        c = chunk.strip()
        if not c or c.lower() in ("none", "null", "undefined", "[no local code]"):
            continue
        # Strip internal audit annotations
        if "Designated as " in c:
            c = c.split("Designated as ")[0].strip(" -|:\n")
        if "Clinical criteria:" in c:
            c = c.split("Clinical criteria:")[0].strip(" -|:\n")
        if "Confidence:" in c:
            c = c.split("Confidence:")[0].strip(" -|:\n")
        # Strip quotes and bullet markers
        c = c.strip(" \"'“”`•–-*#\t\r\n")
        c = c.rstrip(":")
        if c and c.lower() not in ("none", "null"):
            cleaned_chunks.append(c)

    if not cleaned_chunks:
        return []

    # 3. Normalized comparison helper
    def _norm(s: str) -> str:
        s_low = s.lower().strip()
        return re.sub(r"[^\w\s]", "", s_low).strip()

    diag_norm = _norm(diagnosis_name) if diagnosis_name else ""

    # 4. Remove bare diagnosis label if more detailed narrative sentences exist
    has_descriptive_evidence = any(
        len(_norm(c).split()) >= 4 and _norm(c) != diag_norm for c in cleaned_chunks
    )

    filtered_chunks: list[str] = []
    for c in cleaned_chunks:
        c_norm = _norm(c)
        if not c_norm:
            continue
        if diag_norm and c_norm == diag_norm and has_descriptive_evidence:
            # Skip bare header match when descriptive narrative evidence is present
            continue
        filtered_chunks.append(c)

    if not filtered_chunks:
        filtered_chunks = cleaned_chunks

    # 5. Deduplicate exact and subsumed chunks (keeping longer/more informative spans)
    unique_spans: list[str] = []
    for cand in filtered_chunks:
        cand_norm = _norm(cand)
        if not cand_norm:
            continue

        is_subsumed = False
        replace_index = -1

        for i, existing in enumerate(unique_spans):
            ex_norm = _norm(existing)
            # Exact duplicate
            if cand_norm == ex_norm:
                is_subsumed = True
                break
            # cand is substring of existing
            if cand_norm in ex_norm and len(cand_norm) < len(ex_norm):
                is_subsumed = True
                break
            # existing is substring of cand -> cand is more descriptive
            if ex_norm in cand_norm and len(ex_norm) < len(cand_norm):
                replace_index = i
                break

        if replace_index >= 0:
            unique_spans[replace_index] = cand
        elif not is_subsumed:
            unique_spans.append(cand)

    return unique_spans


def clean_and_deduplicate_evidence_quote(
    evidence_text: Any,
    diagnosis_name: str | None = None,
) -> str:
    """Return a single cleaned, non-repetitive evidence string joined by ' | '."""
    if not evidence_text:
        return "None"
    spans = deduplicate_evidence_spans(evidence_text, diagnosis_name=diagnosis_name)
    return " | ".join(spans) if spans else "None"


def clean_and_deduplicate_evidence_list(
    evidence_list: list[Any] | str | None,
    diagnosis_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return a list of deduplicated evidence dictionaries."""
    spans = deduplicate_evidence_spans(evidence_list, diagnosis_name=diagnosis_name)
    return [
        {
            "quote": s,
            "section": "CLINICAL_DOCUMENTATION",
            "evidence_type": "CLINICAL_EVIDENCE",
        }
        for s in spans
    ]


