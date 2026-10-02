"""Clinical document text normalizer preserving semantic structure and formatting."""

import re


class DocumentNormalizer:
    """Normalizes raw extracted PDF text while strictly preserving clinical semantics."""

    def __init__(self, dehyphenate: bool = True) -> None:
        self.dehyphenate = dehyphenate

    def normalize(self, text: str) -> str:
        """Clean extracted text preserving paragraph breaks, list markers, and clinical punctuation.

        Rules:
        1. Replace non-breaking spaces and special unicode dashes/quotes with ASCII equivalents.
        2. Remove non-printable control characters without removing newlines and tabs.
        3. De-hyphenate words broken across linebreaks (e.g. 'hyper-\\ntension' -> 'hypertension').
        4. Normalize horizontal spaces and tabs without destroying vertical line breaks.
        5. Preserve double-newline paragraph and section separations.
        6. Strip trailing spaces per line.
        """
        if not text:
            return ""

        cleaned = text

        # 1. Normalize unicode spaces and quotes
        cleaned = cleaned.replace("\u00a0", " ")
        cleaned = cleaned.replace("\u200b", "")  # Zero-width space
        cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
        cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")

        # 2. Remove non-printable control characters (keep \n and \t)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

        # 3. De-hyphenate soft linebreaks where a word was split at end-of-line
        # Matches e.g. "myo- \n cardial" -> "myocardial"
        if self.dehyphenate:
            cleaned = re.sub(r"([a-zA-Z]{3,})-\s*\n\s*([a-zA-Z]{3,})", r"\1\2", cleaned)

        # 4. Normalize horizontal spaces per line (collapse multiple spaces/tabs into a single space)
        lines = cleaned.splitlines()
        normalized_lines = []
        for line in lines:
            # Collapse horizontal spaces
            norm_line = re.sub(r"[ \t]+", " ", line).strip()
            normalized_lines.append(norm_line)

        # Re-join with single newlines
        rejoined = "\n".join(normalized_lines)

        # 5. Collapse excessive vertical blank lines: max 2 newlines (one blank line between paragraphs)
        collapsed = re.sub(r"\n{3,}", "\n\n", rejoined)

        return collapsed.strip()
