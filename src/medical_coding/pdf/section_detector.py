"""Clinical section boundary detector identifying high-yield diagnostic documentation."""

import re
from typing import NamedTuple

from medical_coding.pdf.models import ExtractedSection


class SectionPattern(NamedTuple):
    section_type: str
    is_high_yield: bool
    pattern: re.Pattern


# Canonical clinical section mappings with regex patterns
SECTION_DEFINITIONS: list[tuple[str, bool, str]] = [
    (
        "DISCHARGE_DIAGNOSES",
        True,
        r"(?:DISCHARGE\s+DIAGNOS[EI]S|FINAL\s+DIAGNOS[EI]S|POSTOPERATIVE\s+DIAGNOS[EI]S)",
    ),
    (
        "PRINCIPAL_DIAGNOSIS",
        True,
        r"(?:PRINCIPAL\s+DIAGNOS[EI]S|PRIMARY\s+DIAGNOS[EI]S|ADMITTING\s+DIAGNOS[EI]S|ADMISSION\s+DIAGNOS[EI]S)",
    ),
    (
        "SECONDARY_DIAGNOSES",
        True,
        r"(?:SECONDARY\s+DIAGNOS[EI]S|ADDITIONAL\s+DIAGNOS[EI]S|CO-?MORBIDITIES|OTHER\s+DIAGNOS[EI]S)",
    ),
    (
        "HOSPITAL_COURSE",
        True,
        r"(?:HOSPITAL\s+COURSE|SUMMARY\s+OF\s+HOSPITAL\s+STAY|BRIEF\s+SUMMARY\s+OF\s+HOSPITAL\s+COURSE|COURSE\s+IN\s+HOSPITAL|CLINICAL\s+COURSE)",
    ),
    (
        "HISTORY_OF_PRESENT_ILLNESS",
        True,
        r"(?:HISTORY\s+OF\s+PRESENT\s+ILLNESS|HPI|CHIEF\s+COMPLAINT|REASON\s+FOR\s+ADMISSION)",
    ),
    (
        "PAST_MEDICAL_HISTORY",
        True,
        r"(?:PAST\s+MEDICAL\s+HISTORY|PMH|MEDICAL\s+HISTORY)",
    ),
    (
        "PAST_SURGICAL_HISTORY",
        False,
        r"(?:PAST\s+SURGICAL\s+HISTORY|PSH|SURGICAL\s+HISTORY)",
    ),
    (
        "PROCEDURES",
        True,
        r"(?:PROCEDURES\s+PERFORMED|OPERATIVE\s+PROCEDURES|SURGICAL\s+PROCEDURES|MAJOR\s+PROCEDURES)",
    ),
    (
        "MEDICATIONS",
        False,
        r"(?:DISCHARGE\s+MEDICATIONS|MEDICATIONS\s+ON\s+DISCHARGE|ACTIVE\s+MEDICATIONS|MEDICATIONS)",
    ),
    (
        "PHYSICAL_EXAM",
        False,
        r"(?:PHYSICAL\s+EXAMINATION|EXAMINATION\s+ON\s+ADMISSION|PHYSICAL\s+EXAM)",
    ),
    (
        "LABORATORY_DATA",
        False,
        r"(?:LABORATORY\s+DATA|PERTINENT\s+LABS|DIAGNOSTIC\s+STUDIES|LABS|IMAGING)",
    ),
    (
        "ALLERGIES",
        False,
        r"(?:ALLERGIES|ADVERSE\s+REACTIONS)",
    ),
    (
        "DISCHARGE_INSTRUCTIONS",
        False,
        r"(?:DISCHARGE\s+INSTRUCTIONS|DISPOSITION|DISCHARGE\s+CONDITION|FOLLOW-?UP)",
    ),
]


class SectionDetector:
    """Parses normalized clinical text to segment document into structured clinical sections."""

    def __init__(self) -> None:
        self._compiled_patterns: list[SectionPattern] = []
        for sec_type, is_high_yield, regex_str in SECTION_DEFINITIONS:
            # Matches header at start of line or after newlines, followed optionally by colon/dashes/newlines
            pat = re.compile(
                rf"(?:^|\n)\s*({regex_str})\s*(?::|--|\n)",
                re.IGNORECASE,
            )
            self._compiled_patterns.append(SectionPattern(sec_type, is_high_yield, pat))

    def detect_sections(
        self,
        full_text: str,
        page_offsets: list[tuple[int, int, int]] | None = None,
    ) -> list[ExtractedSection]:
        """Detect section headers and carve boundaries across full document text.

        Args:
            full_text: Normalized full clinical document string.
            page_offsets: Optional list of (page_number, start_char, end_char) tuples.

        Returns:
            List of ExtractedSection objects ordered by their position in text.
        """
        if not full_text or not full_text.strip():
            return []

        # Find all header matches across text
        raw_matches: list[dict] = []
        for sec_type, is_high_yield, pattern in self._compiled_patterns:
            for match in pattern.finditer(full_text):
                header_text = match.group(1).strip()
                raw_matches.append(
                    {
                        "section_type": sec_type,
                        "is_high_yield": is_high_yield,
                        "header_text": header_text,
                        "start_pos": match.start(1),
                        "content_start": match.end(),
                    }
                )

        if not raw_matches:
            # Fallback if no conventional headers are detected (e.g. narrative note)
            return [
                ExtractedSection(
                    section_type="GENERAL",
                    header_text="CLINICAL DOCUMENT",
                    page_number=1,
                    start_char=0,
                    end_char=len(full_text),
                    content_text=full_text.strip(),
                    is_high_yield_coding=True,
                )
            ]

        # Sort matches chronologically by their position in the text
        raw_matches.sort(key=lambda m: m["start_pos"])

        # Deduplicate overlapping matches (keep the one starting earliest)
        deduped_matches: list[dict] = []
        last_end = -1
        for m in raw_matches:
            if m["start_pos"] >= last_end:
                deduped_matches.append(m)
                last_end = m["content_start"]

        sections: list[ExtractedSection] = []

        # If there's preamble text before the first detected section, capture it
        if deduped_matches and deduped_matches[0]["start_pos"] > 0:
            preamble_text = full_text[: deduped_matches[0]["start_pos"]].strip()
            if preamble_text:
                sections.append(
                    ExtractedSection(
                        section_type="HEADER_INFO",
                        header_text="PATIENT ENCOUNTER HEADER",
                        page_number=1,
                        start_char=0,
                        end_char=deduped_matches[0]["start_pos"],
                        content_text=preamble_text,
                        is_high_yield_coding=False,
                    )
                )

        total_matches = len(deduped_matches)
        for idx, m in enumerate(deduped_matches):
            content_start = m["content_start"]
            content_end = (
                deduped_matches[idx + 1]["start_pos"] if idx + 1 < total_matches else len(full_text)
            )

            content = full_text[content_start:content_end].strip()
            page_num = self._find_page_number(m["start_pos"], page_offsets)

            sections.append(
                ExtractedSection(
                    section_type=m["section_type"],
                    header_text=m["header_text"],
                    page_number=page_num,
                    start_char=m["start_pos"],
                    end_char=content_end,
                    content_text=content,
                    is_high_yield_coding=m["is_high_yield"],
                )
            )

        return sections

    def _find_page_number(
        self,
        char_pos: int,
        page_offsets: list[tuple[int, int, int]] | None,
    ) -> int:
        """Resolve character offset to 1-based page number."""
        if not page_offsets:
            return 1
        for page_num, start, end in page_offsets:
            if start <= char_pos <= end:
                return page_num
        return page_offsets[-1][0] if page_offsets else 1
