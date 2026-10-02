"""Prompt exports."""

from medical_coding.prompts.classification import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_TEMPLATE,
)
from medical_coding.prompts.context import (
    CONTEXT_ANALYSIS_SYSTEM_PROMPT,
    CONTEXT_ANALYSIS_USER_TEMPLATE,
)
from medical_coding.prompts.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
)
from medical_coding.prompts.ranking import (
    RANKING_SYSTEM_PROMPT,
    RANKING_USER_TEMPLATE,
)

__all__ = [
    "CLASSIFICATION_SYSTEM_PROMPT",
    "CLASSIFICATION_USER_TEMPLATE",
    "CONTEXT_ANALYSIS_SYSTEM_PROMPT",
    "CONTEXT_ANALYSIS_USER_TEMPLATE",
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_USER_TEMPLATE",
    "RANKING_SYSTEM_PROMPT",
    "RANKING_USER_TEMPLATE",
]
