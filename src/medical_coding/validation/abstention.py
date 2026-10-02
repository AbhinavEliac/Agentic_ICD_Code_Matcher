"""Abstention decision engine and audit tracking."""

from medical_coding.schemas.enums import AbstentionReason, PipelineStage
from medical_coding.schemas.validation import AbstentionRecord
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class AbstentionEngine:
    """Manages criteria for abstaining from coding decisions when evidence is insufficient."""

    def __init__(self, min_confidence_threshold: float = 0.40) -> None:
        self.min_confidence_threshold = min_confidence_threshold

    def create_record(
        self,
        reason: AbstentionReason,
        detail: str,
        stage: PipelineStage,
        diagnosis_id: str | None = None,
        raw_term: str | None = None,
    ) -> AbstentionRecord:
        """Construct an auditable abstention record."""
        record = AbstentionRecord(
            diagnosis_id=diagnosis_id,
            raw_term=raw_term,
            reason=reason,
            detail=detail,
            stage=stage,
        )
        logger.info(
            "Abstention triggered at [%s] for term '%s': %s (%s)",
            stage.value,
            raw_term or "document",
            reason.value,
            detail,
        )
        return record

    def should_abstain_on_confidence(self, score: float) -> bool:
        """Return True if confidence score falls below minimum threshold."""
        return score < self.min_confidence_threshold
