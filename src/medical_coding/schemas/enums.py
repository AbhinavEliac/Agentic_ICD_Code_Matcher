"""Enumerations for clinical context, ICD roles, pipeline stages, and validation states."""

from enum import StrEnum


class NegationStatus(StrEnum):
    """Negation assessment of clinical assertion."""

    AFFIRMATIVE = "AFFIRMATIVE"  # Condition is explicitly asserted as present
    NEGATED = "NEGATED"  # Condition is explicitly asserted as absent ("no history of", "denies")
    UNCERTAIN = "UNCERTAIN"  # Ambiguous assertion or pending evaluation


class Temporality(StrEnum):
    """Clinical timing and historical status of a condition."""

    CURRENT = "CURRENT"  # Active problem addressed or treated during current encounter
    HISTORICAL = "HISTORICAL"  # Past medical history, not actively evaluated or managed
    RESOLVED = "RESOLVED"  # Previously active condition now clinically resolved
    FAMILY_HISTORY = "FAMILY_HISTORY"  # Family medical background (not personal condition)
    UNKNOWN = "UNKNOWN"  # Temporality not determinable from text


class Certainty(StrEnum):
    """Diagnostic certainty according to clinical documentation."""

    CONFIRMED = "CONFIRMED"  # Definite clinical diagnosis
    SUPPORTED = "SUPPORTED"  # Strongly supported diagnosis by objective clinical facts
    SUSPECTED = "SUSPECTED"  # Probable, working, or differential diagnosis under active evaluation
    POSSIBLE = "POSSIBLE"  # Equivocal, possible, or potential condition
    UNCERTAIN = "UNCERTAIN"  # Ambiguous or questioned diagnostic status
    RULED_OUT = "RULED_OUT"  # Evaluated and definitively excluded / refuted
    NEGATED = "NEGATED"  # Explicitly negated condition
    UNKNOWN = "UNKNOWN"  # Diagnostic certainty cannot be established


class ConditionStatus(StrEnum):
    """Clinical status profile of the condition."""

    ACTIVE = "ACTIVE"  # Currently active and being treated or monitored
    HISTORICAL = "HISTORICAL"  # Past medical condition
    RESOLVED = "RESOLVED"  # Prior condition resolved or cured
    CHRONIC = "CHRONIC"  # Longstanding background condition
    ACUTE = "ACUTE"  # Acute onset condition
    UNKNOWN = "UNKNOWN"  # Status unknown or not documented


class Acuity(StrEnum):
    """Clinical acuity / chronicity profile."""

    ACUTE = "ACUTE"
    CHRONIC = "CHRONIC"
    ACUTE_ON_CHRONIC = "ACUTE_ON_CHRONIC"
    UNSPECIFIED = "UNSPECIFIED"


class Laterality(StrEnum):
    """Anatomical laterality designation."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BILATERAL = "BILATERAL"
    UNSPECIFIED = "UNSPECIFIED"


class ClinicalEntityType(StrEnum):
    """Categorization of extracted clinical entities."""

    DIAGNOSIS = "DIAGNOSIS"
    SYMPTOM = "SYMPTOM"
    SIGN = "SIGN"
    PROCEDURAL_FINDING = "PROCEDURAL_FINDING"
    UNKNOWN = "UNKNOWN"


class EvidenceType(StrEnum):
    """Clinical evidence category providing documentary authorization."""

    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    ADMISSION_REASON = "ADMISSION_REASON"
    CHIEF_COMPLAINT = "CHIEF_COMPLAINT"
    HOSPITAL_COURSE = "HOSPITAL_COURSE"
    PROCEDURE = "PROCEDURE"
    IMAGING = "IMAGING"
    LAB = "LAB"
    HISTORY = "HISTORY"
    CLINICAL_NOTE = "CLINICAL_NOTE"
    OTHER = "OTHER"


class DiagnosisRole(StrEnum):
    """ICD-10-CM coding role designation."""

    PRIMARY = "PRIMARY"  # Chief condition established after study to be responsible for admission
    SECONDARY = "SECONDARY"  # Co-existing condition actively managed or impacting stay
    HISTORICAL = "HISTORICAL"  # Past medical condition not occasioning admission
    SYMPTOM = "SYMPTOM"  # Symptom integral to or secondary to underlying diagnosis
    INCIDENTAL = "INCIDENTAL"  # Incidental finding not driving inpatient care
    RULED_OUT = "RULED_OUT"  # Evaluated and excluded
    UNCERTAIN = "UNCERTAIN"  # Questioned or ambiguous diagnosis
    EXCLUDED = (
        "EXCLUDED"  # Historical, negated, or unsubstantiated condition (omitted from billing)
    )


class PipelineStage(StrEnum):
    """Execution stages across the pipeline graph."""

    INGESTION = "INGESTION"
    EXTRACTION = "EXTRACTION"
    CONTEXT_ANALYSIS = "CONTEXT_ANALYSIS"
    CLASSIFICATION = "CLASSIFICATION"
    RETRIEVAL = "RETRIEVAL"
    RANKING = "RANKING"
    VALIDATION = "VALIDATION"
    FINALIZATION = "FINALIZATION"
    ABSTAINED = "ABSTAINED"
    FAILED = "FAILED"


class AbstentionReason(StrEnum):
    """Explicit, auditable justifications when the pipeline abstains from coding."""

    INSUFFICIENT_CLINICAL_EVIDENCE = "INSUFFICIENT_CLINICAL_EVIDENCE"
    NO_MATCHING_ICD_CANDIDATE = "NO_MATCHING_ICD_CANDIDATE"
    BELOW_CONFIDENCE_THRESHOLD = "BELOW_CONFIDENCE_THRESHOLD"
    MULTIPLE_AMBIGUOUS_PRIMARY = "MULTIPLE_AMBIGUOUS_PRIMARY"
    EXCLUDED_BY_NEGATION = "EXCLUDED_BY_NEGATION"
    EXCLUDED_BY_TEMPORALITY = "EXCLUDED_BY_TEMPORALITY"
    CONTRADICTORY_DOCUMENTATION = "CONTRADICTORY_DOCUMENTATION"
    INVALID_ICD_CODE = "INVALID_ICD_CODE"
    SPECIFICITY_REQUIRED = "SPECIFICITY_REQUIRED"
    EXCLUDES_1_VIOLATION = "EXCLUDES_1_VIOLATION"
    PROCESSING_ERROR = "PROCESSING_ERROR"


class ExecutionStatus(StrEnum):
    """High-level outcome status of document coding."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    ABSTAINED = "ABSTAINED"
    ERROR = "ERROR"
