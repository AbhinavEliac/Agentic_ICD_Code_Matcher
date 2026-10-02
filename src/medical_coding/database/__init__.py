"""Database persistence layer for medical coding encounters and ICD-10 audit trails."""

from medical_coding.database.connection import get_db_session, get_engine, init_db
from medical_coding.database.models import (
    AbstentionRecordModel,
    Base,
    DiagnosisRecord,
    ProcessedDocument,
    ValidationCheckModel,
)
from medical_coding.database.repository import MedicalCodingRepository

__all__ = [
    "AbstentionRecordModel",
    "Base",
    "DiagnosisRecord",
    "MedicalCodingRepository",
    "ProcessedDocument",
    "ValidationCheckModel",
    "get_db_session",
    "get_engine",
    "init_db",
]
