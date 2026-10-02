"""Local ICD catalog access, code existence verification, and Excludes validation."""

import json
from pathlib import Path

from medical_coding.dataset.loader import BaseICDDatasetLoader
from medical_coding.schemas.icd import ICDCodeRecord, ICDDatasetStats
from medical_coding.utils.logging import get_logger
from medical_coding.utils.text import format_icd_code, unformat_icd_code

logger = get_logger(__name__)


class LocalICDCatalog:
    """In-memory authoritative catalog lookup interface for ICD-10-CM codes.

    Enforces that only codes existing in the local authoritative dataset
    can ever be considered or validated.
    """

    def __init__(self, loader: BaseICDDatasetLoader | None = None) -> None:
        self.loader = loader
        self._records_by_clean: dict[str, ICDCodeRecord] = {}
        self._records_by_formatted: dict[str, ICDCodeRecord] = {}
        self._stats: ICDDatasetStats | None = None
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def stats(self) -> ICDDatasetStats | None:
        return self._stats

    def add_record(self, record: ICDCodeRecord) -> None:
        """Register a single validated ICDCodeRecord in the catalog lookup indexes."""
        self._records_by_clean[record.unformatted_code] = record
        self._records_by_formatted[record.code] = record

    def initialize(self) -> None:
        """Load records from the dataset loader into fast lookup indexes."""
        if self._initialized:
            return

        if self.loader is not None:
            records, stats = self.loader.load()
            for rec in records.values():
                self.add_record(rec)
            self._stats = stats
            logger.info(
                "LocalICDCatalog initialized with %d codes.", len(self._records_by_formatted)
            )

        self._initialized = True

    def get_by_code(self, code: str) -> ICDCodeRecord | None:
        """Lookup a code in the local catalog by formatted or unformatted representation."""
        if not code:
            return None
        clean = unformat_icd_code(code)
        if clean in self._records_by_clean:
            return self._records_by_clean[clean]

        formatted = format_icd_code(code)
        return self._records_by_formatted.get(formatted)

    def is_valid_code(self, code: str) -> bool:
        """Check if code exists in the local authoritative dataset."""
        return self.get_by_code(code) is not None

    def is_billable_code(self, code: str) -> bool:
        """Check if code exists and is terminal/billable at full specificity."""
        record = self.get_by_code(code)
        return record.is_valid_billable if record else False

    def get_all_records(self) -> list[ICDCodeRecord]:
        """Return all distinct ICDCodeRecord entries in the catalog."""
        return list(self._records_by_formatted.values())

    def __len__(self) -> int:
        return len(self._records_by_formatted)

    def check_excludes1(self, code_a: str, code_b: str) -> bool:
        """Check if code_a and code_b violate mutual Excludes1 rules.

        Returns:
            True if there is an Excludes1 conflict between code_a and code_b.
        """
        rec_a = self.get_by_code(code_a)
        if rec_a and unformat_icd_code(code_b) in [unformat_icd_code(c) for c in rec_a.excludes1]:
            return True

        rec_b = self.get_by_code(code_b)
        if rec_b and unformat_icd_code(code_a) in [unformat_icd_code(c) for c in rec_b.excludes1]:
            return True

        return False

    def check_excludes2(self, code_a: str, code_b: str) -> bool:
        """Check if code_a and code_b are mutually noted in Excludes2 relationships."""
        rec_a = self.get_by_code(code_a)
        if rec_a and unformat_icd_code(code_b) in [unformat_icd_code(c) for c in rec_a.excludes2]:
            return True

        rec_b = self.get_by_code(code_b)
        if rec_b and unformat_icd_code(code_a) in [unformat_icd_code(c) for c in rec_b.excludes2]:
            return True

        return False

    def save_to_json(self, output_path: Path | str) -> None:
        """Persist catalog records and metadata to JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": [rec.model_dump() for rec in self._records_by_formatted.values()],
            "stats": self._stats.model_dump() if self._stats else None,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: Path | str) -> "LocalICDCatalog":
        """Load catalog from previously serialized JSON."""
        source = Path(file_path)
        with open(source, encoding="utf-8") as f:
            data = json.load(f)

        catalog = cls()
        for r_dict in data.get("records", []):
            rec = ICDCodeRecord.model_validate(r_dict)
            catalog.add_record(rec)

        if data.get("stats"):
            catalog._stats = ICDDatasetStats.model_validate(data["stats"])

        catalog._initialized = True
        return catalog
