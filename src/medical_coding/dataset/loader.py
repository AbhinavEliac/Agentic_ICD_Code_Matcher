"""Local ICD-10-CM dataset ingestion, validation, normalization, and deduplication."""

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from medical_coding.schemas.icd import ICDCodeRecord, ICDDatasetStats
from medical_coding.utils.logging import get_logger
from medical_coding.utils.text import format_icd_code, is_valid_icd_format, unformat_icd_code

logger = get_logger(__name__)


class BaseICDDatasetLoader(ABC):
    """Abstract interface for reading local ICD-10-CM source files into code records."""

    @abstractmethod
    def load(self) -> tuple[dict[str, ICDCodeRecord], ICDDatasetStats]:
        """Load, validate, normalize, and parse ICD-10-CM records from local storage.

        Returns:
            Tuple of (dictionary mapping formatted codes to ICDCodeRecord, dataset statistics).
        """
        pass

    @abstractmethod
    def get_total_records(self) -> int:
        """Return the count of valid codes in the loaded dataset."""
        pass

    @abstractmethod
    def get_stats(self) -> ICDDatasetStats | None:
        """Return the statistics of the ingestion process."""
        pass


class TabularICDLoader(BaseICDDatasetLoader):
    """Parses tabular ICD-10-CM datasets from CSV, TSV, or JSON files.

    Performs strict column validation, code format normalization, malformed row
    filtering, and duplicate detection.
    """

    # Acceptable column aliases for flexible hospital schema ingestion
    CODE_ALIASES = {"code", "icd_code", "icd10_code", "icd10", "code_id", "icd_10_code"}
    DESC_ALIASES = {
        "description",
        "long_description",
        "short_description",
        "clinical_description",
        "title",
        "term",
        "diagnosis",
    }

    def __init__(
        self,
        file_path: Path | str,
        delimiter: str | None = None,
        source_name: str | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.delimiter = delimiter
        self.source_name = source_name or self.file_path.name
        self._records: dict[str, ICDCodeRecord] = {}
        self._stats: ICDDatasetStats | None = None
        self._is_loaded: bool = False

    def get_total_records(self) -> int:
        return len(self._records)

    def get_stats(self) -> ICDDatasetStats | None:
        return self._stats

    def _detect_delimiter(self) -> str:
        if self.delimiter:
            return self.delimiter
        suffix = self.file_path.suffix.lower()
        if suffix == ".tsv":
            return "\t"
        return ","

    def _parse_list_field(self, raw_val: Any) -> list[str]:
        if not raw_val:
            return []
        if isinstance(raw_val, list):
            return [str(v).strip() for v in raw_val if str(v).strip()]
        if isinstance(raw_val, str):
            clean = raw_val.strip()
            if not clean:
                return []
            if clean.startswith("[") and clean.endswith("]"):
                try:
                    parsed = json.loads(clean)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
            # Split by semicolon or pipe or comma
            if ";" in clean:
                return [v.strip() for v in clean.split(";") if v.strip()]
            if "|" in clean:
                return [v.strip() for v in clean.split("|") if v.strip()]
            return [clean]
        return [str(raw_val)]

    def _parse_bool_field(self, raw_val: Any, default: bool = True) -> bool:
        if raw_val is None or raw_val == "":
            return default
        if isinstance(raw_val, bool):
            return raw_val
        val_str = str(raw_val).strip().lower()
        if val_str in {"1", "true", "yes", "t", "y", "billable"}:
            return True
        if val_str in {"0", "false", "no", "f", "n", "header", "non-billable"}:
            return False
        return default

    def load(self) -> tuple[dict[str, ICDCodeRecord], ICDDatasetStats]:
        """Load and normalize tabular records, dropping malformed entries and duplicates."""
        if self._is_loaded and self._stats is not None:
            return self._records, self._stats

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"ICD dataset file not found at expected location: {self.file_path}"
            )

        suffix = self.file_path.suffix.lower()
        raw_rows: list[dict[str, Any]] = []

        if suffix == ".json":
            with open(self.file_path, encoding="utf-8-sig") as f:
                content = json.load(f)
                if isinstance(content, list):
                    raw_rows = content
                elif isinstance(content, dict) and "records" in content:
                    raw_rows = content["records"]
                else:
                    raise ValueError(
                        f"Unsupported JSON structure in {self.file_path}. Expected array of records."
                    )
        else:
            delimiter = self._detect_delimiter()
            with open(self.file_path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                if not reader.fieldnames:
                    raise ValueError(f"Empty or corrupted tabular ICD dataset at {self.file_path}")
                # Check for required columns
                field_lower = {fn.strip().lower(): fn for fn in reader.fieldnames if fn}
                code_col = next(
                    (field_lower[c] for c in self.CODE_ALIASES if c in field_lower), None
                )
                desc_col = next(
                    (field_lower[d] for d in self.DESC_ALIASES if d in field_lower), None
                )

                if not code_col or not desc_col:
                    raise ValueError(
                        f"Missing required columns in {self.file_path}. "
                        f"Must include a code column (found: {reader.fieldnames}) "
                        f"and a description column."
                    )

                for row in reader:
                    raw_rows.append(row)

        # Ingestion metrics
        records: dict[str, ICDCodeRecord] = {}
        duplicates_count = 0
        malformed_count = 0
        total_inclusion = 0
        total_exclusion = 0
        total_synonyms = 0

        for row in raw_rows:
            # Normalize keys to lower
            lower_row = {str(k).strip().lower(): v for k, v in row.items() if k is not None}

            # 1. Resolve Code
            raw_code = None
            for alias in self.CODE_ALIASES:
                if alias in lower_row and lower_row[alias]:
                    raw_code = str(lower_row[alias]).strip()
                    break

            # 2. Resolve Description
            raw_desc = None
            for alias in self.DESC_ALIASES:
                if alias in lower_row and lower_row[alias]:
                    raw_desc = str(lower_row[alias]).strip()
                    break

            # Validation: Code and Description must be present and valid
            if not raw_code or not raw_desc:
                logger.warning("Dropping malformed row: missing code or description: %s", row)
                malformed_count += 1
                continue

            if not is_valid_icd_format(raw_code):
                logger.warning(
                    "Dropping malformed row: '%s' violates ICD-10-CM format rules.",
                    raw_code,
                )
                malformed_count += 1
                continue

            formatted_code = format_icd_code(raw_code)
            unformatted_code = unformat_icd_code(raw_code)

            # Check for duplicates
            if formatted_code in records:
                duplicates_count += 1
                # Merge synonyms and inclusion terms into existing authoritative record
                existing = records[formatted_code]
                new_syns = self._parse_list_field(
                    lower_row.get("synonyms") or lower_row.get("synonym")
                )
                new_incls = self._parse_list_field(
                    lower_row.get("inclusion_terms") or lower_row.get("includes")
                )
                for s in new_syns:
                    if s not in existing.synonyms:
                        existing.synonyms.append(s)
                        total_synonyms += 1
                for inc in new_incls:
                    if inc not in existing.inclusion_terms:
                        existing.inclusion_terms.append(inc)
                        total_inclusion += 1
                continue

            # Additional fields
            category = lower_row.get("category") or unformatted_code[:3]
            chapter = lower_row.get("chapter")
            short_desc = lower_row.get("short_description") or raw_desc
            long_desc = lower_row.get("long_description") or raw_desc
            is_billable = self._parse_bool_field(
                lower_row.get("is_valid_billable")
                or lower_row.get("is_billable")
                or lower_row.get("billable")
                or lower_row.get("valid_billable"),
                default=True,
            )
            inclusion_terms = self._parse_list_field(
                lower_row.get("inclusion_terms") or lower_row.get("includes")
            )
            exclusion_terms = self._parse_list_field(
                lower_row.get("exclusion_terms")
                or lower_row.get("excludes")
                or lower_row.get("excludes1")
            )
            synonyms = self._parse_list_field(lower_row.get("synonyms") or lower_row.get("synonym"))
            excludes1 = self._parse_list_field(lower_row.get("excludes1"))
            excludes2 = self._parse_list_field(lower_row.get("excludes2"))
            code_first = self._parse_list_field(lower_row.get("code_first"))
            use_additional = self._parse_list_field(lower_row.get("use_additional"))

            total_inclusion += len(inclusion_terms)
            total_exclusion += len(exclusion_terms)
            total_synonyms += len(synonyms)

            record = ICDCodeRecord(
                code=formatted_code,
                unformatted_code=unformatted_code,
                description=raw_desc,
                short_description=short_desc,
                long_description=long_desc,
                is_valid_billable=is_billable,
                category=category,
                chapter=chapter,
                inclusion_terms=inclusion_terms,
                exclusion_terms=exclusion_terms,
                synonyms=synonyms,
                excludes1=excludes1,
                excludes2=excludes2,
                code_first=code_first,
                use_additional=use_additional,
                source_metadata={
                    "source": self.source_name,
                    "file_path": str(self.file_path),
                },
            )
            records[formatted_code] = record

        unique_categories = {r.category for r in records.values() if r.category}
        billable_count = sum(1 for r in records.values() if r.is_valid_billable)
        non_billable_count = len(records) - billable_count

        stats = ICDDatasetStats(
            total_records=len(records),
            valid_billable_count=billable_count,
            non_billable_count=non_billable_count,
            unique_categories_count=len(unique_categories),
            duplicates_dropped=duplicates_count,
            malformed_dropped=malformed_count,
            total_inclusion_terms=total_inclusion,
            total_exclusion_terms=total_exclusion,
            total_synonyms=total_synonyms,
            format_type="tabular",
        )

        self._records = records
        self._stats = stats
        self._is_loaded = True

        logger.info(
            "Tabular ICD dataset loaded from %s: %d records (%d billable, %d categories, %d duplicates dropped, %d malformed dropped)",
            self.file_path,
            stats.total_records,
            stats.valid_billable_count,
            stats.unique_categories_count,
            stats.duplicates_dropped,
            stats.malformed_dropped,
        )

        return self._records, self._stats


class CMSOrderFileLoader(BaseICDDatasetLoader):
    """Parses standard CMS / CDC ICD-10-CM Order Files (e.g. icd10cm_order_2026.txt).

    CMS Order file format:
    Positions 1-5: Order number (5 chars)
    Position 6: Blank
    Positions 7-13: ICD-10-CM code (7 chars alphanumeric, no decimal)
    Position 14: Blank
    Position 15: Valid for HIPAA submission (0=header/non-billable, 1=valid billable)
    Position 16: Blank
    Positions 17-76: Short description (60 chars)
    Position 77: Blank
    Positions 78+: Long description
    """

    def __init__(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path)
        self._records: dict[str, ICDCodeRecord] = {}
        self._stats: ICDDatasetStats | None = None
        self._is_loaded: bool = False

    def get_total_records(self) -> int:
        return len(self._records)

    def get_stats(self) -> ICDDatasetStats | None:
        return self._stats

    def load(self) -> tuple[dict[str, ICDCodeRecord], ICDDatasetStats]:
        """Parse local CMS flat file into structured records."""
        if self._is_loaded and self._stats is not None:
            return self._records, self._stats

        if not self.file_path.exists():
            raise FileNotFoundError(f"CMS order file not found at: {self.file_path}")

        records: dict[str, ICDCodeRecord] = {}
        duplicates_count = 0
        malformed_count = 0

        with open(self.file_path, encoding="utf-8-sig", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue

                if len(line) < 16:
                    logger.warning(
                        "Line %d in %s is malformed (too short): '%s'",
                        line_no,
                        self.file_path,
                        line,
                    )
                    malformed_count += 1
                    continue

                try:
                    # Slicing fixed positions
                    raw_code = line[6:13].strip()
                    valid_flag = line[14:15].strip()
                    short_desc = line[16:76].strip() if len(line) >= 76 else line[16:].strip()
                    long_desc = line[77:].strip() if len(line) > 77 else short_desc

                    if not raw_code or not is_valid_icd_format(raw_code):
                        malformed_count += 1
                        continue

                    if not short_desc and not long_desc:
                        malformed_count += 1
                        continue

                    formatted_code = format_icd_code(raw_code)
                    unformatted_code = unformat_icd_code(raw_code)
                    is_billable = valid_flag == "1"

                    if formatted_code in records:
                        duplicates_count += 1
                        continue

                    category = unformatted_code[:3]

                    record = ICDCodeRecord(
                        code=formatted_code,
                        unformatted_code=unformatted_code,
                        description=long_desc or short_desc,
                        short_description=short_desc,
                        long_description=long_desc or short_desc,
                        is_valid_billable=is_billable,
                        category=category,
                        source_metadata={
                            "source": "CMS_Order_File",
                            "line_number": line_no,
                            "file_path": str(self.file_path),
                        },
                    )
                    records[formatted_code] = record
                except Exception as e:
                    logger.warning("Error parsing line %d in CMS order file: %s", line_no, e)
                    malformed_count += 1

        unique_categories = {r.category for r in records.values() if r.category}
        billable_count = sum(1 for r in records.values() if r.is_valid_billable)
        non_billable_count = len(records) - billable_count

        stats = ICDDatasetStats(
            total_records=len(records),
            valid_billable_count=billable_count,
            non_billable_count=non_billable_count,
            unique_categories_count=len(unique_categories),
            duplicates_dropped=duplicates_count,
            malformed_dropped=malformed_count,
            format_type="cms_order_file",
        )

        self._records = records
        self._stats = stats
        self._is_loaded = True

        logger.info(
            "CMS order file loaded from %s: %d records (%d billable, %d categories)",
            self.file_path,
            stats.total_records,
            stats.valid_billable_count,
            stats.unique_categories_count,
        )

        return self._records, self._stats


class ExcelWorkbookLoader(BaseICDDatasetLoader):
    """Parses multi-sheet medical code workbooks containing ICD-10-CM, ICD-O, and CPT datasets.

    Supports reading .xlsx and .xls workbooks (such as Database_2.xlsx and Database_1.xls),
    segregating codes into their respective coding systems ('ICD-10-CM', 'ICD-O', 'CPT'),
    normalizing whitespace, and populating authoritative ICDCodeRecord models.
    """

    def __init__(self, file_or_dir_path: Path | str, **kwargs: Any) -> None:
        p = Path(file_or_dir_path)
        if p.is_dir():
            target = p / "Database_2.xlsx"
            if not target.exists():
                candidates = sorted(list(p.glob("*.xlsx")) + list(p.glob("*.xls")))
                if candidates:
                    target = candidates[0]
                else:
                    raise FileNotFoundError(f"No Excel workbooks found in {p}")
            self.file_path = target
        else:
            self.file_path = p

        self._records: dict[str, ICDCodeRecord] = {}
        self._stats: ICDDatasetStats | None = None
        self._is_loaded: bool = False

    def get_total_records(self) -> int:
        return len(self._records)

    def get_stats(self) -> ICDDatasetStats | None:
        return self._stats

    def load(self) -> tuple[dict[str, ICDCodeRecord], ICDDatasetStats]:
        if self._is_loaded:
            return self._records, self._stats or ICDDatasetStats(
                total_records=len(self._records),
                valid_billable_count=len(self._records),
                non_billable_count=0,
                unique_categories_count=0,
                format_type="excel_workbook",
            )

        import re

        import pandas as pd

        logger.info("Opening Excel workbook for multi-system clinical ingestion: %s", self.file_path)
        xl = pd.ExcelFile(self.file_path)

        records: dict[str, ICDCodeRecord] = {}
        duplicates_count = 0
        malformed_count = 0
        categories: set[str] = set()

        for sheet_name in xl.sheet_names:
            logger.info("Parsing sheet [%s] from %s", sheet_name, self.file_path.name)
            df = xl.parse(sheet_name)
            if df.empty or "code" not in df.columns or "description" not in df.columns:
                logger.warning("Skipping sheet [%s]: missing code or description column", sheet_name)
                continue

            is_cpt_sheet = "cpt" in sheet_name.lower()

            for _, row in df.iterrows():
                if "active_yesno" in row and pd.notna(row["active_yesno"]):
                    try:
                        if int(row["active_yesno"]) == 0:
                            continue
                    except (ValueError, TypeError):
                        pass

                raw_code = str(row["code"]).strip() if pd.notna(row["code"]) else ""
                raw_desc = str(row["description"]).strip() if pd.notna(row["description"]) else ""

                if not raw_code or not raw_desc or raw_code.lower() == "nan" or raw_desc.lower() == "nan":
                    malformed_count += 1
                    continue

                if is_cpt_sheet or (raw_code.isdigit() and len(raw_code) in (4, 5)):
                    coding_system = "CPT"
                    cat_id = row.get("ichi_code_cat1_id", "")
                    cat = f"CPT-{cat_id}" if pd.notna(cat_id) else "CPT"
                elif re.match(r"^M[89][0-9]{3}", raw_code, re.IGNORECASE):
                    coding_system = "ICD-O"
                    cat = "ICD-O-Morphology"
                else:
                    coding_system = "ICD-10-CM"
                    cat3 = row.get("icd10_code_cat3", "")
                    cat = str(cat3).strip() if pd.notna(cat3) else raw_code[:3]

                formatted_code = format_icd_code(raw_code)
                lookup_key = formatted_code

                if lookup_key in records:
                    duplicates_count += 1
                    continue

                categories.add(cat)
                rec = ICDCodeRecord(
                    code=formatted_code,
                    unformatted_code=unformat_icd_code(raw_code),
                    description=raw_desc,
                    short_description=raw_desc,
                    long_description=raw_desc,
                    is_valid_billable=True,
                    category=cat,
                    coding_system=coding_system,
                    source_metadata={
                        "source": "ExcelWorkbook",
                        "sheet": sheet_name,
                        "file": str(self.file_path.name),
                        "row_id": str(row.get("id", "")),
                    },
                )
                records[lookup_key] = rec

        billable_count = sum(1 for r in records.values() if r.is_valid_billable)
        stats = ICDDatasetStats(
            total_records=len(records),
            valid_billable_count=billable_count,
            non_billable_count=len(records) - billable_count,
            unique_categories_count=len(categories),
            duplicates_dropped=duplicates_count,
            malformed_dropped=malformed_count,
            format_type="excel_workbook",
        )

        self._records = records
        self._stats = stats
        self._is_loaded = True

        logger.info(
            "Excel workbook successfully ingested: %d codes (%d categories, %d duplicates filtered)",
            stats.total_records,
            stats.unique_categories_count,
            stats.duplicates_dropped,
        )

        return self._records, self._stats


def load_icd_dataset(
    file_path: Path | str, **kwargs: Any
) -> tuple[dict[str, ICDCodeRecord], ICDDatasetStats]:
    """Factory function detecting format from file extension and returning loaded records and stats."""
    path = Path(file_path)
    if path.is_dir() and (list(path.glob("*.xlsx")) or list(path.glob("*.xls"))):
        loader: BaseICDDatasetLoader = ExcelWorkbookLoader(path, **kwargs)
        return loader.load()

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        loader = ExcelWorkbookLoader(path, **kwargs)
    elif suffix in {".csv", ".tsv", ".json"}:
        loader = TabularICDLoader(path, **kwargs)
    elif suffix in {".txt"}:
        loader = CMSOrderFileLoader(path)
    else:
        # Default fallback to tabular loader
        loader = TabularICDLoader(path, **kwargs)

    return loader.load()
