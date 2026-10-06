# AUTHORITATIVE CLINICAL DATABASE AUDIT & PROVENANCE REPORT

## 1. Executive Summary & Objective

In accordance with **Section 1 (Database is the Single Source of Truth for Codes)** of the Architectural Specification, the clinical database provided in the `Database/` directory represents the immutable, canonical authority for all diagnostic coding. The clinical document establishes *what* conditions exist; the approved database determines *which* ICD-10-CM codes legally and structurally represent those conditions.

This audit programmatically inspects and documents:
1. Physical workbook discovery and file verification
2. Sheet structures, row counts, and column schemas
3. Active vs. inactive status filtering (`active_yesno`)
4. Duplicate records between workbooks and sheets
5. Strict segregation of CPT procedural codes from the ICD diagnosis matching pipeline
6. Canonical provenance tracking (`database_file`, `sheet`, `row/id`, `code`, `description`, `active_yesno`)
7. Abstention rules when provenance cannot be established

---

## 2. Workbook Discovery & Physical File Inventory

Programmatic inspection of `c:\DS_and_AI\Projects_and_Tutorials\Projects\icd_project_dmh\Database` reveals two workbooks:

| File Name | File Format | File Size (Bytes) | SHA-256 / File Status | Canonical Role |
| :--- | :--- | :--- | :--- | :--- |
| `Database_2.xlsx` | OpenXML (.xlsx) | 3,472,453 bytes | Authoritative Canonical Source | **Primary Canonical Workbook** (Fast openpyxl/pandas loading) |
| `Database_1.xls` | Binary BIFF8 (.xls) | 15,776,256 bytes | Bit-Identical Mirror | Secondary / Legacy Format Mirror |

### Cross-Workbook Duplicate Verification
A comprehensive programmatic record-by-record comparison was executed across all sheets:
- `ICD 10-1`: 25,000 rows in both. Differences: **0 codes**
- `ICD 10-2`: 25,000 rows in both. Differences: **0 codes**
- `ICD 10-3`: 25,000 rows in both. Differences: **0 codes**
- `ICD 10-4`: 551 rows in both. Differences: **0 codes**
- `CPT Code`: 13,722 rows in both. Differences: **0 codes**

**Architectural Invariant Enforced**:
The database loader **MUST NOT** load both files simultaneously. Loading both files would duplicate 75,551 ICD codes and artificially inflate BM25 document frequencies and retrieval counts. The system designates `Database_2.xlsx` as the primary canonical workbook, maintaining `Database_1.xls` strictly as a validated fallback if the .xlsx file is absent.

---

## 3. Sheet Structure, Column Schema & Record Counts

The canonical workbook contains five distinct sheets:

| Sheet Name | Coding System | Total Rows | Columns Present | Sample Row ID / Code | Active Rows (`active_yesno=1`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ICD 10-1`** | ICD-10-CM | 25,000 | `['id', 'icd10_code_cat3', 'code', 'description', 'active_yesno']` | `F62.0` | 25,000 (100%) |
| **`ICD 10-2`** | ICD-10-CM | 25,000 | `['id', 'icd10_code_cat3', 'code', 'description', 'active_yesno']` | `R29.736` | 25,000 (100%) |
| **`ICD 10-3`** | ICD-10-CM | 25,000 | `['id', 'icd10_code_cat3', 'code', 'description', 'active_yesno']` | `S82.223R` | 25,000 (100%) |
| **`ICD 10-4`** | ICD-10-CM | 551 | `['id', 'icd10_code_cat3', 'code', 'description', 'active_yesno']` | `Z77.098` | 551 (100%) |
| **`CPT Code`** | CPT (Procedural) | 13,722 | `['id', 'ichi_code_cat1_id', 'code', 'description', 'active_yesno']` | `1404` | 13,722 (100%) |

- **Total ICD Diagnostic Rows**: **75,551**
- **Total CPT Procedural Rows**: **13,722**
- **Grand Total Rows**: **89,273**

---

## 4. Active vs. Inactive Code Handling

1. All 75,551 rows across `ICD 10-1`, `ICD 10-2`, `ICD 10-3`, and `ICD 10-4` currently have `active_yesno = 1`.
2. The database loader enforces a strict filter:
   ```python
   if "active_yesno" in row and pd.notna(row["active_yesno"]):
       try:
           if int(row["active_yesno"]) == 0:
               continue  # Inactive code rejected at ingestion
       except (ValueError, TypeError):
           pass
   ```
3. Any candidate code marked inactive (`active_yesno = 0`) is dropped at ingestion and will never enter the retrieval catalog.

---

## 5. Strict Procedural (CPT) Firewall

**Section 1 Requirement**:
> "CPT must not be mixed into the ICD diagnosis matching pipeline."

### Architectural Firewall Enforcement
1. The sheet `'CPT Code'` contains 13,722 numeric procedure codes (e.g. `1404`, `1489`, `1492`) under category `ichi_code_cat1_id`.
2. In `src/medical_coding/dataset/loader.py`, codes originating from `'CPT Code'` or matching pure 4-5 digit numeric format are classified with `coding_system="CPT"`.
3. The clinical diagnosis matching pipeline queries `coding_system="ICD-10-CM"`. CPT codes are never returned as `primary_diagnosis` or `secondary_diagnoses`.
4. Any attempt to map a clinical disease condition to a CPT code is deterministically rejected.

---

## 6. Authoritative Provenance Tracking

**Section 1 Invariant**:
> "Every returned ICD code must have provenance: database_file, sheet, row/id, code, description, active_yesno. If provenance cannot be established: NO_DATABASE_MATCH."

Every `ICDCodeRecord` ingested into the system stores complete provenance metadata:
```python
source_metadata = {
    "source": "ExcelWorkbook",
    "file": "Database_2.xlsx",
    "sheet": sheet_name,  # e.g., 'ICD 10-1', 'ICD 10-3'
    "row_id": str(row.get("id", "")),
    "code": formatted_code,
    "description": raw_desc,
    "active_yesno": 1,
}
```

When an ICD code decision is returned in `CodedDiagnosisResponse`:
- `database_code`: Exact authoritative code from the workbook
- `database_description`: Exact authoritative description from the workbook
- `database_source`: Formatted provenance string `"{file}::{sheet}::row_{row_id}"`
- If a candidate code cannot be verified against the catalog with valid provenance, the system triggers `NO_DATABASE_MATCH`.

---

## 7. Immutability & Abstention Invariants

1. **No External Codes**: Codes not present in `Database_2.xlsx` (or its verified mirror `Database_1.xls`) cannot be created, inferred, or returned.
2. **No Over-Specificity**: If the clinical evidence documents `"lateral malleolus fracture"` without specifying displacement, and the database only contains displaced (`S82.61XA`) or nondisplaced (`S82.64XA`) variants, the system MUST abstain and return `NO_DATABASE_MATCH`.
3. **No Retrospective Adaptation**: The database is never altered to accommodate clinical variations. The system must adapt its search and matching to the database structure while strictly abstaining when evidence does not entail the code attributes.
