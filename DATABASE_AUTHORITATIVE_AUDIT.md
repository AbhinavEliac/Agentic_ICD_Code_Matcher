# Database-Authoritative Clinical Coding & Evidence Extraction Engine
## Comprehensive Architecture Audit & Remediation Specification

**Date**: 2026-10-05  
**Corpus**: AbhinavEliac/Agentic_ICD_Code_Matcher  
**Authoritative Source**: `Database/Database_1.xls` & `Database/Database_2.xlsx`  
**Compliance Mandate**: CMS Official Guidelines for Coding and Reporting, UHDDS, Zero-Hallucination, Evidence-First  

---

## 1. CURRENT PIPELINE

The medical coding pipeline is orchestrated via a 10-node LangGraph state machine (`src/medical_coding/graph/workflow.py` and `src/medical_coding/graph/nodes.py`):

```text
[START]
   │
   ▼
[1. validate_document] ───────► (Validates text presence / PDF integrity; aborts on empty)
   │
   ▼
[2. extract_text] ────────────► (PyMuPDF / RapidOCR text extraction)
   │
   ▼
[3. extract_diagnoses] ───────► (ClinicalExtractor: Section-based + Narrative parsing)
   │
   ▼
[4. analyze_context] ─────────► (Context & Relevance: PMH, Negation, Ruled-out, Management)
   │
   ▼
[5. classify_diagnoses] ──────► (PrimarySecondaryClassifier: UHDDS Authority Hierarchy)
   │
   ▼
[6. retrieve_candidates] ────► (HybridICDRetriever: BM25 + FAISS candidate generation)
   │
   ▼
[7. rank_candidates] ─────────► (CandidateRankingAgent: Evidence overlap & Acuity match)
   │
   ▼
[8. validate_codes] ──────────► (DeterministicValidator: Catalog existence, Billability, Specificity)
   │
   ▼
[9. evaluate_confidence] ─────► (Confidence, Abstention & Audit Trail construction)
   │
   ▼
[10. finalize_output] ────────► (Pydantic CodingResult synthesis)
   │
   ▼
 [END]
```

### State Transitions & Data Structures
1. **Document Input**: Raw string or PDF filepath wrapped in `PipelineGraphState`.
2. **Extraction Stage**: Outputs `ClinicalDiagnosisCandidate` / `ExtractedDiagnosis` models linked to verbatim evidence quotes.
3. **Context Stage**: Generates `ContextAssessment` models tracking temporality, assertion, clinical relevance, and source section.
4. **Classification Stage**: Scores admitting significance and tags at most ONE `PRIMARY` condition, with active co-existing conditions assigned `SECONDARY` and unmanaged/ruled-out conditions assigned `EXCLUDED`.
5. **Retrieval Stage**: Queries the local catalog to produce `ICDCandidate` pools for billable conditions.
6. **Ranking & Validation Stage**: Selects top candidate and filters via deterministic gates into `ValidatedDiagnosis`.

---

## 2. CURRENT DATABASE USAGE

### Direct Inspection of `Database/` Workbooks
The `Database/` directory contains two workbook files:
1. `Database/Database_1.xls` (15,776,256 bytes — legacy BIFF8 binary workbook)
2. `Database/Database_2.xlsx` (3,472,453 bytes — OpenXML workbook)

### Detailed Comparison & Verification
An automated sheet-by-sheet and cell-by-cell inspection reveals:
* **Identical Structure**: Both files contain identical sheets: `['ICD 10-1', 'ICD 10-2', 'ICD 10-3', 'ICD 10-4', 'CPT Code']`.
* **Identical Data**: Every single cell across all 5 sheets is 100% identical between `Database_1.xls` and `Database_2.xlsx`. `Database_1.xls` is simply the uncompressed legacy binary export, while `Database_2.xlsx` is the modern zipped OpenXML format of the exact same database.
* **Row Counts by Sheet**:
  * `ICD 10-1`: 25,000 rows
  * `ICD 10-2`: 25,000 rows
  * `ICD 10-3`: 25,000 rows
  * `ICD 10-4`: 551 rows
  * `CPT Code`: 13,722 rows
  * **Total Raw Rows across sheets**: 89,273 rows.
* **Active Status (`active_yesno`)**:
  * All 89,273 rows have `active_yesno = 1` (100% active).
  * Inactive record count: 0.
* **Unique Codes vs. Duplicates**:
  * Total unique `(code, coding_system)` tuples: **83,319**.
  * Redundant/duplicate rows within identical coding systems across sheets: **5,954** duplicates.
  * Runtime indexing deduplicates these 89,273 rows into exactly 83,319 authoritative records.
* **Coding Systems Breakdown**:
  * **ICD-10-CM**: 72,204 diagnosis codes (spanning chapters A00 through Z99).
  * **ICD-O**: 3,347 morphology codes (WHO International Classification of Diseases for Oncology, prefixed with `M8000.0` through `M9999.9` located in sheet `ICD 10-1`).
  * **CPT**: 13,722 procedure and service codes (numeric 5-digit codes `00100` through `99499` in sheet `CPT Code`).

### Persisted Catalog Cache
* `data/indexes/icd_catalog.json` contains the pre-compiled JSON serialization of the 83,319 deduplicated records from `Database_2.xlsx`, storing:
  * `code`: formatted alphanumeric code.
  * `unformatted_code`: code without period delimiters.
  * `description`: verbatim database description.
  * `coding_system`: `'ICD-10-CM'`, `'ICD-O'`, or `'CPT'`.
  * `category`: chapter/category identifier.
  * `is_valid_billable`: terminal specificity status.
  * `source_metadata`: provenance tracking (`file='Database_2.xlsx'`, `sheet`, `row_id`).

---

## 3. CURRENT RETRIEVAL STRATEGY

Retrieval is handled by `HybridICDRetriever` (`src/medical_coding/retrieval/hybrid.py`), combining:
1. **Lexical Retrieval (`BM25ICDRetriever`)**:
   * Tokenizes searchable text (description, synonyms, unformatted code) using clinical regex tokenization.
   * Computes normalized BM25 score combined with query term coverage:
     $$\text{Score}_{\text{lex}} = 0.5 \times \frac{\text{raw\_bm25}}{\max(\text{raw\_bm25})} + 0.5 \times \text{coverage}$$
2. **Dense Vector Retrieval (`FAISSICDRetriever`)**:
   * Generates 384-dimensional dense vectors using `FastLocalEmbeddings` (sentence-transformers / MiniLM).
   * Executes cosine similarity search over `data/indexes/faiss.index`.
3. **Hybrid Rank Fusion**:
   * Weighted combination:
     $$\text{Score}_{\text{hybrid}} = 0.6 \times \text{Score}_{\text{sem}} + 0.4 \times \text{Score}_{\text{lex}}$$
   * Top-N window: `top_k=25` (with an initial search window of $3 \times k$).
   * System filtering: Neoplasms query ICD-O; procedural interventions query CPT; medical diagnoses query ICD-10-CM.

---

## 4. CURRENT VALIDATION STRATEGY

Validation is performed by `DeterministicValidator` (`src/medical_coding/validation/deterministic.py`):
1. **Gate 1: Catalog Existence**: Verifies that the proposed code exists in the 83,319-record database. Rejects hallucinated codes.
2. **Gate 2: Terminal Specificity (HIPAA Billability)**: Verifies that the proposed code is a terminal leaf code, rejecting non-billable 3-character category headers.
3. **Gate 3: Specificity Support Check**:
   * Penalizes unsupported heart failure subtypes (systolic, diastolic, right, left, high output, end stage, biventricular, other) and unsupported diabetes complications.
   * Realignment: If a specific code was proposed for general documentation, realigns to the unspecified terminal code (`I50.9`).
4. **Gate 4: Single Primary Invariant**: Enforces UHDDS requirement of at most ONE primary diagnosis per inpatient encounter.
5. **Gate 5: Excludes1 Conflict Detection**: Checks tabular instruction conflicts between co-billed codes.

---

## 5. FAILURE ROOT CAUSES (EMPIRICAL AUDIT)

### Issue A: Case 5 Failure — Over-Specific Code Assignment for Chronic Gastritis
* **Empirical Run**:
  * Input: `"Patient admitted with chronic gastritis."`
  * System Output: Primary diagnosis: `chronic gastritis` | Code: `K29.41` | Description: `Chronic atrophic gastritis with bleeding`.
* **Root Cause**:
  1. The database contains multiple gastritis codes:
     * `K29.50` (`Unspecified chronic gastritis without bleeding`)
     * `K29.41` (`Chronic atrophic gastritis with bleeding`)
     * `K29.40` (`Chronic atrophic gastritis without bleeding`)
  2. Because the patient document mentioned `"chronic gastritis"`, lexical overlap matched `"chronic"` and `"gastritis"`.
  3. Dense embeddings scored `K29.41` slightly higher than `K29.50`.
  4. The candidate ranker and validator currently lack a **Reverse Attribute Checker**:
     * Implied by `K29.41`: `atrophic=True`, `bleeding=True`.
     * Supported by evidence: `atrophic=False`, `bleeding=False`.
  5. Because the code was not explicitly checked for unsupported attributes (`atrophic`, `bleeding`), `K29.41` passed through and was assigned to the patient, directly violating Section 18 (Evidence-Bounded Specificity) and Section 31 (Case 5 Prevention).

### Issue B: Near-Match Semantic Drift (Case 4 Type)
* **Context**:
  * Input: `"Patient admitted with right ureteric calculus."`
  * Database contains:
    * `N20.1` (`Calculus of ureter`)
    * `N23` (`Unspecified renal colic`)
* **Risk Mechanism**:
  * In pure semantic search, `"renal colic"` and `"ureteric calculus"` have high cosine similarity because renal colic is a classic clinical presentation of ureteric calculus.
  * If lexical token overlap is insufficient or semantic weighting dominates, a retriever without strict diagnostic entity equivalence can rank `N23` above `N20.1`.
  * Coding a symptom/presentation (`N23`) when the definitive underlying calculus (`N20.1`) is documented violates CMS Coding Guideline I.B.4 and Section 30 of the Master Prompt.

### Issue C: Heading & Test Metadata Contamination
* If a document or synthetic test contains structural markers such as `EXPECTED: Primary: Acute bronchitis, ICD: J20.9` or section titles like `COURSE IN THE HOSPITAL`, naive regex/token extraction risks parsing the word `EXPECTED` or `PRIMARY_DIAGNOSIS` as a clinical condition.

---

## 6. RISK ANALYSIS & INVARIANT ENFORCEMENT

| Risk Category | Invariant Violated | Impact | Architectural Remediation |
|---|---|---|---|
| **Database Integrity** | Rule 1 & 2: Database is Read-Only | Corrupting source workbooks or inventing codes. | Enforce strict read-only access to `Database/Database_2.xlsx`. Store all runtime indices in `data/indexes/`. |
| **Clinical Evidence** | Rule 4 & 7: Evidence-First Authority | Hallucinating clinical facts or evidence quotes. | Mandatory source span and verbatim quote extraction before code retrieval. If evidence is absent, `evidence_supported = False` and candidate is rejected. |
| **Matching & Specificity** | Rule 18 & 22: Code Specificity $\le$ Evidence Specificity | Over-coding general conditions into severe/complicated subtypes (e.g. `K29.41`). | Implement automated **Reverse Attribute Engine** extracting qualifiers (`bleeding`, `atrophic`, `gangrene`, `obstruction`, `perforation`) from database descriptions and rejecting candidates with unsupported attributes. |
| **Near-Match Semantic Drift** | Rule 20 & 21: Database Description Authoritative | Substituting related symptoms (renal colic `N23`) for definitive disease (`N20.1`). | Require diagnostic concept equivalence over semantic association; symptom terms deprioritized when definitive etiology is present. |
| **Assertion & Temporality** | Rule 8: Assertion Rigor | Coding ruled-out (`MI ruled out`) or historical (`PMH appendectomy`) conditions as active. | Pre-retrieval Assertion & Temporality filter classifying mentions into `CONFIRMED`, `RULED_OUT`, `HISTORICAL`, `SUSPECTED`. |
| **Primary/Secondary Role** | Rule 12 & 13: UHDDS Authority | Selecting primary based on embedding score or heading order. | UHDDS scoring hierarchy: Discharge diagnoses (+5.0), Chief Complaint / Admission Reason (+3.5), Acuity (+2.0), Procedures (+3.5). Maximum ONE Primary. |

---

## 7. RECOMMENDED TARGET ARCHITECTURE

To satisfy all requirements without touching the source database, we implement a **Database-Authoritative, Reverse-Attribute-Bounded Pipeline**:

```text
                  SOURCE DOCUMENT
                        │
                        ▼
             1. DOCUMENT PARSING & CLEANING
         (Strip test metadata, EXPECTED:, headings)
                        │
                        ▼
          2. CLINICAL EVIDENCE EXTRACTION
      (Verbatim text, section, location, acuity)
                        │
                        ▼
       3. ASSERTION & TEMPORALITY FILTER
  (Classify CONFIRMED, RULED_OUT, HISTORICAL, SUSPECTED)
                        │
                        ▼
     4. PRIMARY / SECONDARY CLASSIFIER
 (UHDDS Significance Scoring; max 1 Primary; Symptoms deprioritized)
                        │
                        ▼
       5. CODING SYSTEM ROUTER
    (ICD-10-CM for diagnoses, ICD-O for morphology, CPT for procedures)
                        │
                        ▼
        6. DATABASE-ONLY CANDIDATE RETRIEVAL
      (BM25 + FAISS over Database_2.xlsx 83,319 records)
                        │
                        ▼
       7. HARD COMPATIBILITY & REVERSE ATTRIBUTE GATE
   (Verify active_yesno=1; Extract implied attributes from code desc;
    Reject candidate if implied attribute NOT supported by evidence)
                        │
                        ▼
        8. BIDIRECTIONAL VALIDATION
 (Evidence entails Diagnosis AND Database Description entails Evidence)
                        │
                        ▼
        9. DETERMINISTIC SPECIFICITY REALIGNMENT
   (If general condition documented, select exact general/unspecified code,
    or output NO_DATABASE_MATCH if no valid code exists)
                        │
                        ▼
       10. STRUCTURED AUDIT & OUTPUT GENERATION
  (Full traceability: document -> evidence -> diagnosis -> database code)
```

---

## 8. IMPLEMENTATION PLAN

### Phase 1: Reverse Attribute Checker (`src/medical_coding/validation/reverse_attributes.py`)
* Construct a deterministic reverse attribute parser that extracts clinical qualifiers from database descriptions:
  * **Complications**: `with bleeding`, `with perforation`, `with obstruction`, `with gangrene`, `with ulceration`.
  * **Pathological Subtypes**: `atrophic`, `superficial`, `eosinophilic`, `metaplastic`, `ulcerative`.
  * **Acuity & Chronicity**: `acute`, `chronic`, `subacute`, `acute on chronic`.
  * **Etiologies**: `alcoholic`, `toxic`, `rheumatic`, `transplant`, `hypertensive`, `postprocedural`, `congenital`, `infectious`.
  * **Laterality**: `right`, `left`, `bilateral`, `unspecified`.
* For every candidate code, compare its implied attributes against the patient's documented evidence:
  $$\forall \text{attr} \in \text{Code Attributes}: \text{attr} \in \text{Evidence} \lor \text{attr} == \text{"unspecified"} \lor \text{attr} == \text{"without"}$$
* If any positive clinical qualifier is absent from the evidence, **REJECT** the candidate code.

### Phase 2: Candidate Ranking & Selection Alignment (`src/medical_coding/agents/ranker.py`)
* Integrate the Reverse Attribute Checker directly into candidate scoring:
  * Candidates with unsupported attributes receive an immediate disqualification score ($0.05$).
  * Candidates whose description matches the exact evidence level (e.g. `K29.50` `Unspecified chronic gastritis without bleeding` for `"chronic gastritis"`) receive top preference.
* Prevent near-match semantic drift:
  * Ensure `N20.1` (`Calculus of ureter`) is selected for `"ureteric calculus"` rather than `N23` (`Unspecified renal colic`), enforcing that disease entities outrank symptom entities.

### Phase 3: Coding System Boundary Enforcement (`src/medical_coding/graph/nodes.py`)
* Ensure diagnostic conditions are strictly routed to `ICD-10-CM` (or `ICD-O` for neoplasms).
* Ensure procedural interventions are routed strictly to `CPT`.
* Verify `active_yesno == 1` from the database.

### Phase 4: Output Contract & Auditability (`src/medical_coding/schemas/response.py`)
* Format output conforming to Section 27 of the Master Prompt, providing full audit provenance:
  * `database_source`: `"Database/Database_2.xlsx"`
  * `database_record_id`: row ID from the database workbook
  * `code`: formatted database code
  * `database_description`: verbatim description from the database
  * `validation`: explicit booleans confirming database-only usage, zero unsupported specificity, and zero hallucination.

---

## 9. REGRESSION PLAN

1. **Verify Existing Suite**:
   * Run full test suite (`pytest -q`): Must maintain 100% pass rate (125/125 tests passing).
2. **Execute Diagnostic Suite**:
   * Run `scripts/run_pipeline_diagnostic_report.py`: Ensure all standard cases pass with exact codes.
3. **Verify Case 4**:
   * Test: `"Patient admitted with right ureteric calculus."`
   * Target: Primary code = `N20.1` (`Calculus of ureter`), NOT `N23` (`Unspecified renal colic`).
4. **Verify Case 5**:
   * Test: `"Patient admitted with chronic gastritis."`
   * Target: Primary code = `K29.50` (`Unspecified chronic gastritis without bleeding`), NOT `K29.41` (`Chronic atrophic gastritis with bleeding`).
5. **Verify No-Database-Match**:
   * Test: Unmatched clinical term without database representation returns `coding_status: NO_DATABASE_MATCH` and `code: None`.
6. **Zero Database Modifications**:
   * Verify git diff confirms zero edits, zero deletions, and zero renames in `Database/`.
