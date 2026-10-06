# COMPREHENSIVE ROOT-CAUSE ARCHITECTURAL AUDIT REPORT

## 1. System Overview & Architecture Map

The medical coding pipeline is an asynchronous, evidence-first clinical information extraction and database-authoritative ICD-10-CM matching engine. It operates under a strict directionality constraint:
```
CLINICAL DOCUMENT
       │
       ▼
1. DOCUMENT PARSER & VALIDATOR
       │
       ▼
2. SECTION CLASSIFIER & SEGMENTER
       │
       ▼
3. CLINICAL EVIDENCE EXTRACTION
       │
       ▼
4. ASSERTION, NEGATION & TEMPORALITY ENGINE
       │
       ▼
5. HARD CLINICAL CANDIDATE GATE (20 Criteria)
       │
       ▼
6. ATOMIC COMPOUND DECOMPOSITION & ATTRIBUTE ISOLATION
       │
       ▼
7. CANONICAL DEDUPLICATION & MERGING
       │
       ▼
8. COMPLETE CANDIDATE INVENTORY
       │
       ▼
9. GLOBAL PRIMARY / SECONDARY ROLE CLASSIFICATION
       │
       ▼
10. STAGED CONSTRAINED DATABASE RETRIEVAL (Isolated Context)
       │
       ▼
11. HARD DATABASE COMPATIBILITY VALIDATION (CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY)
       │
       ▼
12. FINAL DETERMINISTIC CONSISTENCY GATE
       │
       ▼
STRUCTURED AUDITABLE JSON (Final Payload)
```

---

## 2. Comprehensive Root Cause Matrix

Below is the detailed architectural audit of the 10 failure modes identified in Section 41 and the codebase inspection.

---

### Issue 1: Orthopedic Compound Diagnosis Collapse & Splitting Errors
- **Location**: `src/medical_coding/agents/clinical_extractor.py` (`_split_compound_clinical_phrase`)
- **Architectural Cause**:
  Clinical fracture diagnoses frequently present as compound anatomical clauses (e.g., *"Multiple fractures of left lateral malleolus, tarsals and fifth metatarsal with deltoid and calcaneofibular ligament sprains"*). The existing regex splitting either collapsed the entire clause into a single non-matching concept or split on conjunctions haphazardly, producing malformed fragments such as *"fifth metatarsal with deltoid"*. Furthermore, laterality (*"left"*) and the entity type (*"fracture"*) were not systematically distributed across the decomposed anatomical sites.
- **Affected Behavior**:
  Orthopedic cases failed candidate ranking, returned empty codes (` `), and missed secondary fractures and sprains.
- **Generalized Fix**:
  Implement an Orthopedic Compound Decomposer that parses anatomical lists, distributes shared laterality and injury types across each listed bone, and decomposes associated ligament injuries into distinct, independent atomic candidates:
  1. `left lateral malleolus fracture`
  2. `left tarsal fracture`
  3. `left fifth metatarsal fracture`
  4. `left deltoid ligament sprain`
  5. `left calcaneofibular ligament sprain`
- **Expected Regression Impact**:
  Each orthopedic injury is evaluated independently against the database; 100% recall of multiple fractures and sprains; zero malformed candidate phrases.

---

### Issue 2: Candidate Pollution by Metadata, Section Headings & Isolated Anatomical Sites
- **Location**: `src/medical_coding/validation/clinical_gate.py`
- **Architectural Cause**:
  The candidate gate lacked explicit rejection patterns for:
  1. Document sub-headers and metadata artifacts (e.g., `"Classification of Fracture: Not Applicable"`, `"BIOMARKERS & RECEPTOR STATUS"`)
  2. Isolated anatomical locations lacking pathological processes (e.g., `"Left foot"`)
  3. Biomarkers and receptor statuses (e.g., `"Triple negative"`, `"BRCA pathogenic"`, `"PD-L1 positive"`)
- **Affected Behavior**:
  These non-disease phrases passed extraction, queried the database, and generated absurd false positive matches (e.g., `"Triple negative"` matched `"Gram-negative sepsis"`; `"PD-L1 positive"` matched `"Lumbar spinal cord injury"`; `"BIOMARKERS"` matched `"Cystostomy"`).
- **Generalized Fix**:
  Enforce the 20-point Hard Candidate Gate:
  - Reject standalone section headers and metadata.
  - Reject isolated anatomical nouns without condition descriptors (e.g., `"foot"`, `"ankle"`, `"breast"`, `"chest"`).
  - Reject standalone molecular, genetic, receptor, and staging attributes.
- **Expected Regression Impact**:
  Total elimination of candidate pollution; 0 false positive ICD codes; 0 noisy abstentions.

---

### Issue 3: Oncology Concept Flattening & Metastatic Site Disconnection
- **Location**: `src/medical_coding/schemas/evidence.py`, `src/medical_coding/agents/clinical_extractor.py`, `src/medical_coding/retrieval/hybrid.py`
- **Architectural Cause**:
  Oncology presentations were treated as flat text strings. Attributes (ER/PR/HER2, BRCA, PD-L1, Stage IV, Grade III) were treated as potential disease candidates rather than being bound to a structured `CancerConcept`. Furthermore, secondary metastases (e.g., `"Pleural metastasis"`) were searched as isolated respiratory phrases, retrieving non-malignant symptom codes (`J94.9` "Pleural condition, unspecified") rather than oncology secondary neoplasm codes (`C78.2` "Secondary malignant neoplasm of pleura").
- **Affected Behavior**:
  Primary breast cancer was demoted or failed matching; pleural metastasis was misclassified as a benign pleural effusion/condition; receptor attributes leaked into retrieval.
- **Generalized Fix**:
  1. Define a typed `CancerConcept` model encapsulating primary site, histology, stage, grade, receptor status, and metastatic sites.
  2. Bind receptor and molecular markers to the primary cancer object as attributes.
  3. When evidence establishes a metastatic site linked to a primary malignancy, constrain retrieval for the metastatic site to ICD-10 Chapter 2 (`C77-C79` Secondary malignant neoplasms).
- **Expected Regression Impact**:
  Metastatic breast cancer mapped to primary `C50.919` (or site-specific); pleural metastasis mapped to secondary `C78.2`; zero attribute leakage.

---

### Issue 4: Premature Role Classification & Generic Chronicity Penalties
- **Location**: `src/medical_coding/agents/classifier.py` (`_score_and_classify_rules`)
- **Architectural Cause**:
  A hardcoded penalty of `-1.5` was subtracted from any condition containing `"chronic"`, regardless of whether the provider explicitly documented it under `PRIMARY DIAGNOSIS:` or `DISCHARGE DIAGNOSES:`. When a patient was admitted specifically for chronic gastritis without bleeding, this penalty undermined explicit provider documentation. Furthermore, role classification was performed without a globally deduplicated candidate inventory.
- **Affected Behavior**:
  Explicit primary chronic diagnoses were demoted or duplicate primary/secondary candidates were formed.
- **Generalized Fix**:
  1. Provide overwhelming priority (+10.0) to conditions explicitly designated under `PRIMARY DIAGNOSIS:`, `PRINCIPAL DIAGNOSIS:`, or `ADMITTED FOR:`.
  2. Forbid chronicity penalties when a condition is documented as the principal diagnosis or admission driver.
  3. Enforce deterministic candidate deduplication prior to role classification, producing exactly ONE canonical candidate per disease concept.
- **Expected Regression Impact**:
  100% primary diagnosis precision and recall; complete prevention of duplicate primary/secondary representations.

---

### Issue 5: Reverse Attribute Entailment & Unsupported Specificity
- **Location**: `src/medical_coding/validation/reverse_attributes.py`, `src/medical_coding/agents/ranker.py`
- **Architectural Cause**:
  Incomplete token coverage in `CLINICAL_QUALIFIERS` allowed certain database codes requiring unsupported attributes (e.g., displacement, fracture type, sequela, open/closed) to slip past validation, or caused unjustified rejections when valid synonyms existed.
- **Affected Behavior**:
  Risk of selecting displaced fracture codes when displacement was unmentioned, or failing to abstain when the database only offers over-specific codes.
- **Generalized Fix**:
  Strictly enforce `CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY`:
  - If a code description contains positive attributes (e.g. `displaced`), verify that positive evidence exists in the clinical text.
  - If the document states `nondisplaced`, only accept nondisplaced codes (e.g. `S82.64XA` / `S82.65XA`).
  - If the document is silent on displacement and the database only provides displaced/nondisplaced codes, deterministically abstain with `NO_DATABASE_MATCH`.
- **Expected Regression Impact**:
  Zero unsupported specificity violations; safe, auditable abstention on ambiguous records.

---

### Issue 6: Cross-Candidate Retrieval Contamination
- **Location**: `src/medical_coding/graph/nodes.py` (`retrieve_candidates_node`, `rank_candidates_node`)
- **Architectural Cause**:
  Potential pooling of candidate queries or shared scoring spaces could allow candidates from one disease to pollute another.
- **Affected Behavior**:
  Diabetes retrieval poisoned by hypertension candidates; breast cancer retrieval poisoned by thoracic trauma candidates.
- **Generalized Fix**:
  Enforce strict retrieval isolation:
  - Each canonical diagnosis candidate executes an independent, isolated query against the catalog.
  - No global candidate pool; no shared top-K.
- **Expected Regression Impact**:
  Complete isolation of candidate ranking; zero cross-contamination.

---

### Issue 7: Inpatient Corroboration for Pre-existing Comorbidities
- **Location**: `src/medical_coding/agents/clinical_extractor.py`, `src/medical_coding/agents/classifier.py`
- **Architectural Cause**:
  Under UHDDS, pre-existing comorbidities (e.g., diabetes, hypertension, CKD) documented in PMH can only be coded as secondary diagnoses if they required clinical evaluation, therapeutic treatment, or monitoring during the inpatient stay. The extractor previously failed to link inpatient monitoring (e.g., sliding scale insulin, blood pressure medication, creatinine checks) to PMH conditions.
- **Affected Behavior**:
  Legitimate secondary comorbidities were either excluded or missed.
- **Generalized Fix**:
  Implement cross-section corroboration: when a condition appears in PMH, inspect the Hospital Course and Medications sections for active monitoring or treatment. If corroborated, promote to active secondary; if unmanaged, exclude.
- **Expected Regression Impact**:
  100% recall of secondary comorbidities (T2DM `E11.9`, HTN `I10`, CKD `N18.30`) with zero leakage of unmanaged history (e.g., history of falling).
