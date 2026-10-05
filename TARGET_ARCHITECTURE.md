# Target Architecture: Evidence-First Clinical ICD Coding System

**Document Version**: 2.1  
**Design Principle**: Clinical Truth Authorization Precedes Code Retrieval  
**Scope**: Agentic Medical Coding Pipeline (`src/medical_coding`)  
**Branch**: `recall-fix`

---

## 1. Core Architectural Pipeline

The system is strictly ordered such that clinical truth understanding is decoupled from and precedes ICD catalog retrieval:

```
[DOCUMENT (PDF / Plain Text)]
   ↓
[Node 1: DOCUMENT VALIDATION & SANITIZATION]
   ↓
[Node 2: TEXT EXTRACTION & SECTION SEGMENTATION]
   ↓
[Node 3: EVIDENCE-FIRST FACT EXTRACTION]
   • Narrative Parsing (Unstructured Fallback)
   • Formatting & Role Prefix Stripping
   • Exact Span Anchoring & Polarity Assessment
   ↓
[Node 4: CONTEXT & RELEVANCE ASSESSMENT]
   • Normalized Section Mapping (PAST_MEDICAL_HISTORY vs Active)
   • Inpatient Management & Monitoring Verification
   ↓
[Node 5: UHDDS ROLE CLASSIFICATION]
   • Primary vs Secondary Hierarchy Scoring
   • Candidate-Specific Procedural Corroboration
   • Non-Arbitrary Ambiguity Abstention Gate
   ↓
[Node 6: AUTHORITATIVE ICD CANDIDATE RETRIEVAL]
   • Hybrid BM25 + FAISS Dense Vector Search
   • Bound to Local Authoritative Catalog
   ↓
[Node 7: MULTI-DIMENSIONAL CANDIDATE RANKING]
   • 7-Dimensional Scoring Engine
   • Terminal Unspecified Leaf Realignment
   ↓
[Node 8: 10 DETERMINISTIC VALIDATION GATES]
   • Catalog Existence, Excludes1, Leaf Specificity
   ↓
[Node 9: CONFIDENCE SYNTHESIS & AUDIT TRAIL]
   • Structured Abstention Records
   ↓
[Node 10: DETERMINISTIC JSON FINALIZATION]
   • Decoupled Clinical Diagnosis + ICD Code Mapping
```

### Invariant Principles:
1. **Clinical Diagnosis $\neq$ ICD Code Mapping**: A clinically confirmed diagnosis must never be discarded (`primary_diagnosis = null`) simply because ICD retrieval is uncertain or non-billable. If diagnosis is certain but mapping is uncertain:
   - `diagnosis`: Accepted (e.g., *"Acute right emphysematous pyelonephritis"*)
   - `icd10cm`: Abstained (`null`)
   - `abstentions`: Record detailing exact reason for ICD abstention
2. **Retrieval Cannot Authorize Diagnoses**: Vector similarity scores or keyword retrieval results cannot invent or promote a clinical diagnosis. Evidence from the clinical document alone authorizes diagnoses.
3. **Investigation / Diagnostic Findings are Not Autonomous Diagnoses**: Incidental or descriptive findings in echocardiograms, ultrasound, or CT scans (e.g. *"left ventricular hypertrophy secondary to systemic hypertension"*) do not constitute a documented primary or secondary diagnosis unless independently affirmed by treating clinicians in the diagnostic summary or hospital course.
4. **Output Specificity Cannot Exceed Documented Evidence**: A documented general condition (e.g. *"Heart failure"*) must never be mapped to a specific subtype (e.g. *"Acute systolic heart failure"* `I50.21`). If documented generally, the system must assign the terminal unspecified billable code (`I50.9`) or abstain if no leaf code exists.
5. **No Monkey Patching / Case-Specific Hacks**: No file-name, case-index, or hardcoded disease tables are permitted. All extraction and classification logic must be general, section-aware, and clinical-grammar driven.

---

## 2. Decoupled State Architecture

The pipeline replaces coupled dictionaries with two distinct typed state representations:

### 2.1 Clinical Diagnosis State (`ClinicalDiagnosisState`)
Represents the patient's medical condition as documented in the encounter:
```python
class StructuredEvidence(BaseModel):
    quote: str
    section: str  # DISCHARGE_DIAGNOSES, REASON_FOR_ADMISSION, HOSPITAL_COURSE, PROCEDURES, INVESTIGATIONS, PAST_MEDICAL_HISTORY, GENERAL
    evidence_type: str  # CLINICAL_SUMMARY, ADMISSION_REASON, PROCEDURAL_INTERVENTION, IMAGING, LAB, HISTORY
    polarity: str  # AFFIRMATIVE, NEGATED, UNCERTAIN
    certainty: str  # CONFIRMED, SUSPECTED, POSSIBLE, RULED_OUT
    temporality: str  # CURRENT, HISTORICAL, RESOLVED, BACKGROUND
    clinical_relevance: float  # [0.0 - 1.0]

class ClinicalDiagnosisCandidate(BaseModel):
    diagnosis_id: str
    raw_term: str
    normalized_diagnosis: str
    role: str  # PRIMARY, SECONDARY, HISTORICAL, SYMPTOM, INCIDENTAL, RULED_OUT, UNCERTAIN
    certainty: str  # CONFIRMED, SUPPORTED, UNCERTAIN, NEGATED
    temporality: str  # CURRENT, HISTORICAL, RESOLVED
    evidence: list[StructuredEvidence]
    evidence_strength: float
    clinical_relevance: float
    encounter_relevance: float
    treatment_relevance: float
    procedure_relevance: float
    contradiction_detected: bool = False
    is_authorized: bool = False
```

### 2.2 ICD Mapping State (`ICDMappingState`)
Represents the code assignment process for an authorized clinical diagnosis candidate:
```python
class ICDMappingCandidate(BaseModel):
    diagnosis_id: str
    code: str
    description: str
    coding_system: str  # ICD-10-CM, ICD-O, CPT
    is_terminal_billable: bool
    retrieval_score: float
    semantic_match_score: float
    specificity_match_score: float
    decision: str  # ACCEPTED, REJECTED, ABSTAINED
    rejection_reason: str | None = None
```

---

## 3. Primary Diagnosis Evidence Hierarchy & UHDDS Rules

Primary diagnosis determination strictly enforces the official UHDDS definition: **"The condition established after study to be chiefly responsible for occasioning admission to the hospital."**

### Authority Hierarchy:
1. **Priority 1**: Explicitly stated in `DISCHARGE DIAGNOSES`, `PRINCIPAL DIAGNOSIS`, or `FINAL DIAGNOSES` section.
2. **Priority 2**: Documented as chief reason for admission (`CHIEF COMPLAINT`, `REASON FOR ADMISSION`).
3. **Priority 3**: Condition chiefly responsible for major inpatient treatment or procedural source control (e.g., DJ stenting for acute pyelonephritis, emergent PCI for NSTEMI).
4. **Priority 4**: Supported by objective diagnostic investigations (CT, MRI, cultures).
5. **Priority 5**: Documented repeatedly throughout the inpatient hospital course.

### Ambiguity / Tie-Break Invariant:
When two or more acute conditions meet Priority 1 and Priority 2 equally with emergent inpatient therapy (e.g. simultaneous acute MI and acute stroke), the system must **not** arbitrarily select one. Both conditions must be designated with high admitting scores and the primary diagnosis marked as ambiguous/abstained for physician query, satisfying official coding guidance.

---

## 4. Multi-Dimensional Scoring Engine

Rather than relying on an opaque single confidence score, each candidate is evaluated across 7 interpretable dimensions:

$$\text{Final Acceptance} = f(\text{Dim}_1, \dots, \text{Dim}_7) \ge \text{Configured Thresholds}$$

1. **`evidence_score`** ($E \in [0.0, 1.0]$): Verbatim source text quality, presence of exact quote in document.
2. **`diagnostic_certainty`** ($C \in [0.0, 1.0]$): Confirmed (1.0), Supported (0.8), Suspected (0.5), Ruled Out (0.0).
3. **`encounter_relevance`** ($R \in [0.0, 1.0]$): Inpatient monitoring, medication administration, or active clinical management.
4. **`role_confidence`** ($P \in [0.0, 1.0]$): Margin of evidence hierarchy over competing conditions.
5. **`semantic_match`** ($S \in [0.0, 1.0]$): Exact clinical alignment between normalized diagnosis and ICD description.
6. **`specificity_match`** ($M \in [0.0, 1.0]$): Anatomical site, laterality, acuity, and etiology alignment.
7. **`contradiction_penalty`** ($K \in [0.0, 1.0]$): Penalty for negation cues ("ruled out", "no evidence of", "denies").

---

## 5. The 10 Deterministic Validation Gates

Every candidate must pass 10 discrete deterministic validation gates before reaching final coding output:

| Gate | Name | Gate Rule & Invariant | Failure Action |
|:---|:---|:---|:---|
| **Gate 1** | **Evidence Gate** | Candidate must have non-empty verbatim source quote present in raw document text. | REJECT |
| **Gate 2** | **Clinical Diagnosis Gate** | Candidate must represent an actual clinical condition, not a medication, procedure, or lab value. | ABSTAIN |
| **Gate 3** | **Role Gate** | Determine role (PRIMARY, SECONDARY, HISTORICAL, SYMPTOM, INCIDENTAL, RULED_OUT, UNCERTAIN). Max 1 PRIMARY. Inactive PMH excluded. | ENFORCE HIERARCHY / EXCLUDE |
| **Gate 4** | **ICD Candidate Gate** | Proposed codes must be retrieved from the local authoritative catalog (`LocalICDCatalog`). | ABSTAIN MAPPING |
| **Gate 5** | **Semantic Match Gate** | Core diagnostic meaning of clinical condition must match the retrieved ICD description. | REJECT CODE |
| **Gate 6** | **Specificity Gate** | Verified terminal/billable leaf code. If condition is unspecified (e.g. "Heart failure"), realign to terminal unspecified leaf (`I50.9`). | REALIGN TO LEAF / ABSTAIN |
| **Gate 7** | **Context Gate** | Anatomical site and laterality must conform to clinical documentation. | ABSTAIN CODE |
| **Gate 8** | **Contradiction Gate** | Reject if evidence contains negative polarity ("no evidence of", "ruled out", "denies"). | REJECT / EXCLUDE |
| **Gate 9** | **Unsupported-Inference Gate**| Reject any diagnosis proposed solely by retrieval similarity or indirect imaging text without diagnostic documentation. | REJECT |
| **Gate 10** | **Final Consistency Gate** | Enforce JSON schema invariants: 1 Primary max, no duplicates, valid audit trails. | SANITIZE OUTPUT |

---

## 6. Targeted Engineering Specifications for the 8 Root Causes

### 6.1 Diagnosis Prefix Stripping (`RC-1`)
In `src/medical_coding/agents/clinical_extractor.py`:
Before term/narrative splitting, clean lines of structural role prefixes:
```python
cleaned_line = re.sub(
    r"^(?:primary|principal|secondary|final|admitting)\s*(?:diagnosis|condition)?\s*:\s*",
    "",
    cleaned_line,
    flags=re.IGNORECASE,
).strip()
```

### 6.2 Narrative Fallback in Fact Extraction (`RC-2`)
In `src/medical_coding/agents/clinical_extractor.py`:
When `SectionSegmenter` outputs `section_name in ("GENERAL", "DISCHARGE_SUMMARY", "UNSTRUCTURED")`:
Extract diagnostic assertions from sentences containing admission phrases:
- `"admitted with [condition]"`
- `"diagnosed with [condition]"`
- `"presents with [condition]"`
- `"impression: [condition]"`
This ensures short clinical summaries (e.g., `"DISCHARGE SUMMARY\nPatient admitted with acute systolic heart failure."`) extract valid candidates.

### 6.3 Standardized Section Normalization (`RC-3`)
In `src/medical_coding/graph/nodes.py` (`_assess_context_deterministically`):
Normalize section strings to space-separated lowercase before pattern checking:
```python
norm_sec = section_lower.replace("_", " ")
is_pmh = "past medical" in norm_sec or "pmh" in norm_sec or "history" in norm_sec
```
This guarantees that `PAST_MEDICAL_HISTORY` correctly marks inactive conditions as non-billable historical entities.

### 6.4 Robust Admission Synonyms in UHDDS Tie Detection (`RC-4`)
In `src/medical_coding/agents/classifier.py` (`_classify_by_rules`):
Expand admission cue matching beyond literal `"chief complaint"`:
```python
admission_cues = [
    "chief complaint",
    "reason for admission",
    "admitted for",
    "presenting complaint",
    "occasioning admission",
    "emergent",
]
top_has_adm = any(cue in top_ctx for cue in admission_cues)
sec_has_adm = any(cue in sec_ctx for cue in admission_cues)
if top_has_adm and sec_has_adm:
    has_ambiguous_tie = True
```

### 6.5 Terminal Unspecified Leaf Code Realignment (`RC-5`)
In `src/medical_coding/validation/deterministic.py` and `src/medical_coding/agents/ranker.py`:
When a clinical diagnosis is documented without subtype (e.g. `"Heart failure"`), candidate ranking must select the terminal billable unspecified code (e.g. `I50.9`) instead of rejecting all codes or abstaining.

### 6.6 Candidate-Specific Procedural Corroboration (`RC-6`)
In `src/medical_coding/agents/clinical_extractor.py`:
Procedure relevance (+3.0 admitting score) is only assigned to a candidate if the procedure description contains words overlapping with the candidate's diagnostic term or anatomical site.

### 6.7 Chief Complaint Symptom Demotion (`RC-7`)
In `src/medical_coding/agents/clinical_extractor.py`:
Chief complaint items are tagged as `evidence_type = ADMISSION_REASON` and `role = SYMPTOM`. They serve to corroborate discharge diagnoses, and only provide a primary diagnosis fallback when no definitive discharge diagnosis exists.

### 6.8 Unification of Clinical State (`RC-8`)
In `src/medical_coding/graph/nodes.py`:
Preserve `ClinicalDiagnosisState` across nodes, ensuring that Node 4 and Node 5 validate and refine the rich candidate state rather than desynchronizing data models.

---

## 7. Concurrency & Performance Guarantees

- **Document Isolation**: Per-document pipelines operate on clean, isolated state instances without shared mutable objects.
- **Bounded Concurrency**: $\ge 10$ documents concurrent execution via `asyncio.Semaphore(10)`.
- **Cached Read-Only Resources**: The ICD catalog and FAISS/BM25 indices are loaded once into memory and treated as read-only.
- **Sub-500ms Latency for Text**: Deterministic section parsing, clinical fact normalization, and vectorized retrieval execute in pure Python without superfluous LLM loops.
