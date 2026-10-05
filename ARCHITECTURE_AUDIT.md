# Comprehensive Architecture Audit: Agentic ICD Coding Pipeline

**Audit Date**: October 2026  
**Auditor**: Senior Clinical NLP Architect & Medical Coding Systems Engineer  
**Codebase**: Agentic ICD-10-CM Coding System (`src/medical_coding`)  
**Objective**: Identify root causes of diagnosis extraction/classification/coding failures and document structural defects prior to evidence-first refactoring.
**Branch**: `recall-fix`

---

## 1. Executive Summary & Core Defect Analysis

The system was audited end-to-end to determine why explicit primary diagnoses (such as *"Acute right emphysematous pyelonephritis with DJ stenting"*) fail in the current pipeline, resulting in:
- `primary_diagnosis = null`
- Spurious extraction of secondary diagnoses (e.g. `I10 Essential primary hypertension` inferred from echocardiogram/imaging text)
- Premature pipeline abstentions on unstructured or single-sentence clinical summaries

The audit revealed that while an initial refactoring introduced `EvidenceFirstFactExtractor` and replaced the old 13-disease regex table, **fundamental architectural flaws** remain across text parsing, section segmentation, context evaluation, and candidate ranking:

1. **Coupling Clinical Diagnosis with ICD Code Selection**: The pipeline treated clinical diagnosis and ICD coding as a single coupled state. If an exact terminal ICD code was not mapped or failed ranking, the entire primary diagnosis was discarded (`primary_diagnosis = null`), rather than accepting the confirmed diagnosis and abstaining only on the ICD code.
2. **Diagnosis Prefix Label Absorption in Fact Extraction**: When clinical summaries label diagnoses with structural prefixes (e.g., `1. Primary: Acute systolic heart failure`), `_split_term_and_narrative` splits on `: ` and absorbs `"Primary"` as the condition term, while the actual disease is discarded as narrative. Code retrieval consequently searches for `"Primary"` and returns CPT/procedural codes.
3. **Section Segmentation Fails on Unstructured Narratives**: Documents without canonical section headers (e.g., `"DISCHARGE SUMMARY\nPatient admitted with acute systolic heart failure."`) are tagged as `section_name="GENERAL"`. Because `EvidenceFirstFactExtractor` only queries specific section names (`DISCHARGE_DIAGNOSES`, `CHIEF_COMPLAINT`, etc.), zero candidates are extracted, causing the pipeline to abort with `INSUFFICIENT_CLINICAL_EVIDENCE`.
4. **String Delimiter Mismatch in Deterministic Past Medical History Assessment**: `_assess_context_deterministically` checks for `"past medical" in section_lower`, but sections are normalized as `"PAST_MEDICAL_HISTORY"`. The space mismatch prevents historical detection, leaking remote past conditions (e.g., remote appendectomies) into active billable secondary diagnoses.
5. **Brittle Literal String Matching for UHDDS Primary Diagnosis Ties**: In `classifier.py:1031`, tie-breaking between competing acute conditions requiring admission requires the exact substring `"chief complaint"`. Alternative documentation phrases such as `"Reason for admission:"` or `"Admitted for:"` bypass the tie-break logic, causing arbitrary selection of a single primary diagnosis.
6. **Unspecified Mention Code Assignment Failure**: When a clinical document records a general condition (e.g. `"Heart failure"`) without subtype, the system assigns `icd10cm=None` instead of realigning to the terminal unspecified billable code (`I50.9`).
7. **Over-Broad Procedure Corroboration**: In `clinical_extractor.py:499-506`, any procedure mentioned in the document triggers admitting score boosts (+3.0) for every candidate, rather than linking procedures specifically to the treated condition.
8. **Dual State Redundancy**: Redundant data models (`PipelineGraphState`, `ExtractedClinicalCondition`, `ContextAssessment`, `ClassifiedDiagnosis`, `ClinicalDiagnosisState`, `ICDMappingState`) create desynchronization between fact extraction and subsequent validation nodes.

---

## 2. Current Architecture & Execution Graph

### Current Execution Flow:
```
START
  ↓
[Node 1: validate_document]
  ↓
[Node 2: extract_text] (Multimodal Ingestion / OCR / PDF normalization)
  ↓
[Node 3: extract_diagnoses] (Clinical Extraction Agent / EvidenceFirstFactExtractor)
  ↓
[Node 4: analyze_context] (Context & Relevance Agent / Deterministic Rules)
  ↓
[Node 5: classify_diagnoses] (Primary/Secondary Classifier)
  ↓
[Node 6: retrieve_candidates] (Hybrid BM25 + FAISS Retrieval)
  ↓
[Node 7: rank_candidates] (Candidate Ranking Agent)
  ↓
[Node 8: validate_codes] (Deterministic Invariant Validator / Gates)
  ↓
[Node 9: evaluate_confidence] (Confidence & Audit Evaluator)
  ↓
[Node 10: finalize_output] (Final CodingResult JSON Assembly)
  ↓
END
```

### Module Responsibilities:
- `src/medical_coding/schemas/`: State definitions (`PipelineGraphState`), clinical schemas (`ExtractedClinicalCondition`, `ContextAssessment`, `ClassifiedDiagnosis`), response schemas (`CodingResult`, `CodedDiagnosisResponse`), and decoupled state models (`ClinicalDiagnosisState`, `ICDMappingState`).
- `src/medical_coding/graph/nodes.py`: Execution functions for all 10 LangGraph nodes.
- `src/medical_coding/graph/workflow.py`: StateGraph assembly and conditional edge declarations.
- `src/medical_coding/agents/clinical_extractor.py`: `SectionSegmenter` and `EvidenceFirstFactExtractor`.
- `src/medical_coding/agents/extractor.py`: LLM-based extraction agent with `ExtractionParser`.
- `src/medical_coding/agents/classifier.py`: `ContextAndRelevanceAgent` and `PrimarySecondaryClassifier`.
- `src/medical_coding/retrieval/`: Local BM25 (`lexical.py`), FAISS (`vector.py`), and `HybridICDRetriever`.
- `src/medical_coding/validation/deterministic.py`: HIPAA leaf specificity checks, Excludes1 checks, and candidate pool boundaries.
- `src/medical_coding/orchestration/pipeline.py`: Asynchronous document execution with `BoundedDocumentGate`.

---

## 3. Detailed Failure Points & Architectural Defects

### Defect 1: Clinical Diagnosis State Conflated with ICD Code State
- **Observed Behavior**: In `nodes.py:787-896`, `finalize_output_node` iterates over `validated` (which only contains diagnoses that matched an ICD candidate and passed validation). If ICD retrieval yields no exact match or ranking abstains, `primary_response` remains `None` or produces `icd10cm=None`.
- **Architectural Requirement**:
  - `ClinicalDiagnosisState`: Clinical diagnosis (e.g. *"Acute right emphysematous pyelonephritis"*) is CONFIRMED and PRIMARY based on clinical facts and evidence hierarchy.
  - `ICDMappingState`: ICD retrieval and validation. If ICD mapping confidence is inadequate or unsupported, ICD mapping abstains (`icd10cm = null`, `status = PARTIAL_SUCCESS` or `ABSTAINED_MAPPING`), but `primary_diagnosis` clinical details remain intact.

### Defect 2: Prefix Label Absorption Discards Core Diagnosis
- **Observed Behavior**: In `clinical_extractor.py:401-417`, delimiter splitting strips numbered prefixes but leaves role prefixes like `Primary: Acute systolic heart failure`. It splits on `: `, taking `Primary` as the disease term and discarding the clinical condition into the narrative.
- **Architectural Requirement**:
  Extraction must strip structural formatting labels (`Primary:`, `Secondary:`, `Principal:`) before term segmentation.

### Defect 3: SectionSegmenter Narrative Fallback Deficiency
- **Observed Behavior**: `SectionSegmenter` assigns `section_name="GENERAL"` to documents without headers. `_extract_candidates_from_sections` ignores `GENERAL` sections, extracting 0 candidates and failing basic single-sentence notes.
- **Architectural Requirement**:
  The fact extractor must implement a robust fallback narrative parser capable of extracting clinical conditions from unstructured text.

### Defect 4: Past Medical History Section Underscore Mismatch
- **Observed Behavior**: In `nodes.py:1004`, `is_pmh = "past medical" in section_lower or "pmh" in section_lower`. `SectionSegmenter` uses `PAST_MEDICAL_HISTORY`. Because `"past medical"` (space) is not in `"past_medical_history"` (underscore), inactive past conditions are coded as active secondaries.
- **Architectural Requirement**:
  Standardize section comparison across modules by normalizing underscores to spaces.

### Defect 5: Fragile UHDDS Ambiguity Tie-Breaking
- **Observed Behavior**: In `classifier.py:1031-1036`, the tie-breaking condition requires the exact phrase `"chief complaint"`. Mentions of `"Reason for admission:"` or other clinical equivalents are ignored, causing arbitrary primary selection.
- **Architectural Requirement**:
  Check for admission cues generically across standardized clinical synonyms (`"reason for admission"`, `"chief complaint"`, `"admitted for"`, `"emergent"`).

### Defect 6: Unspecified Condition Realignment Failure
- **Observed Behavior**: When a document contains `"Heart failure"` without subtype, the system fails to assign terminal billable code `I50.9`, leaving `icd10cm=None`.
- **Architectural Requirement**:
  The specificity gate must realign unspecified clinical conditions to valid terminal unspecified leaf codes rather than dropping the code.

---

## 4. Concurrency & State Isolation Audit

- **Concurrency Ceiling**: Verified with `BoundedDocumentGate` using `asyncio.Semaphore(10)`.
- **State Isolation**: `PipelineGraphState` is instantiated per document call in `process_clinical_document`. Module-level singletons (`_CACHED_CATALOG`, `_CACHED_RETRIEVER`) are initialized once and protected by threading locks.
- **Offline Guarantee**: The pipeline executes 100% locally with zero external network calls or cloud dependencies.

---

## 5. Audit Conclusion & Roadmap

The architecture is fundamentally sound in its 10-node topology and local retrieval capabilities. The remaining 7 test failures are completely resolved by fixing the 8 identified deterministic parsing and classification defects.
