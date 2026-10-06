# Production Clinical AI Reasoning Pipeline Audit

**Repository:** `AbhinavEliac/Agentic_ICD_Code_Matcher`  
**Branch:** `secondary-fix`  
**Engineer:** Senior Production AI Architect  
**Objective:** Audit existing pipeline and define minimal architecture delta for Clinical Concept Reasoning, MatchSpec constraints, Staged Retrieval, and Compatibility Validation.

---

## 1. Current Architecture Map

The system currently processes clinical discharge summaries and EHR notes to extract diagnoses and match authoritative ICD-10-CM codes.

```text
[Input Document (PDF / Text)]
       │
       ▼
[Ingestion & Section Segmentation] (ClinicalDocumentIngestionAgent / SectionSegmenter)
       │
       ▼
[Clinical Extraction] (ClinicalExtractionAgent / EvidenceFirstFactExtractor)
       │
       ▼
[Context Assessment] (ContextAndRelevanceAgent)
       │
       ▼
[Primary/Secondary Classification] (PrimarySecondaryClassifier)
       │
       ▼
[ICD Candidate Retrieval] (HybridICDRetriever: FAISS + BM25)
       │
       ▼
[Candidate Ranking & Attribute Checking] (CandidateRankingAgent + ReverseAttributeChecker)
       │
       ▼
[Deterministic Validation] (CandidateRankingDeterministicValidator)
       │
       ▼
[Confidence Assessment & Output Serialization] (finalize_output_node -> CodingResult)
```

---

## 2. Current LLM Call Graph

```text
extract_diagnoses_node:
  ├── Check settings.get_resolved_model_path().exists()
  │     ├── If True: ClinicalExtractionAgent.extract_clinical_conditions(raw_text) [LLM Call 1]
  │     └── If False / Exception: _extract_conditions_deterministically(raw_text) [Deterministic]
  └── EvidenceFirstFactExtractor.extract_clinical_state(raw_text) [Deterministic]

analyze_context_node:
  └── ContextAndRelevanceAgent.assess_conditions(conditions)
        ├── If LLM configured: LLM Call 2 (JSON parse attempt, retries on failure)
        └── Fallback: _fallback_rule_assessment [Deterministic]

classify_diagnoses_node:
  └── PrimarySecondaryClassifier.classify_conditions(assessments)
        ├── Passed RulePassThroughLLM -> attempts LLM Call 3 -> fails JSON -> _score_and_classify_rules [Deterministic]

rank_candidates_node:
  └── CandidateRankingAgent.rank_candidates(cond, candidates)
        ├── If LLM configured: LLM Call 4 (rank prompt)
        └── Fallback: _evaluate_candidates_deterministically [Deterministic]
```

### LLM Call Budget Analysis
- In environments without local GGUF weights, `RulePassThroughLLM` returns `""`, which causes `JSONDecodeError` exceptions and wasted retry loops in `analyze_context` and `classify_diagnoses`.
- **Target Budget**: 1 single structured extraction call when an LLM is active; 0 LLM calls required when deterministic fact extractor is active; 0 LLM calls for reasoning unless selective ambiguity short-circuiting triggers.

---

## 3. Current Database Call Graph

```text
Application Startup / First Request:
  └── get_or_initialize_retrieval_system()
        ├── ExcelWorkbookLoader.load("Database/Database_2.xlsx")
        │     └── Ingests 75,551 ICD-10-CM records (active_yesno == 1)
        │     └── Ingests 13,722 CPT records (segregated)
        ├── LocalICDCatalog(records)
        ├── BM25ICDRetriever.build_index(records)
        └── FAISSICDRetriever.build_index(records, FastLocalEmbeddings)
```

- **Database Invariance**: Database loading happens once under a threading lock and is cached as a singleton. No per-request re-indexing occurs.
- **CPT Firewall**: CPT codes are segregated by sheet and length; never retrieved in the ICD-10-CM diagnosis pipeline.

---

## 4. Current Retrieval Flow

```text
Classified Diagnosis (cond.raw_term)
       │
       ▼
HybridICDRetriever.retrieve(query=cond.raw_term, top_k=35)
       ├── BM25 query against all 75,551 records
       ├── FAISS query against all 75,551 embeddings
       └── Reciprocal Rank Fusion (RRF) -> top 35 candidates
       │
       ▼
Post-retrieval Regex Filtering (nodes.py:570-620)
       └── Ad-hoc keyword filters (e.g. is_neoplasm, is_procedure)
```

### Retrieval Flaw
- Retrieval is performed across the **entire 75k database** before any disease family filter is applied.
- Semantic vectors can pull noisy matches from unrelated chapters (e.g., pregnancy-related hypertension for essential hypertension, or lung neoplasm for pleural metastasis).

---

## 5. Current LangGraph Flow

Registered Nodes in [`workflow.py`](file:///c:/DS_and_AI/Projects_and_Tutorials/Projects/icd_project_dmh/src/medical_coding/graph/workflow.py):
1. `validate_document`
2. `extract_text`
3. `extract_diagnoses`
4. `analyze_context`
5. `classify_diagnoses`
6. `retrieve_candidates`
7. `rank_candidates`
8. `validate_codes`
9. `evaluate_confidence`
10. `finalize_output`

---

## 6. Redundant Operations Identified

1. **Repeated JSON Decode Failures**: `ContextAndRelevanceAgent` and `PrimarySecondaryClassifier` execute LLM generate calls on dummy LLMs that return empty strings, catch JSON decode errors, retry, and then fall back to deterministic evaluation.
2. **Double Candidate Attribute Validation**: `ReverseAttributeChecker` is executed inside `_evaluate_candidates_deterministically` in `ranker.py`, and then executed again inside `validate_codes_node` in `deterministic.py`.
3. **Repeated Term Tokenization**: Each condition is re-tokenized in extraction, context assessment, ranking, and validation without sharing the parsed morphological token set.

---

## 7. Mock / Fake Operations Audit

- **Grep Search Results**: Zero mocks found in `src/`. No `unittest.mock` or `MagicMock` in the production pipeline.
- Database records are 100% real rows loaded from `Database/Database_2.xlsx`.
- Embeddings are generated locally via `FastLocalEmbeddings` (hashing vectorizer) or model embeddings.

---

## 8. Monkey Patches Audit

- Zero dynamic monkey patches found.
- Zero case-specific checks (`case_id == ...` or `document_id == ...`).
- All clinical logic is generalized.

---

## 9. Latency Bottlenecks

1. **Unconstrained 75k Search Space**: Querying 35 candidates across all 75,551 codes for every condition causes unnecessary matrix operations in FAISS and BM25.
2. **Retries on Empty LLM Responses**: Wasted 50-80ms per document attempting to parse empty string responses from pass-through models.
3. **Missing Granular Telemetry**: Pipeline only records total processing time; lacks stage-by-stage millisecond breakdown (`parsing_ms`, `extraction_ms`, `reasoning_ms`, `retrieval_ms`, `validation_ms`, `serialization_ms`).

---

## 10. Missing Reasoning Components

1. **`ClinicalConcept` Model**: Missing a formal typed representation linking canonical disease family, body site, laterality, metastatic direction, and supported vs unknown attributes.
2. **`MatchSpec` Builder**: Missing deterministic constraints (`allowed_code_families`, `required_attributes`, `forbidden_attributes`, `unknown_attributes`).
3. **Hard Disease Family Filtering**: Retrieval does not pre-filter candidates by code family prefixes (`C50.*`, `C83.*`, `N20.*`, `K29.*`, `S82.*`, `I10`, `E11.*`, etc.) before vector/BM25 scoring.
4. **Compatibility Reasoner**: Missing an explicit post-retrieval validation node that verifies semantic equivalence, required attributes, metastatic direction, and absence of forbidden attributes.

---

## 11. Proposed Minimal Architecture Delta

1. **Schema Layer (`schemas/reasoning.py`)**:
   - Define `ClinicalConcept`, `MatchSpec`, and `CompatibilityResult`.
2. **Clinical Concept Reasoner (`reasoning/concept_reasoner.py`)**:
   - Deterministically infers `disease_family`, `body_site`, `laterality`, `histology`, `metastatic_status`, and attributes from validated candidates.
3. **MatchSpec Builder (`reasoning/match_spec.py`)**:
   - Builds `MatchSpec` containing `allowed_code_families`, `required_attributes`, `forbidden_attributes`, and `unknown_attributes`.
4. **Family-Filtered Staged Retrieval**:
   - Constrain FAISS and BM25 search space to candidate subsets matching `allowed_code_families`.
5. **Compatibility Reasoner (`reasoning/compatibility.py`)**:
   - Validates retrieved codes against `ClinicalConcept` and `MatchSpec`.
   - Rejects forbidden attributes (e.g., `C79.81` secondary breast malignancy when breast is primary site).
6. **Subsecond Latency Instrumentation**:
   - Measure and record granular millisecond metrics for all pipeline stages.
7. **Eliminate Dummy LLM Retry Loops**:
   - Short-circuit immediately to deterministic evaluation when LLM is unavailable or unconfigured.
