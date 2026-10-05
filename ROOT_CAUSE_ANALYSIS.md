# Root Cause Analysis: ICD-10-CM Extraction, Classification & Coding Pipeline

**Document Version**: 1.0  
**Audit & Analysis Date**: October 2026  
**Investigator**: Senior Clinical NLP Architect & Systems Engineer  
**Branch**: `recall-fix`  
**Target Repository**: `src/medical_coding`  
**Execution Environment**: Python 3.13.13 (Offline / Air-Gapped)

---

## 1. Executive Summary

A comprehensive, line-by-line runtime and static trace of the medical coding pipeline was executed across unit, integration, and end-to-end regression suites. The test suite execution revealed **7 critical test failures** stemming from **8 core architectural and algorithmic root causes**.

The failures demonstrate that although previous attempts introduced `EvidenceFirstFactExtractor`, critical flaws remain in:
1. Narrative parsing and prefix stripping.
2. Section segmentation handling of unstructured or generic notes.
3. String normalization in deterministic context evaluation (causing past medical history to leak into active billable secondaries).
4. Fragile string matching for UHDDS primary diagnosis tie-breaking.
5. Specificity realignment for unspecified clinical mentions.
6. Over-broad procedural corroboration logic.

None of these issues require LLM hallucinations or external APIs to fix. All 8 root causes are deterministic flaws in clinical text processing, state routing, and rule evaluation.

---

## 2. Regression Test Failure Matrix

The following table summarizes the 7 test failures observed during the audit run (`pytest -q`):

| # | Test Identifier | Source File:Line | Failure Type | Root Cause ID | Summary of Failure |
|---|---|---|---|---|---|
| 1 | `test_code_pdf_success_endpoint` | `tests/test_api.py:60` | `AssertionError` (`'ABSTAINED' in ('SUCCESS', 'PARTIAL_SUCCESS')`) | **RC-2** | PDF containing short discharge summary without explicit section headers is classified as `GENERAL`, resulting in 0 extracted conditions and pipeline abort. |
| 2 | `test_scenario_4_two_plausible_primary_conditions_abstains` | `tests/test_classification.py:215` | `AssertionError` (`assert True is False` on `has_unique_primary`) | **RC-4** | Two equally qualifying emergent conditions (AMI & Stroke) were not tied because the tie check strictly looked for the substring `"chief complaint"`, ignoring `"Reason for admission:"`. AMI was arbitrarily chosen as unique primary. |
| 3 | `test_end_to_end_pipeline_success` | `tests/test_end_to_end_pipeline.py:109` | `AssertionError` (`assert not any("append" in s.description.lower()...)`) | **RC-3** | Remote appendectomy from Past Medical History was coded as an active secondary diagnosis because `"past medical"` (space) did not match `"PAST_MEDICAL_HISTORY"` (underscore). |
| 4 | `test_guardrail_unsupported_specificity_realigned_or_abstained` | `tests/test_end_to_end_pipeline.py:201` | `AssertionError` (`assert '' in ('I50.9', 'I50')`) | **RC-5** | "Heart failure" without subtype was assigned an empty code (`icd10cm=None`) rather than being realigned to terminal unspecified code `I50.9`. |
| 5 | `test_diagnostic_failure_report_execution` | `tests/test_end_to_end_pipeline.py:312` | `AssertionError` (`assert 'ABSTAINED' in ('SUCCESS', 'PARTIAL_SUCCESS')`) | **RC-2** | "Valid Acute Systolic HF" with single-sentence narrative lacked section headers, extracted 0 conditions, and prematurely aborted as `ABSTAINED`. |
| 6 | `test_modality_1_pdf_end_to_end` | `tests/test_multimodal_e2e.py:62` | `AssertionError` (`assert 'I50.21' in all_codes`) | **RC-1** | Discharge diagnosis `1. Primary: Acute systolic heart failure (...)` was split at `: `, treating `"Primary"` as the term and discarding the clinical diagnosis as narrative. Retrieval searched for `"Primary"` and produced CPT/irrelevant codes. |
| 7 | `test_pipeline_execution_success_persistence` | `tests/test_pipeline_persistence.py:98` | `AssertionError` (`assert 'ABSTAINED' in ('SUCCESS', 'PARTIAL_SUCCESS')`) | **RC-2** | Single-sentence narrative note lacked section headers, resulting in 0 extracted conditions and pipeline abort. |

---

## 3. Exhaustive Root Cause Investigations

### RC-1: Diagnosis Prefix Label Absorption Discards Clinical Terms
- **Location**: `src/medical_coding/agents/clinical_extractor.py:243, 401-417` (`_split_term_and_narrative`)
- **Impacted Tests**: `test_modality_1_pdf_end_to_end` (and any discharge note formatting diagnoses as `1. Primary: ...` or `Principal: ...`)
- **Mechanism**:
  1. Clinicians frequently document discharge diagnoses with role prefixes:
     ```
     DISCHARGE DIAGNOSES:
     1. Primary: Acute systolic heart failure (decompensated congestive heart failure with systolic dysfunction).
     2. Essential primary hypertension (longstanding, maintained on amlodipine).
     ```
  2. In `_parse_diagnosis_items`, line numbers are stripped:
     `cleaned_line = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()`
     Yielding: `"Primary: Acute systolic heart failure (decompensated congestive heart failure with systolic dysfunction)."`
  3. `_split_term_and_narrative` evaluates delimiters in order:
     `[" - ", " – ", " — ", " : ", ": ", " / ", " (", "; "]`
  4. The delimiter `": "` matches at position 7.
  5. The method splits:
     - `term = "Primary"`
     - `narrative = "Acute systolic heart failure (decompensated congestive heart failure with systolic dysfunction)."`
  6. The candidate is created with `raw_term = "Primary"` and `normalized_diagnosis = "Primary"`.
  7. In Node 6, `HybridICDRetriever` searches for `"Primary"`, retrieving CPT/procedural codes (e.g. `77299`, `99285`, `Z91.81`), while `I50.21` is never considered.
- **Architectural Invariant Violated**:
  - *Clinical Entity Grounding*: Diagnosis extraction must extract clinical disease entities, not structural or typographical metadata labels.
- **Remediation**:
  Pre-process lines in `_parse_diagnosis_items` to strip leading structural labels:
  ```python
  cleaned_line = re.sub(r"^(?:primary|principal|secondary|final|admitting)\s*(?:diagnosis|condition)?\s*:\s*", "", cleaned_line, flags=re.IGNORECASE).strip()
  ```

---

### RC-2: SectionSegmenter Omits Unstructured Text and "GENERAL" Section Handling
- **Location**: `src/medical_coding/agents/clinical_extractor.py:141-150, 197-227` (`SectionSegmenter.segment`, `_extract_candidates_from_sections`)
- **Impacted Tests**: `test_code_pdf_success_endpoint`, `test_diagnostic_failure_report_execution`, `test_pipeline_execution_success_persistence`
- **Mechanism**:
  1. When a clinical note does not contain canonical section headers (e.g., `"DISCHARGE SUMMARY\nPatient admitted with acute systolic heart failure."` or `"Patient admitted with acute systolic congestive heart failure and bilateral lower extremity edema."`), `SectionSegmenter.segment` finds no matches and returns a single section:
     `section_name="GENERAL", raw_header="DOCUMENT", content=text.strip()`
  2. `EvidenceFirstFactExtractor._extract_candidates_from_sections` only parses sections matching:
     - `("DISCHARGE_DIAGNOSES", "PRINCIPAL_DIAGNOSIS", "SECONDARY_DIAGNOSES", "ASSESSMENT_PLAN")`
     - `"CHIEF_COMPLAINT"`
     - `"PAST_MEDICAL_HISTORY"`
  3. Sections with name `"GENERAL"` or `"DISCHARGE_SUMMARY"` are completely skipped.
  4. `candidates` returns empty `[]`.
  5. In Node 3 (`nodes.py:413`), when `not deduplicated`, an `AbstentionRecord(reason=INSUFFICIENT_CLINICAL_EVIDENCE)` is emitted and `is_aborted = True`.
  6. The pipeline aborts with `ExecutionStatus.ABSTAINED`, failing all short-form and unstructured note tests.
- **Architectural Invariant Violated**:
  - *Generalizability & Robustness*: The pipeline must support unstructured, brief, or emergency clinical encounter text without requiring rigid section formatting.
- **Remediation**:
  In `_extract_candidates_from_sections`, add a fallback parsing pass for sections named `"GENERAL"`, `"DISCHARGE_SUMMARY"`, or `"UNSTRUCTURED"`. Split sentences and extract clinical condition assertions using clinical grammatical heuristics (e.g. phrases following "admitted with", "diagnosed with", "impression:", "presents with").

---

### RC-3: String Delimiter Mismatch in Deterministic Past Medical History Assessment
- **Location**: `src/medical_coding/graph/nodes.py:1004` (`_assess_context_deterministically`)
- **Impacted Tests**: `test_end_to_end_pipeline_success`
- **Mechanism**:
  1. `SectionSegmenter` extracts Past Medical History with:
     `section="PAST_MEDICAL_HISTORY"`
  2. In Node 3, conditions are extracted and passed to Node 4 (`_assess_context_deterministically` in fallback mode).
  3. `_assess_context_deterministically` checks:
     ```python
     section_lower = section.lower()  # "past_medical_history"
     is_pmh = "past medical" in section_lower or "pmh" in section_lower
     ```
  4. The string `"past medical"` contains a space, whereas `section_lower` contains underscores (`"past_medical_history"`).
  5. `"past medical" in "past_medical_history"` evaluates to `False`. `"pmh" in "past_medical_history"` evaluates to `False`.
  6. Therefore, `is_pmh` is `False`.
  7. The condition falls through to `Rule 3: Active condition`:
     ```python
     current_relevance = True
     coding_candidate = True
     status = ConditionStatus.ACTIVE
     ```
  8. Historical conditions (e.g., "Remote appendectomy in 2005") are marked as active candidates and flow into candidate ranking and validation.
  9. Validation Gate 3 fails to filter it, and "Remote appendectomy" is coded as an active secondary diagnosis, directly violating Guardrail 5 and the test assertion:
     `assert not any("append" in s.description.lower() for s in result.secondary_diagnoses)`
- **Architectural Invariant Violated**:
  - *CMS Coding Guideline IV.J & Guardrail 5*: Previous conditions that have no bearing on the current hospital stay must not be coded as active secondary diagnoses.
- **Remediation**:
  Normalize `section_lower` by replacing underscores with spaces or check normalized tokens:
  ```python
  norm_sec = section_lower.replace("_", " ")
  is_pmh = "past medical" in norm_sec or "pmh" in norm_sec or "history" in norm_sec
  ```

---

### RC-4: Brittle Literal String Requirement in UHDDS Primary Diagnosis Tie-Breaking
- **Location**: `src/medical_coding/agents/classifier.py:1031-1036` (`PrimarySecondaryClassifier._classify_by_rules`)
- **Impacted Tests**: `test_scenario_4_two_plausible_primary_conditions_abstains`
- **Mechanism**:
  1. In `test_scenario_4_two_plausible_primary_conditions_abstains`, two emergent conditions (Acute myocardial infarction and Acute ischemic stroke) both present as equal reasons for admission and receive emergent procedural interventions.
  2. Both conditions obtain equal admitting scores ($\ge 4.0$).
  3. The tie-breaking logic in `_classify_by_rules` states:
     ```python
     if top_score >= 4.0 and second_score >= 4.0 and abs(top_score - second_score) < 0.01:
         top_ctx = f"{top_asm.evidence} {top_asm.reason}".lower()
         sec_ctx = f"{second_asm.evidence} {second_asm.reason}".lower()
         if "chief complaint" in top_ctx and "chief complaint" in sec_ctx:
             has_ambiguous_tie = True
     ```
  4. The test fixture provides evidence formatted as:
     `"Reason for admission: Acute myocardial infarction, emergent catheterization and stent placed."`
     `"Reason for admission: Acute ischemic stroke, emergent mechanical thrombectomy performed."`
  5. The literal string `"chief complaint"` is absent from both `top_ctx` and `sec_ctx`.
  6. `has_ambiguous_tie` remains `False`.
  7. The classifier arbitrarily selects the first candidate (`ami-01`) as unique primary:
     `result.has_unique_primary = True`
  8. The test assertion `assert result.has_unique_primary is False` fails.
- **Architectural Invariant Violated**:
  - *UHDDS Rule on Equal Plausible Primaries*: When two or more conditions equally meet the criteria for principal diagnosis and neither the alphabetic index nor the tabular list directs otherwise, any may be sequenced first, but an autonomous coding system must flag the ambiguity rather than making an arbitrary decision without physician query.
- **Remediation**:
  Expand the tie-breaking context detection to recognize clinical synonyms for admission reason:
  ```python
  admission_cues = ["chief complaint", "reason for admission", "admitted for", "presenting complaint", "occasioning admission", "emergent"]
  top_has_adm = any(cue in top_ctx for cue in admission_cues)
  sec_has_adm = any(cue in sec_ctx for cue in admission_cues)
  if top_has_adm and sec_has_adm:
      has_ambiguous_tie = True
  ```

---

### RC-5: Unspecified Code Assignment Failure for Documented General Conditions
- **Location**: `src/medical_coding/graph/nodes.py:842-896` (`finalize_output_node`), `src/medical_coding/validation/deterministic.py:488-510`
- **Impacted Tests**: `test_guardrail_unsupported_specificity_realigned_or_abstained`
- **Mechanism**:
  1. The test input documents only:
     `"DISCHARGE DIAGNOSES: 1. Heart failure. Patient was monitored on oral diuretic."`
  2. The clinical extraction accurately extracts `"Heart failure"` without hallucinating subtype (systolic/diastolic).
  3. In candidate retrieval, `I50.9` (Heart failure, unspecified) and `I50` (category header, non-billable) are retrieved.
  4. The candidate ranking / validation node evaluates specificity:
     - If `I50` is evaluated, it is rejected because `is_valid_billable = False`.
     - In `validate_specificity_alignment`, if candidate description is compared against "Heart failure", or if confidence evaluation marks mapping as abstained, `validated` receives no primary entry with a code.
  5. In `finalize_output_node`:
     Because no validated ICD candidate was approved, it creates a `CodedDiagnosisResponse` with `icd10cm=None`.
  6. The test asserts:
     `assert result.primary_diagnosis.code in ("I50.9", "I50")`
     `result.primary_diagnosis.code` returns `""` because `self.icd10cm` is `None`.
- **Architectural Invariant Violated**:
  - *Guardrail 11 & Specificity Rule*: When a condition is documented without clinical specification, the system must code to the highest level of specificity supported by the documentation (in this case, terminal unspecified code `I50.9`), rather than failing code assignment completely.
- **Remediation**:
  Ensure that when a clinically confirmed condition has no documented subtype, the candidate ranker selects the authoritative terminal unspecified billable leaf code (`I50.9`) from the candidate pool and validates it through Gate 6.

---

### RC-6: Over-Broad Procedure Corroboration Globally Inflates Admitting Scores
- **Location**: `src/medical_coding/agents/clinical_extractor.py:499-506` (`_corroborate_with_hospital_course_and_procedures`)
- **Mechanism**:
  1. In `_corroborate_with_hospital_course_and_procedures`:
     ```python
     for cand in candidates:
         ...
         # Check procedure text
         if procedure_text:
             for proc_cue in PROCEDURE_SOURCE_CONTROL_CUES:
                 if re.search(proc_cue, procedure_text, re.IGNORECASE):
                     cand.procedure_relevance = 1.0
                     cand.scores.admitting_score += 3.0
                     break
     ```
  2. Notice that `procedure_text` is searched for `proc_cue`, completely detached from `cand`!
  3. If a patient had "DJ stenting" for acute pyelonephritis, but also had chronic hypertension, chronic diabetes, and remote allergic rhinitis, **all four candidates** received `procedure_relevance = 1.0` and `admitting_score += 3.0`.
  4. This invalidates the discriminating power of procedural source control in determining the primary diagnosis over co-morbidities.
- **Architectural Invariant Violated**:
  - *Evidence-Entity Association*: Supporting procedural evidence must be linked specifically to the condition it was performed to treat.
- **Remediation**:
  Condition procedural corroboration on lexical or anatomical overlap between the candidate's diagnostic concept (or anatomical site) and the procedure description.

---

### RC-7: Sentence-as-Diagnosis Extraction Pollutes Candidate Pool
- **Location**: `src/medical_coding/agents/clinical_extractor.py:305-351` (`_parse_complaint_items`)
- **Mechanism**:
  1. In `_parse_complaint_items`:
     `sentences = [s.strip() for s in re.split(r"[\.\n;]+", cleaned) if s.strip()]`
     `cand = ClinicalDiagnosisCandidate(raw_term=s, normalized_diagnosis=s, role=DiagnosisRole.PRIMARY, ...)`
  2. For a chief complaint such as:
     `CHIEF COMPLAINT: Acute shortness of breath and fluid overload.`
     The entire sentence `"Acute shortness of breath and fluid overload"` is generated as a diagnosis candidate.
  3. This symptom phrase competes with the definitive discharge diagnosis `"Acute systolic heart failure"`, leading to retrieval pollution and potential primary diagnosis confusion.
- **Architectural Invariant Violated**:
  - *ICD-10-CM Guideline IV.I*: Signs and symptoms that are routinely associated with a disease process should not be assigned as additional codes or primary diagnoses when the definitive condition has been established.
- **Remediation**:
  Tag chief complaint items with `entity_type = SYMPTOM` or `evidence_type = ADMISSION_REASON`, and ensure they serve only to corroborate discharge diagnoses unless no definitive discharge diagnosis was documented.

---

### RC-8: Dual-Pipeline Disconnect Between Extracted Entities and Clinical Diagnosis State
- **Location**: `src/medical_coding/graph/nodes.py:340-437`, `src/medical_coding/graph/nodes.py:932-968`
- **Mechanism**:
  1. The pipeline maintains parallel data representations: `PipelineGraphState` (`extracted_conditions`, `context_assessments`, `classified_diagnoses`) and `ClinicalDiagnosisState` (`all_candidates`, `primary_diagnosis`, `secondary_diagnoses`).
  2. In `_extract_conditions_deterministically`, `ClinicalDiagnosisState` is generated by `EvidenceFirstFactExtractor`, but then unpacked into a flat list of `ExtractedClinicalCondition` objects where rich scores and evidence connections are discarded.
  3. Subsequent nodes (Node 4, Node 5) re-evaluate context and classification from scratch using separate rule functions (`_assess_context_deterministically`, `PrimarySecondaryClassifier._classify_by_rules`), causing logic discrepancies between the extractor's classification and Node 5's classification.
- **Architectural Invariant Violated**:
  - *Single Source of Clinical Truth*: Clinical assessment state must be unified and preserved through the graph without contradictory parallel re-evaluations.
- **Remediation**:
  Propagate `ClinicalDiagnosisState` directly through the graph state, using Node 4 and Node 5 to validate, refine, and enforce gates on the established state.

---

## 4. Invariant Verification Checklist

| Invariant Rule | Current Status | Post-Fix Guarantee |
|---|:---:|:---|
| **Diagnosis Extraction Precedes ICD Retrieval** | **PASS** | Fact extraction in Node 3 operates solely on source text without ICD lookup. |
| **Retrieval Never Creates Diagnoses** | **PASS** | Node 6 queries catalog solely for authorized clinical conditions. |
| **Output Specificity $\le$ Source Evidence** | **FAIL** (RC-5) | Realign unspecified mentions to valid terminal unspecified codes (`I50.9`). |
| **Maximum One Primary Diagnosis** | **FAIL** (RC-4) | Enforce unique primary, or abstain when multiple plausible conditions tie. |
| **Historical Conditions Excluded from Billing** | **FAIL** (RC-3) | Fix PMH section normalization so resolved/past conditions are strictly non-billable. |
| **No Hardcoded Disease Whitelists** | **PASS** | Fact extraction operates generically via section segmentation and clinical grammar. |
| **No File-Specific or Case-Specific Hacks** | **PASS** | All logic changes are general and test-agnostic. |

---

## 5. Remediation Plan

The remediation will be executed according to the target architecture in three discrete steps:
1. **Clinical Fact Extraction & Section Segmenter Healing**:
   - Strip diagnosis role prefixes (`Primary:`, `Secondary:`) in `clinical_extractor.py`.
   - Add unstructured narrative fallback in `SectionSegmenter` and `EvidenceFirstFactExtractor`.
   - Constrain procedural corroboration to relevant candidates.
   - Demote chief complaint symptom sentences when definitive diagnoses exist.
2. **Context & Classification Rule Normalization**:
   - Fix PMH section normalization in `nodes.py:1004` to recognize `PAST_MEDICAL_HISTORY`.
   - Expand UHDDS admission reason cues in `classifier.py:1031` for robust tie-breaking.
3. **ICD Specificity & Terminal Leaf Assignment**:
   - Ensure `validate_specificity_alignment` and candidate ranking assign terminal unspecified leaf codes (`I50.9`) to unelaborated conditions.
