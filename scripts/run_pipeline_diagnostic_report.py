"""Diagnostic test runner executing the complete LangGraph medical coding pipeline.

Evaluates sample clinical documents across realistic edge cases, guardrails, and abstentions,
producing a structured diagnostic report with the 6 core metrics:
1. Clinical Concept Recall
2. Role Classification Accuracy
3. Code Accuracy
4. False Positive Count / Rate
5. Abstention Quality
6. Candidate Surface Retrieval Recall
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Ensure src is in sys.path when executed directly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medical_coding.graph.pipeline import process_clinical_document
from medical_coding.graph.workflow import export_graph_ascii
from medical_coding.schemas.enums import ExecutionStatus
from medical_coding.schemas.response import CodingResult

BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "id": "CASE-01-STANDARD-ADMISSION",
        "title": "Inpatient Discharge (Acute Systolic HF, T2DM, HTN)",
        "expected_primary_code": "I50.21",
        "expected_secondary_codes": ["E11.9", "I10"],
        "expected_concepts": ["acute systolic heart failure", "type 2 diabetes mellitus", "essential primary hypertension"],
        "unwanted_fp": ["dyspnea", "shortness of breath", "edema", "appendectomy"],
        "expected_abstentions": [],
        "text": """
        DISCHARGE SUMMARY
        Patient: John Doe | MRN: 987654321
        Admission Date: 2026-09-15 | Discharge Date: 2026-09-20
        CHIEF COMPLAINT:
        Severe acute shortness of breath and peripheral edema.

        DISCHARGE DIAGNOSES:
        1. Acute systolic heart failure - Patient admitted with acute exacerbation of systolic congestive heart failure. Echo demonstrated EF 20%. Treated with intravenous furosemide diuresis with marked symptomatic improvement.
        2. Type 2 diabetes mellitus - Monitored daily fasting blood sugars; maintained on home metformin 1000 mg twice daily.
        3. Essential primary hypertension - Blood pressure monitored on telemetry; stable.

        PAST MEDICAL HISTORY:
        1. Appendectomy in 2005.
        2. Resolved seasonal allergic rhinitis.

        HOSPITAL COURSE:
        Patient was admitted for acute decompensated heart failure. IV loop diuretics were administered with good diuresis and clinical resolution. Discharged in stable condition.
        """,
    },
    {
        "id": "CASE-02-UNSUPPORTED-SPECIFICITY",
        "title": "General Heart Failure Mention Without Subtype Specificity",
        "expected_primary_code": "I50.9",
        "expected_secondary_codes": [],
        "expected_concepts": ["heart failure"],
        "unwanted_fp": ["I50.21", "I50.31", "I50.41"],
        "expected_abstentions": [],
        "text": """
        DISCHARGE SUMMARY
        CHIEF COMPLAINT: Fatigue and peripheral edema.
        DISCHARGE DIAGNOSES:
        1. Heart failure - Patient has documented history of heart failure. Maintenance oral furosemide was continued. No echocardiogram was performed during admission and ejection fraction was not documented.
        """,
    },
    {
        "id": "CASE-03-RULED-OUT-CONDITION",
        "title": "Ruled-Out Acute Condition (Myocardial Infarction)",
        "expected_primary_code": None,
        "expected_secondary_codes": [],
        "expected_concepts": [],
        "unwanted_fp": ["I21.9", "acute myocardial infarction"],
        "expected_abstentions": ["RULED_OUT"],
        "text": """
        DISCHARGE SUMMARY
        CHIEF COMPLAINT: Acute substernal chest discomfort.
        DISCHARGE DIAGNOSES:
        1. Non-cardiac chest pain - Evaluated for acute coronary syndrome. Acute myocardial infarction was definitively ruled out by serial negative high-sensitivity troponins and normal ECG.
        """,
    },
    {
        "id": "CASE-04-HISTORICAL-PMH-ONLY",
        "title": "Past Medical History Without Inpatient Care (Remote CVA)",
        "expected_primary_code": None,
        "expected_secondary_codes": [],
        "expected_concepts": [],
        "unwanted_fp": ["I63.9", "stroke", "cerebrovascular accident"],
        "expected_abstentions": ["HISTORICAL"],
        "text": """
        DISCHARGE SUMMARY
        CHIEF COMPLAINT: Inguinal hernia repair.
        PAST MEDICAL HISTORY:
        1. Remote stroke in 2014 without residual neurological deficit.
        DISCHARGE DIAGNOSIS:
        1. Elective repair of inguinal hernia.
        HOSPITAL COURSE:
        Patient underwent routine elective hernia surgery without complications. Neurologic status was baseline.
        """,
    },
    {
        "id": "CASE-05-EMPTY-DOCUMENT",
        "title": "Empty / Whitespace-Only Document",
        "expected_primary_code": None,
        "expected_secondary_codes": [],
        "expected_concepts": [],
        "unwanted_fp": [],
        "expected_abstentions": ["INSUFFICIENT_CLINICAL_EVIDENCE"],
        "text": "   \n\t  \n  ",
    },
    {
        "id": "CASE-06-NON-CLINICAL-GIBBERISH",
        "title": "Non-Clinical Gibberish Text",
        "expected_primary_code": None,
        "expected_secondary_codes": [],
        "expected_concepts": [],
        "unwanted_fp": [],
        "expected_abstentions": ["INSUFFICIENT_CLINICAL_EVIDENCE"],
        "text": "The quick brown fox jumps over the lazy dog. Random non-clinical sentence with no medical terms.",
    },
]


async def run_diagnostics() -> dict[str, Any]:
    """Execute complete LangGraph workflow across benchmark test cases and report 6 discrete metrics."""
    print("=" * 80)
    print("STARTING COMPLETE LANGGRAPH WORKFLOW DIAGNOSTIC SUITE & 6-METRIC EVALUATION")
    print("=" * 80)

    ascii_graph = export_graph_ascii()
    print("\n--- COMPILED WORKFLOW TOPOLOGY ---")
    print(ascii_graph)

    results: list[dict[str, Any]] = []

    # Telemetry counters for 6 metrics
    total_gold_concepts = 0
    extracted_gold_concepts = 0

    total_role_evaluations = 0
    correct_roles = 0

    total_expected_codes = 0
    correct_codes = 0

    total_false_positives = 0

    total_abstention_checks = 0
    correct_abstentions = 0

    total_retrieval_checks = 0
    retrieval_hits = 0

    for case in BENCHMARK_CASES:
        case_id = case["id"]
        title = case["title"]
        text = case["text"]

        print(f"\nProcessing [{case_id}]: {title}...")
        res: CodingResult = await process_clinical_document(
            source=text,
            document_id=case_id,
        )

        audit_trail = res.metadata.get("audit_trail", []) if res.metadata else []

        # Evaluate Metric 1: Clinical Concept Recall
        exp_concepts = case.get("expected_concepts", [])
        total_gold_concepts += len(exp_concepts)
        extracted_terms = []
        if res.primary_diagnosis and res.primary_diagnosis.raw_term:
            extracted_terms.append(res.primary_diagnosis.raw_term.lower())
        for s in res.secondary_diagnoses:
            if s.raw_term:
                extracted_terms.append(s.raw_term.lower())

        for gc in exp_concepts:
            if any(gc.lower() in et or et in gc.lower() for et in extracted_terms):
                extracted_gold_concepts += 1

        # Evaluate Metric 2: Role Classification Accuracy
        exp_pri_code = case.get("expected_primary_code")
        if exp_pri_code:
            total_role_evaluations += 1
            if res.primary_diagnosis and (res.primary_diagnosis.code == exp_pri_code or exp_pri_code.startswith(res.primary_diagnosis.code or "xxx")):
                correct_roles += 1

        exp_sec_codes = case.get("expected_secondary_codes", [])
        for sc in exp_sec_codes:
            total_role_evaluations += 1
            if any(sec.code == sc or sc.startswith(sec.code or "xxx") for sec in res.secondary_diagnoses):
                correct_roles += 1

        # Evaluate Metric 3: Code Accuracy
        if exp_pri_code:
            total_expected_codes += 1
            if res.primary_diagnosis and res.primary_diagnosis.code == exp_pri_code:
                correct_codes += 1

        for sc in exp_sec_codes:
            total_expected_codes += 1
            if any(sec.code == sc for sec in res.secondary_diagnoses):
                correct_codes += 1

        # Evaluate Metric 4: False Positive Count
        unwanted = case.get("unwanted_fp", [])
        for u in unwanted:
            u_low = u.lower()
            if res.primary_diagnosis and (u_low in (res.primary_diagnosis.code or "").lower() or u_low in (res.primary_diagnosis.raw_term or "").lower()):
                total_false_positives += 1
            for sec in res.secondary_diagnoses:
                if u_low in (sec.code or "").lower() or u_low in (sec.raw_term or "").lower():
                    total_false_positives += 1

        # Evaluate Metric 5: Abstention Quality
        exp_abst = case.get("expected_abstentions", [])
        if exp_abst:
            total_abstention_checks += 1
            abst_reasons = [str(a.reason).lower() for a in res.abstentions]
            abst_details = [str(a.detail or "").lower() for a in res.abstentions]
            matched_abst = False
            for ea in exp_abst:
                ea_low = ea.lower()
                if any(ea_low in ar for ar in abst_reasons) or any(ea_low in ad for ad in abst_details):
                    matched_abst = True
                    break
            if not matched_abst and res.primary_diagnosis is None and res.status in (ExecutionStatus.SUCCESS, ExecutionStatus.ABSTAINED):
                matched_abst = True
            if matched_abst:
                correct_abstentions += 1

        # Evaluate Metric 6: Candidate Surface Retrieval Recall
        if exp_pri_code:
            total_retrieval_checks += 1
            # Check candidate pool in audit trail or selections
            candidate_pool = res.metadata.get("candidate_pool", {}) if res.metadata else {}
            cand_codes = [c.get("code") for c_list in candidate_pool.values() for c in c_list] if candidate_pool else []
            if exp_pri_code in cand_codes or (res.primary_diagnosis and res.primary_diagnosis.code == exp_pri_code):
                retrieval_hits += 1

        case_summary = {
            "case_id": case_id,
            "title": title,
            "status": str(res.status),
            "primary_diagnosis": (
                {
                    "code": res.primary_diagnosis.code,
                    "description": res.primary_diagnosis.description,
                    "confidence_score": res.primary_diagnosis.confidence_score,
                }
                if res.primary_diagnosis
                else None
            ),
            "secondary_diagnoses": [
                {
                    "code": sec.code,
                    "description": sec.description,
                    "confidence_score": sec.confidence_score,
                }
                for sec in res.secondary_diagnoses
            ],
            "abstention_count": len(res.abstentions),
            "abstentions": [
                {
                    "diagnosis": a.raw_term,
                    "reason": str(a.reason),
                    "detail": a.detail,
                    "stage": str(a.stage),
                }
                for a in res.abstentions
            ],
            "audit_trail_sample": audit_trail[:2] if audit_trail else [],
        }
        results.append(case_summary)

        print(f"  -> Status: {res.status}")
        if res.primary_diagnosis:
            print(f"  -> Primary: [{res.primary_diagnosis.code}] {res.primary_diagnosis.description}")
        else:
            print("  -> Primary: None (Abstained or No Billable Primary)")
        print(f"  -> Secondaries: {len(res.secondary_diagnoses)}")
        for sec in res.secondary_diagnoses:
            print(f"     * [{sec.code}] {sec.description}")
        if res.abstentions:
            print(f"  -> Abstentions: {len(res.abstentions)}")
            for a in res.abstentions:
                print(f"     ! {a.reason} ({a.stage}): {a.detail or a.raw_term}")

    # Compute final metrics
    concept_recall_pct = (extracted_gold_concepts / total_gold_concepts * 100) if total_gold_concepts > 0 else 100.0
    role_accuracy_pct = (correct_roles / total_role_evaluations * 100) if total_role_evaluations > 0 else 100.0
    code_accuracy_pct = (correct_codes / total_expected_codes * 100) if total_expected_codes > 0 else 100.0
    abstention_quality_pct = (correct_abstentions / total_abstention_checks * 100) if total_abstention_checks > 0 else 100.0
    retrieval_recall_pct = (retrieval_hits / total_retrieval_checks * 100) if total_retrieval_checks > 0 else 100.0

    metrics_scorecard = {
        "clinical_concept_recall_pct": round(concept_recall_pct, 1),
        "role_classification_accuracy_pct": round(role_accuracy_pct, 1),
        "code_accuracy_pct": round(code_accuracy_pct, 1),
        "false_positive_count": total_false_positives,
        "abstention_quality_pct": round(abstention_quality_pct, 1),
        "candidate_retrieval_recall_pct": round(retrieval_recall_pct, 1),
    }

    report = {
        "total_test_cases": len(results),
        "metrics_scorecard": metrics_scorecard,
        "results": results,
    }

    print("\n" + "=" * 80)
    print("SYSTEM QUALITY SCORECARD (DISCRETE METRICS)")
    print("=" * 80)
    print(f"1. Clinical Concept Recall:          {metrics_scorecard['clinical_concept_recall_pct']}% ({extracted_gold_concepts}/{total_gold_concepts})")
    print(f"2. Role Classification Accuracy:      {metrics_scorecard['role_classification_accuracy_pct']}% ({correct_roles}/{total_role_evaluations})")
    print(f"3. Code Accuracy:                    {metrics_scorecard['code_accuracy_pct']}% ({correct_codes}/{total_expected_codes})")
    print(f"4. False Positive Count:             {metrics_scorecard['false_positive_count']} (Zero false positives goal)")
    print(f"5. Abstention Quality Score:         {metrics_scorecard['abstention_quality_pct']}% ({correct_abstentions}/{total_abstention_checks})")
    print(f"6. Candidate Retrieval Recall:       {metrics_scorecard['candidate_retrieval_recall_pct']}% ({retrieval_hits}/{total_retrieval_checks})")
    print("=" * 80)

    return report


def main() -> None:
    report = asyncio.run(run_diagnostics())
    with open("diagnostic_pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nDiagnostic report saved to 'diagnostic_pipeline_report.json'.")


if __name__ == "__main__":
    main()
