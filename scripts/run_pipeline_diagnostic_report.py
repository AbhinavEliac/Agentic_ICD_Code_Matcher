"""Diagnostic test runner executing the complete LangGraph medical coding pipeline.

Evaluates sample clinical documents across realistic edge cases, guardrails, and abstentions,
producing a structured diagnostic failure report with internal audit trails.
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
from medical_coding.schemas.response import CodingResult

DIAGNOSTIC_CASES: list[dict[str, str]] = [
    {
        "id": "CASE-01-STANDARD-ADMISSION",
        "title": "Comprehensive Inpatient Discharge (Acute Systolic HF, T2DM, HTN)",
        "expected": "Success with Primary I50.21, Secondaries E11.9, I10. Historical PMH excluded.",
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
        "expected": "Guardrail 11: Candidate ranking must align with I50.9 (unspecified) or abstain; never force I50.21.",
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
        "expected": "Guardrail 4: Ruled-out MI cannot be coded as confirmed or primary; triggers abstention or exclusion.",
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
        "expected": "Guardrail 5: Historical-only conditions cannot automatically become coded secondary diagnoses.",
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
        "expected": "Guardrail 12: Missing/insufficient text causes explicit document-level abstention/abortion.",
        "text": "   \n\t  \n  ",
    },
    {
        "id": "CASE-06-NON-CLINICAL-GIBBERISH",
        "title": "Non-Clinical Gibberish Text",
        "expected": "Guardrail 12: Insufficient clinical evidence triggers graceful extraction abstention.",
        "text": "The quick brown fox jumps over the lazy dog. Random non-clinical sentence with no medical terms.",
    },
]


async def run_diagnostics() -> dict[str, Any]:
    """Execute complete LangGraph workflow across all diagnostic test cases."""
    print("=" * 80)
    print("STARTING COMPLETE LANGGRAPH WORKFLOW DIAGNOSTIC SUITE")
    print("=" * 80)

    # Topology
    ascii_graph = export_graph_ascii()
    print("\n--- COMPILED WORKFLOW TOPOLOGY ---")
    print(ascii_graph)

    results: list[dict[str, Any]] = []

    for case in DIAGNOSTIC_CASES:
        case_id = case["id"]
        title = case["title"]
        text = case["text"]
        expected = case["expected"]

        print(f"\nProcessing [{case_id}]: {title}...")
        res: CodingResult = await process_clinical_document(
            source=text,
            document_id=case_id,
        )

        audit_trail = res.metadata.get("audit_trail", []) if res.metadata else []

        case_summary = {
            "case_id": case_id,
            "title": title,
            "expected_behavior": expected,
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
            print(
                f"  -> Primary: [{res.primary_diagnosis.code}] {res.primary_diagnosis.description}"
            )
        else:
            print("  -> Primary: None (Abstained or No Billable Primary)")
        print(f"  -> Secondaries: {len(res.secondary_diagnoses)}")
        for sec in res.secondary_diagnoses:
            print(f"     * [{sec.code}] {sec.description}")
        if res.abstentions:
            print(f"  -> Abstentions/Guardrail Triggers: {len(res.abstentions)}")
            for a in res.abstentions:
                print(f"     ! {a.reason} ({a.stage}): {a.detail or a.raw_term}")

    # Produce failure diagnostic summary
    failures_and_abstentions = [
        r for r in results if r["status"] != "ExecutionStatus.SUCCESS" or r["abstention_count"] > 0
    ]

    report = {
        "total_test_cases": len(results),
        "cases_with_abstentions_or_guarded_failures": len(failures_and_abstentions),
        "results": results,
    }

    print("\n" + "=" * 80)
    print("DIAGNOSTIC FAILURE & ABSTENTION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Test Cases Evaluated: {len(results)}")
    print(f"Cases with Enforced Guardrails / Abstentions: {len(failures_and_abstentions)}")
    for item in results:
        flag = "GUARDRULES_ACTIVE" if item["abstention_count"] > 0 else "CLEAN_SUCCESS"
        print(
            f"- [{item['case_id']}] Status={item['status']} ({flag}) | Abstentions={item['abstention_count']}"
        )

    return report


def main() -> None:
    report = asyncio.run(run_diagnostics())
    with open("diagnostic_pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nDiagnostic report saved to 'diagnostic_pipeline_report.json'.")


if __name__ == "__main__":
    main()
