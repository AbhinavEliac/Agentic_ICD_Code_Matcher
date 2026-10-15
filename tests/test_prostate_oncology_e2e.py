"""End-to-End Test for Metastatic Prostate Oncology Case.

Validates:
1. Normalization and capture of primary prostate cancer (C61).
2. Normalization and capture of metastatic disease sites:
   - Bone metastases (lumbar spine, pelvis) -> C79.51
   - Retroperitoneal lymph node metastases -> C77.2
3. Complete preservation of structured oncology attributes:
   - Gleason score: 9 (or 4+5)
   - WHO Grade Group: 5
   - PSA level: 45 ng/mL
   - Castration status: Castration-resistant / mCRPC
   - Metastatic sites list: bone, lymph nodes
   - Systemic therapies: ADT, leuprolide, enzalutamide, denosumab, radiotherapy
4. Rejection of all five incorrect secondary mappings:
   - No C79.89 for "Metastatic sites"
   - No K68.2 for "retroperitoneal lymphadenopathy"
   - No CPT codes (e.g. 11057) for clinical diagnoses
   - No fragmented adjectival modifiers ("metastatic castration-resistant", "pelvis", "Gleason 9 (4")
5. Latency profiling:
   - Measures retrieval and ranking latency separately.
"""

import asyncio
import time
import pytest

from medical_coding.agents.clinical_extractor import EvidenceFirstFactExtractor
from medical_coding.graph.pipeline import process_clinical_document


PROSTATE_MCRPC_NOTE = """CLINICAL DOCUMENTATION / DISCHARGE SUMMARY:
Patient is a 72-year-old male with metastatic castration-resistant prostate cancer (mCRPC), adenocarcinoma of the prostate, Gleason score 9 (4+5), Grade Group 5, PSA 45 ng/mL.
Metastatic sites: bone (lumbar spine, pelvis), retroperitoneal lymphadenopathy.
Patient on androgen deprivation therapy (ADT) with leuprolide and enzalutamide.
Admitted for evaluation and management of secondary bone pain and osteoblastic spine lesions.
Past Medical History: Essential hypertension, Type 2 diabetes mellitus without complications.
Active Medications: Leuprolide, Enzalutamide, Denosumab, Amlodipine 10mg daily, Metformin 500mg BID.
Hospital Course: Evaluated by Medical Oncology. Palliative radiotherapy recommended for painful lumbar spine osseous metastases. Hypertension managed with amlodipine; blood pressure stable. Blood glucose monitored; metformin continued.
Discharge Diagnoses:
1. Metastatic castration-resistant adenocarcinoma of prostate
2. Metastatic sites: bone metastases (lumbar spine, pelvis)
3. Retroperitoneal lymph node metastases
4. Essential hypertension
5. Type 2 diabetes mellitus
"""


def test_oncology_context_and_normalization_extraction():
    """Verify evidence-to-concept normalization and oncology attribute extraction."""
    extractor = EvidenceFirstFactExtractor()
    state = extractor.extract_clinical_state(PROSTATE_MCRPC_NOTE, "doc-test-prostate")

    # 1. Primary cancer concept verification
    assert state.primary_diagnosis is not None
    assert "prostate" in state.primary_diagnosis.raw_term.lower()

    # 2. Oncology context attributes verification
    onc_ctx = state.oncology_context
    assert onc_ctx is not None
    assert onc_ctx.primary_site == "prostate"
    assert onc_ctx.gleason_score in ("9", "9 (4+5)", "4+5=9")
    assert "5" in str(onc_ctx.grade_group)
    assert "45" in str(onc_ctx.psa_level)
    assert "castrat" in str(onc_ctx.castration_status).lower()
    assert "bone" in onc_ctx.metastatic_sites
    assert any("lymph" in s.lower() for s in onc_ctx.metastatic_sites)

    # 3. Check extracted candidate terms: no junk fragments
    all_raw_terms = [c.raw_term.lower() for c in state.all_candidates]
    for junk in ("metastatic sites", "pelvis", "metastatic castration-resistant", "gleason score 9 (4"):
        assert junk not in all_raw_terms, f"Junk candidate '{junk}' should not be in extracted candidates!"

    # 4. Bone metastases and retroperitoneal lymph nodes must be present
    has_bone_met = any(
        "bone" in c.raw_term.lower() or "bone" in c.normalized_diagnosis.lower()
        for c in state.all_candidates
    )
    assert has_bone_met, "Bone metastases must be extracted!"

    has_lymph_met = any(
        "retroperitoneal" in c.raw_term.lower() or "intra-abdominal" in c.normalized_diagnosis.lower() or "lymph" in c.raw_term.lower()
        for c in state.all_candidates
    )
    assert has_lymph_met, "Retroperitoneal lymph node metastases must be extracted!"


@pytest.mark.asyncio
async def test_prostate_pipeline_end_to_end_and_latency():
    """Verify end-to-end coding, absence of incorrect mappings, and profiled latency."""
    node_timings = {}

    def progress_callback(step, total, node, msg, duration_ms):
        node_timings[node] = duration_ms

    from medical_coding.graph.nodes import get_or_initialize_retrieval_system
    # Pre-warm catalog/indices from disk so recorded latency measures steady-state inference
    get_or_initialize_retrieval_system()

    t0 = time.perf_counter()
    result = await process_clinical_document(
        source=PROSTATE_MCRPC_NOTE,
        document_id="doc-prostate-mcrpc-e2e",
        progress_callback=progress_callback,
        persist_thread=False,
    )
    total_time = time.perf_counter() - t0

    print(f"\n[E2E Latency Profile] Total: {total_time:.3f}s")
    for node, dur in node_timings.items():
        print(f"  - {node}: {dur:.1f}ms")

    # 1. Primary code must be C61 (Malignant neoplasm of prostate)
    assert result.primary_diagnosis is not None
    assert result.primary_diagnosis.database_code == "C61"

    # 2. Secondary codes must include C79.51 (bone metastasis) and C77.2 (intra-abdominal / retroperitoneal lymph nodes)
    # 2. Secondary codes must include C79.51 (bone metastasis) and C77.2 (intra-abdominal / retroperitoneal lymph nodes)
    sec_codes = {s.database_code for s in result.secondary_diagnoses}
    print(f"Assigned secondary codes: {sec_codes}")
    for sec in result.secondary_diagnoses:
        print(f"  {sec.database_code} - {sec.database_description} (term: '{sec.raw_term}')")

    assert "C79.51" in sec_codes, f"Expected C79.51 (Secondary malignant neoplasm of bone) in {sec_codes}"
    assert "C77.2" in sec_codes, f"Expected C77.2 (Secondary and unspecified malignant neoplasm of intra-abdominal lymph nodes) in {sec_codes}"

    # 3. Must include chronic co-morbidities
    assert "I10" in sec_codes, f"Expected I10 (Essential hypertension) in {sec_codes}"
    assert "E11.9" in sec_codes, f"Expected E11.9 (Type 2 diabetes mellitus) in {sec_codes}"

    # 4. Zero incorrect mappings:
    # No C79.89, no K68.2, no CPT codes (e.g. 11057)
    assert "C79.89" not in sec_codes, "C79.89 must NOT be assigned!"
    assert "K68.2" not in sec_codes, "K68.2 (Retroperitoneal fibrosis) must NOT be assigned!"
    for c in sec_codes:
        assert not (c.isdigit() and len(c) in (4, 5)), f"CPT code '{c}' must not be assigned to a diagnosis!"

    # 5. Check retrieval and ranking timings
    retrieval_ms = node_timings.get("retrieve_candidates", 0.0)
    ranking_ms = node_timings.get("rank_candidates", 0.0)
    print(f"Candidate Retrieval: {retrieval_ms:.1f}ms | Candidate Ranking: {ranking_ms:.1f}ms")
    # Candidate ranking should be under 500ms (achieving the ~1.0s to ~0.15s reduction)
    assert ranking_ms < 500.0, f"Ranking latency {ranking_ms:.1f}ms should be under 500ms"
