"""Tests verifying multi-system database ingestion, vector & semantic retrieval, and JSON output formatting.

Validates that:
1. ICD-10-CM, ICD-O, and CPT codes are indexed and retrieved with zero hallucination.
2. The final JSON payload adheres to:
   {
       "code": ...,
       "description": ...,
       "role": ...,
       "evidence_quote": ...,
       "confidence_score": ...,
       "is_terminal_billable": ...,
       "icdo": <code if matched, None if not>,
       "cpt": <code if matched, None if not>,
       "icd10cm": <code if matched, None if not>
   }
"""


import pytest

from medical_coding.config.settings import get_settings
from medical_coding.dataset.loader import load_icd_dataset
from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.models.factory import FastLocalEmbeddings
from medical_coding.retrieval.hybrid import HybridICDRetriever
from medical_coding.retrieval.lexical import BM25ICDRetriever
from medical_coding.retrieval.vector import FAISSICDRetriever
from medical_coding.schemas.enums import Acuity, Certainty, DiagnosisRole
from medical_coding.schemas.response import CodedDiagnosisResponse


def test_multisystem_catalog_and_retrieval() -> None:
    """Test that catalog correctly partitions and retrieves across ICD-10-CM, ICD-O, and CPT."""
    settings = get_settings()
    db_wb = settings.get_database_workbook_path()

    if not db_wb or not db_wb.exists():
        pytest.skip("Local Database folder not present in test environment.")

    records_dict, stats = load_icd_dataset(db_wb)
    assert len(records_dict) > 1000
    assert stats.total_records > 1000

    catalog = LocalICDCatalog()
    for rec in records_dict.values():
        catalog.add_record(rec)
    catalog._stats = stats
    catalog._initialized = True

    # Check system partitioning
    icd10_codes = catalog.get_codes_by_system("ICD-10-CM")
    icdo_codes = catalog.get_codes_by_system("ICD-O")
    cpt_codes = catalog.get_codes_by_system("CPT")

    assert len(icd10_codes) > 10000
    assert len(icdo_codes) > 100
    assert len(cpt_codes) > 1000

    # Build hybrid retriever
    rec_list = list(records_dict.values())
    bm25 = BM25ICDRetriever()
    bm25.build_index(rec_list)

    faiss_ret = FAISSICDRetriever(FastLocalEmbeddings(dim=384))
    faiss_ret.build_index(rec_list)

    hybrid = HybridICDRetriever(
        lexical_retriever=bm25,
        vector_retriever=faiss_ret,
        catalog=catalog,
        min_score_threshold=0.20,
    )

    # 1. Retrieve ICD-10-CM
    results_icd10 = hybrid.retrieve("heart failure", top_k=5, system="ICD-10-CM")
    assert len(results_icd10) > 0
    assert all(catalog.is_valid_code(c.code) for c in results_icd10)
    assert any("I50" in c.code for c in results_icd10)

    # 2. Retrieve ICD-O
    results_icdo = hybrid.retrieve("carcinoma", top_k=5, system="ICD-O")
    assert len(results_icdo) > 0
    assert all(catalog.is_valid_code(c.code) for c in results_icdo)
    assert all(c.code.startswith("M") for c in results_icdo)

    # 3. Retrieve CPT
    results_cpt = hybrid.retrieve("excision", top_k=5, system="CPT")
    assert len(results_cpt) > 0
    assert all(catalog.is_valid_code(c.code) for c in results_cpt)


def test_json_output_format_with_multisystem_fields() -> None:
    """Validate JSON serializability and schema structure conforming to exact user format."""
    diag = CodedDiagnosisResponse(
        code="I50.21",
        description="Acute systolic (congestive) heart failure",
        role=DiagnosisRole.PRIMARY,
        acuity=Acuity.ACUTE,
        certainty=Certainty.CONFIRMED,
        evidence_quote="Patient presented with acute decompensated systolic heart failure.",
        confidence_score=0.96,
        is_terminal_billable=True,
        icd10cm="I50.21",
        icdo=None,
        cpt=None,
    )

    dumped = diag.model_dump()
    assert dumped["code"] == "I50.21"
    assert dumped["icd10cm"] == "I50.21"
    assert dumped["icdo"] is None
    assert dumped["cpt"] is None
    assert dumped["is_terminal_billable"] is True
    assert dumped["confidence_score"] == 0.96

    # Test oncology scenario with morphology code
    diag_onc = CodedDiagnosisResponse(
        code="C50.911",
        description="Malignant neoplasm of unspecified site of right female breast",
        role=DiagnosisRole.PRIMARY,
        acuity=Acuity.UNSPECIFIED,
        certainty=Certainty.CONFIRMED,
        evidence_quote="Biopsy confirmed right breast infiltrating duct carcinoma.",
        confidence_score=0.94,
        is_terminal_billable=True,
        icd10cm="C50.911",
        icdo="M8500.3",
        cpt="19120",
    )

    dumped_onc = diag_onc.model_dump()
    assert dumped_onc["code"] == "C50.911"
    assert dumped_onc["icd10cm"] == "C50.911"
    assert dumped_onc["icdo"] == "M8500.3"
    assert dumped_onc["cpt"] == "19120"
