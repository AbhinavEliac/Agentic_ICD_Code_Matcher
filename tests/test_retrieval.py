"""Comprehensive unit and integration tests for the Local ICD-10-CM Retrieval and Knowledge Layer."""

import json
from pathlib import Path

import pytest

from medical_coding.dataset.loader import (
    CMSOrderFileLoader,
    TabularICDLoader,
)
from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.models.factory import FastLocalEmbeddings
from medical_coding.retrieval.hybrid import HybridICDRetriever
from medical_coding.retrieval.lexical import BM25ICDRetriever
from medical_coding.retrieval.vector import FAISSICDRetriever
from medical_coding.schemas.icd import ICDCodeRecord


@pytest.fixture
def sample_records() -> list[ICDCodeRecord]:
    """Curated authoritative ICD records for testing retrieval behaviors."""
    return [
        ICDCodeRecord(
            code="I50.21",
            unformatted_code="I5021",
            description="Acute systolic heart failure",
            short_description="Acute systolic heart failure",
            long_description="Acute systolic (congestive) heart failure",
            is_valid_billable=True,
            category="I50",
            inclusion_terms=["Acute congestive heart failure with systolic dysfunction"],
            exclusion_terms=["Combined systolic and diastolic heart failure (I50.4-)"],
            synonyms=[
                "acute systolic HF",
                "acute decompensated systolic heart failure",
                "HFrEF acute",
            ],
        ),
        ICDCodeRecord(
            code="I50.22",
            unformatted_code="I5022",
            description="Chronic systolic heart failure",
            short_description="Chronic systolic heart failure",
            long_description="Chronic systolic (congestive) heart failure",
            is_valid_billable=True,
            category="I50",
            inclusion_terms=["Chronic congestive heart failure with systolic dysfunction"],
            exclusion_terms=["Combined systolic and diastolic heart failure (I50.4-)"],
            synonyms=["chronic systolic HF", "compensated systolic heart failure", "chronic HFrEF"],
        ),
        ICDCodeRecord(
            code="I50.9",
            unformatted_code="I509",
            description="Heart failure, unspecified",
            short_description="Heart failure, unspecified",
            long_description="Congestive heart failure NOS; Cardiac failure NOS",
            is_valid_billable=True,
            category="I50",
            inclusion_terms=["Congestive heart failure NOS"],
            synonyms=["HF", "CHF", "heart failure"],
        ),
        ICDCodeRecord(
            code="I10",
            unformatted_code="I10",
            description="Essential primary hypertension",
            short_description="Essential primary hypertension",
            long_description="Essential primary hypertension; High blood pressure",
            is_valid_billable=True,
            category="I10",
            inclusion_terms=["High blood pressure", "Arterial hypertension"],
            synonyms=["HTN", "primary hypertension", "high blood pressure"],
        ),
        ICDCodeRecord(
            code="E11.9",
            unformatted_code="E119",
            description="Type 2 diabetes mellitus without complications",
            short_description="Type 2 diabetes mellitus without complications",
            long_description="Type 2 diabetes mellitus without complications",
            is_valid_billable=True,
            category="E11",
            inclusion_terms=["Diabetes mellitus type 2 NOS"],
            synonyms=["T2DM", "type 2 diabetes", "diabetes mellitus type 2"],
        ),
    ]


@pytest.fixture
def indexed_hybrid_system(
    sample_records: list[ICDCodeRecord],
) -> tuple[LocalICDCatalog, HybridICDRetriever]:
    """Provide a fully configured local catalog and hybrid retriever."""
    catalog = LocalICDCatalog()
    for rec in sample_records:
        catalog.add_record(rec)
    catalog._initialized = True

    bm25 = BM25ICDRetriever()
    bm25.build_index(sample_records)

    embeddings = FastLocalEmbeddings(dim=384)
    faiss_ret = FAISSICDRetriever(embeddings=embeddings)
    faiss_ret.build_index(sample_records)

    hybrid = HybridICDRetriever(
        lexical_retriever=bm25,
        vector_retriever=faiss_ret,
        catalog=catalog,
        weight_semantic=0.6,
        weight_lexical=0.4,
        min_score_threshold=0.35,
        default_top_k=5,
    )
    return catalog, hybrid


# ==============================================================================
# 1. RETRIEVAL BEHAVIOR TESTS (Required Test Cases)
# ==============================================================================


def test_exact_diagnosis_retrieval(
    indexed_hybrid_system: tuple[LocalICDCatalog, HybridICDRetriever],
) -> None:
    """Exact clinical diagnosis must return the correct ICD code with high confidence."""
    _, hybrid = indexed_hybrid_system
    results = hybrid.retrieve("Acute systolic heart failure", top_k=3)

    assert len(results) > 0
    top = results[0]
    assert top.code == "I50.21"
    assert "Acute systolic heart failure" in top.description
    assert top.retrieval_score >= 0.85
    assert top.is_valid_billable is True


def test_synonym_retrieval(
    indexed_hybrid_system: tuple[LocalICDCatalog, HybridICDRetriever],
) -> None:
    """Clinical synonyms like 'HFrEF acute' must retrieve the correct underlying code."""
    _, hybrid = indexed_hybrid_system
    results = hybrid.retrieve("HFrEF acute", top_k=3)

    assert len(results) > 0
    top = results[0]
    assert top.code == "I50.21"
    assert top.retrieval_score > 0.50


def test_abbreviation_retrieval(
    indexed_hybrid_system: tuple[LocalICDCatalog, HybridICDRetriever],
) -> None:
    """Standard clinical abbreviations like 'HTN' or 'T2DM' must retrieve correct codes."""
    _, hybrid = indexed_hybrid_system

    # Test HTN -> I10
    htn_results = hybrid.retrieve("HTN", top_k=3)
    assert len(htn_results) > 0
    assert htn_results[0].code == "I10"

    # Test T2DM -> E11.9
    dm_results = hybrid.retrieve("T2DM", top_k=3)
    assert len(dm_results) > 0
    assert dm_results[0].code == "E11.9"


def test_partial_terminology_retrieval(
    indexed_hybrid_system: tuple[LocalICDCatalog, HybridICDRetriever],
) -> None:
    """Incomplete/partial clinical terms like 'systolic failure' must retrieve relevant candidate codes."""
    _, hybrid = indexed_hybrid_system
    results = hybrid.retrieve("systolic failure", top_k=5)

    assert len(results) >= 2
    codes = [c.code for c in results]
    assert "I50.21" in codes or "I50.22" in codes


def test_ambiguous_diagnosis_retrieval(
    indexed_hybrid_system: tuple[LocalICDCatalog, HybridICDRetriever],
) -> None:
    """Ambiguous diagnosis like 'heart failure' must retrieve multiple candidates with graded scores."""
    _, hybrid = indexed_hybrid_system
    results = hybrid.retrieve("heart failure", top_k=5)

    assert len(results) >= 2
    codes = [c.code for c in results]
    assert "I50.9" in codes
    assert "I50.21" in codes or "I50.22" in codes
    # Ensure scores are strictly monotonically descending
    scores = [c.retrieval_score for c in results]
    assert scores == sorted(scores, reverse=True)


def test_no_match_abstention(
    indexed_hybrid_system: tuple[LocalICDCatalog, HybridICDRetriever],
) -> None:
    """Uncovered, nonsensical, or out-of-catalog conditions must return an empty list (abstention)."""
    _, hybrid = indexed_hybrid_system
    results = hybrid.retrieve("completely unknown martian syndrome x789", top_k=5)

    assert results == []  # Explicit abstention state


# ==============================================================================
# 2. DATASET INGESTION: DUPLICATES & MALFORMED RECORDS
# ==============================================================================


def test_duplicate_candidate_handling(tmp_path: Path) -> None:
    """Dataset loader must detect duplicate codes, merge synonyms/terms, and record metrics."""
    csv_file = tmp_path / "duplicates.csv"
    csv_content = (
        "code,description,is_valid_billable,synonyms\n"
        "I50.21,Acute systolic heart failure,1,acute systolic HF\n"
        "I50.21,Acute systolic heart failure duplicate,1,HFrEF acute\n"
        "E11.9,Type 2 diabetes,1,T2DM\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    loader = TabularICDLoader(csv_file)
    records, stats = loader.load()

    assert stats.total_records == 2
    assert stats.duplicates_dropped == 1
    assert "I50.21" in records
    # Merged synonyms check
    assert "acute systolic HF" in records["I50.21"].synonyms
    assert "HFrEF acute" in records["I50.21"].synonyms


def test_malformed_icd_dataset_handling(tmp_path: Path) -> None:
    """Dataset loader must filter out malformed codes, empty descriptions, and invalid rows."""
    csv_file = tmp_path / "malformed.csv"
    csv_content = (
        "code,description,is_valid_billable\n"
        "I50.21,Acute systolic heart failure,1\n"
        ",Missing code diagnosis,1\n"  # Malformed: no code
        "123.45,Code starts with digit,1\n"  # Malformed: invalid ICD pattern
        "I50.22,,1\n"  # Malformed: empty description
        "I50.23,Acute on chronic systolic heart failure,1\n"  # Valid
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    loader = TabularICDLoader(csv_file)
    records, stats = loader.load()

    assert stats.total_records == 2
    assert stats.malformed_dropped == 3
    assert "I50.21" in records
    assert "I50.23" in records


def test_missing_required_columns_error(tmp_path: Path) -> None:
    """Tabular loader must reject dataset missing required code/description columns."""
    bad_csv = tmp_path / "bad_columns.csv"
    bad_csv.write_text("random_col1,random_col2\nval1,val2\n", encoding="utf-8")

    loader = TabularICDLoader(bad_csv)
    with pytest.raises(ValueError, match="Missing required columns"):
        loader.load()


# ==============================================================================
# 3. FORMATS & CMS ORDER FILE INGESTION
# ==============================================================================


def test_json_dataset_ingestion(tmp_path: Path) -> None:
    """Loader must support JSON array format."""
    json_file = tmp_path / "dataset.json"
    data = [
        {
            "code": "I50.21",
            "description": "Acute systolic heart failure",
            "is_valid_billable": True,
        },
        {"code": "I10", "description": "Essential primary hypertension", "is_valid_billable": True},
    ]
    json_file.write_text(json.dumps(data), encoding="utf-8")

    loader = TabularICDLoader(json_file)
    records, stats = loader.load()

    assert stats.total_records == 2
    assert "I50.21" in records
    assert "I10" in records


def test_cms_order_file_ingestion(tmp_path: Path) -> None:
    """CMSOrderFileLoader must parse fixed-width CMS order lines correctly."""
    cms_file = tmp_path / "cms_order.txt"
    # Format: order(5) space code(7) space valid(1) space short(60) space long
    line1 = f"{1:05d} I5021   1 {'Acute systolic heart failure':<60} Acute systolic (congestive) heart failure"
    line2 = f"{2:05d} I50     0 {'Heart failure':<60} Heart failure category header"
    cms_file.write_text(f"{line1}\n{line2}\n", encoding="utf-8")

    loader = CMSOrderFileLoader(cms_file)
    records, stats = loader.load()

    assert stats.total_records == 2
    assert stats.valid_billable_count == 1
    assert stats.non_billable_count == 1
    assert "I50.21" in records
    assert records["I50.21"].is_valid_billable is True
    assert "I50" in records
    assert records["I50"].is_valid_billable is False


# ==============================================================================
# 4. CONFIGURABLE RETRIEVAL WEIGHTS & CATALOG SOURCE OF TRUTH
# ==============================================================================


def test_configurable_retrieval_weights(sample_records: list[ICDCodeRecord]) -> None:
    """Configuring pure semantic or pure lexical weights alters candidate ranking accordingly."""
    bm25 = BM25ICDRetriever()
    bm25.build_index(sample_records)

    faiss_ret = FAISSICDRetriever(embeddings=FastLocalEmbeddings(dim=384))
    faiss_ret.build_index(sample_records)

    # 100% lexical weight
    lexical_only = HybridICDRetriever(
        lexical_retriever=bm25,
        vector_retriever=faiss_ret,
        weight_semantic=0.0,
        weight_lexical=1.0,
        min_score_threshold=0.10,
    )
    lex_res = lexical_only.retrieve("Acute systolic heart failure", top_k=1)
    assert len(lex_res) > 0
    assert lex_res[0].code == "I50.21"
    assert lex_res[0].retrieval_score == lex_res[0].lexical_score

    # 100% semantic weight
    semantic_only = HybridICDRetriever(
        lexical_retriever=bm25,
        vector_retriever=faiss_ret,
        weight_semantic=1.0,
        weight_lexical=0.0,
        min_score_threshold=0.10,
    )
    sem_res = semantic_only.retrieve("Acute systolic heart failure", top_k=1)
    assert len(sem_res) > 0
    assert sem_res[0].code == "I50.21"
    assert sem_res[0].retrieval_score == sem_res[0].semantic_score


def test_catalog_enforcement_strictly_prevents_unindexed_codes(
    sample_records: list[ICDCodeRecord],
) -> None:
    """Catalog filtering must discard any candidate not present in authoritative dataset."""
    catalog = LocalICDCatalog()
    # Catalog contains ONLY I10 and E11.9
    catalog.add_record(sample_records[3])  # I10
    catalog.add_record(sample_records[4])  # E11.9
    catalog._initialized = True

    bm25 = BM25ICDRetriever()
    bm25.build_index(sample_records)  # Built with all 5 records

    faiss_ret = FAISSICDRetriever(embeddings=FastLocalEmbeddings(dim=384))
    faiss_ret.build_index(sample_records)

    hybrid = HybridICDRetriever(
        lexical_retriever=bm25,
        vector_retriever=faiss_ret,
        catalog=catalog,  # Constrained catalog
        min_score_threshold=0.10,
    )

    # Query for heart failure (I50.21 is in retriever index but NOT in catalog)
    results = hybrid.retrieve("Acute systolic heart failure")
    # Must NOT return I50.21 because catalog is the sole source of truth
    for c in results:
        assert catalog.is_valid_code(c.code)
    assert not any(c.code == "I50.21" for c in results)


# ==============================================================================
# 5. PERSISTENCE & DISK RELOADING
# ==============================================================================


def test_retrieval_index_persistence(tmp_path: Path, sample_records: list[ICDCodeRecord]) -> None:
    """Indices and catalogs saved to disk must reload and produce identical retrieval results."""
    catalog = LocalICDCatalog()
    for rec in sample_records:
        catalog.add_record(rec)
    catalog._initialized = True
    catalog.save_to_json(tmp_path / "catalog.json")

    bm25 = BM25ICDRetriever()
    bm25.build_index(sample_records)
    bm25.save(tmp_path)

    embeddings = FastLocalEmbeddings(dim=384)
    faiss_ret = FAISSICDRetriever(embeddings=embeddings)
    faiss_ret.build_index(sample_records)
    faiss_ret.save(tmp_path)

    # Reload from disk
    reloaded_catalog = LocalICDCatalog.load_from_json(tmp_path / "catalog.json")
    reloaded_bm25 = BM25ICDRetriever()
    reloaded_bm25.load(tmp_path)
    reloaded_faiss = FAISSICDRetriever(embeddings=embeddings)
    reloaded_faiss.load(tmp_path)

    assert len(reloaded_catalog) == len(sample_records)
    assert reloaded_bm25.is_indexed is True
    assert reloaded_faiss.is_indexed is True

    reloaded_hybrid = HybridICDRetriever(
        lexical_retriever=reloaded_bm25,
        vector_retriever=reloaded_faiss,
        catalog=reloaded_catalog,
    )

    results = reloaded_hybrid.retrieve("Acute systolic heart failure", top_k=1)
    assert len(results) == 1
    assert results[0].code == "I50.21"
