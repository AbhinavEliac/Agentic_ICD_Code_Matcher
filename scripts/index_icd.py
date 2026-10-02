"""CLI script for ingesting, validating, indexing, and persisting local ICD-10-CM datasets.

Builds both BM25 lexical and FAISS dense vector indices offline from hospital-supplied data.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on sys.path when running as a standalone script
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from medical_coding.config.settings import get_settings
from medical_coding.dataset.loader import load_icd_dataset
from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.models.base import BaseLocalEmbeddings
from medical_coding.models.factory import (
    FastLocalEmbeddings,
    SentenceTransformerLocalEmbeddings,
)
from medical_coding.retrieval.hybrid import HybridICDRetriever
from medical_coding.retrieval.lexical import BM25ICDRetriever
from medical_coding.retrieval.vector import FAISSICDRetriever
from medical_coding.schemas.icd import ICDDatasetStats
from medical_coding.utils.logging import get_logger

logger = get_logger("index_icd")


def build_and_persist_indexes(
    data_path: Path | str,
    output_dir: Path | str,
    embeddings: BaseLocalEmbeddings | None = None,
    force_fast_embeddings: bool = False,
) -> tuple[
    LocalICDCatalog, BM25ICDRetriever, FAISSICDRetriever, HybridICDRetriever, ICDDatasetStats
]:
    """Execute complete ingestion, validation, and indexing pipeline.

    Args:
        data_path: Path to user/hospital ICD source file (CSV, TSV, JSON, TXT).
        output_dir: Target directory to persist index artifacts.
        embeddings: Optional custom embedding model instance.
        force_fast_embeddings: If True, use deterministic FastLocalEmbeddings.

    Returns:
        Tuple of (catalog, bm25_retriever, faiss_retriever, hybrid_retriever, stats).
    """
    data_file = Path(data_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("  LOCAL ICD-10-CM INGESTION & INDEXING PIPELINE")
    print("=======================================================")
    print(f"Source Dataset: {data_file}")
    print(f"Output Directory: {out_dir}")

    # 1. Ingest and Validate Records
    print("\n[1/4] Ingesting and validating ICD dataset...")
    records_dict, stats = load_icd_dataset(data_file)
    record_list = list(records_dict.values())

    if not record_list:
        raise ValueError(f"No valid ICD records found in {data_file}. Indexing halted.")

    # 2. Build In-Memory Catalog and Persist
    catalog = LocalICDCatalog()
    for rec in record_list:
        catalog.add_record(rec)
    catalog._stats = stats
    catalog._initialized = True
    catalog.save_to_json(out_dir / "icd_catalog.json")

    # 3. Build Lexical BM25 Index
    print(f"[2/4] Building BM25 lexical index over {len(record_list)} records...")
    bm25_retriever = BM25ICDRetriever()
    bm25_retriever.build_index(record_list)
    bm25_retriever.save(out_dir)

    # 4. Resolve Local Embeddings and Build FAISS Index
    print("[3/4] Initializing local embeddings and building FAISS vector index...")
    if embeddings is None:
        cfg = get_settings()
        if force_fast_embeddings or not cfg.embedding_model_path.exists():
            print("  -> Using FastLocalEmbeddings (dim=384, deterministic hashing)")
            embeddings = FastLocalEmbeddings(dim=384)
        else:
            print(f"  -> Loading SentenceTransformer from {cfg.embedding_model_path}")
            embeddings = SentenceTransformerLocalEmbeddings(
                model_path=cfg.embedding_model_path,
                device=cfg.embedding_device,
            )

    faiss_retriever = FAISSICDRetriever(embeddings=embeddings)
    faiss_retriever.build_index(record_list)
    faiss_retriever.save(out_dir)

    # 5. Persist Index Manifest & Ingestion Stats
    print("[4/4] Persisting catalog metadata and index statistics...")
    stats_file = out_dir / "index_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats.model_dump(), f, indent=2)

    # 6. Instantiate Hybrid Retriever
    hybrid_retriever = HybridICDRetriever(
        lexical_retriever=bm25_retriever,
        vector_retriever=faiss_retriever,
        catalog=catalog,
        weight_semantic=0.6,
        weight_lexical=0.4,
        min_score_threshold=0.35,
        default_top_k=10,
    )

    print("\n=======================================================")
    print("  INDEXING COMPLETED SUCCESSFULLY")
    print("=======================================================")
    print(f"Total Authoritative Codes:  {stats.total_records}")
    print(f"Billable Specific Codes:    {stats.valid_billable_count}")
    print(f"Non-Billable Header Codes:  {stats.non_billable_count}")
    print(f"Unique 3-Char Categories:   {stats.unique_categories_count}")
    print(f"Duplicate Codes Filtered:   {stats.duplicates_dropped}")
    print(f"Malformed Rows Filtered:    {stats.malformed_dropped}")
    print(f"Total Synonyms Indexed:     {stats.total_synonyms}")
    print(f"Total Inclusion Terms:      {stats.total_inclusion_terms}")
    print(f"Index Artifacts Saved To:   {out_dir}")
    print("=======================================================\n")

    return catalog, bm25_retriever, faiss_retriever, hybrid_retriever, stats


def demonstrate_sample_retrieval(hybrid_retriever: HybridICDRetriever) -> None:
    """Run demonstration across clinical test query categories and explain score calculation."""
    sample_queries = [
        ("Exact Diagnosis", "Acute systolic heart failure"),
        ("Synonym", "HFrEF acute decompensated"),
        ("Medical Abbreviation", "HTN"),
        ("Partial Terminology", "systolic failure"),
        ("Ambiguous Diagnosis", "heart failure"),
        ("No Match / Abstention", "completely unknown extraterrestrial illness"),
    ]

    print("\n" + "=" * 70)
    print("  DEMONSTRATION: CLINICAL HYBRID RETRIEVAL & SCORING")
    print("=" * 70)
    print("Scoring Formula:")
    print("  Score_hybrid = (0.60 * Score_semantic) + (0.40 * Score_lexical)")
    print("  - Score_semantic in [0, 1]: Cosine similarity via FAISS IndexFlatIP")
    print("  - Score_lexical  in [0, 1]: Normalized BM25 with query coverage")
    print("  - Threshold: 0.35 (scores below 0.35 abstain to prevent false matches)")
    print("=" * 70 + "\n")

    for category, query in sample_queries:
        print("----------------------------------------------------------------------")
        print(f'Test Query [{category}]: "{query}"')
        candidates = hybrid_retriever.retrieve(query, top_k=3)

        if not candidates:
            print("  >> RESULT: ABSTENTION (No candidate met minimum relevance threshold)")
            print("  >> Clinical Safety: Agent will NOT guess or invent codes.")
        else:
            for rank, c in enumerate(candidates, start=1):
                sem = f"{c.semantic_score:.4f}" if c.semantic_score is not None else "0.0000"
                lex = f"{c.lexical_score:.4f}" if c.lexical_score is not None else "0.0000"
                print(
                    f"  Rank {rank}: [{c.code}] {c.description}\n"
                    f"          Score: {c.retrieval_score:.4f} | Method: {c.retrieval_method} "
                    f"(Semantic: {sem}, Lexical: {lex}, Billable: {c.is_valid_billable})"
                )
    print("-" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index hospital-supplied local ICD-10-CM dataset into BM25 and FAISS."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/icd10/sample_hospital_icd.csv",
        help="Path to hospital ICD-10 source file (CSV, TSV, JSON, or CMS TXT).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/indexes",
        help="Directory to save FAISS, BM25, and metadata indices.",
    )
    parser.add_argument(
        "--force-fast-embeddings",
        action="store_true",
        default=True,
        help="Use deterministic offline FastLocalEmbeddings instead of heavy model weights.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Demonstrate retrieval with sample clinical diagnoses after indexing.",
    )

    args = parser.parse_args()

    try:
        catalog, bm25, faiss_ret, hybrid, stats = build_and_persist_indexes(
            data_path=args.data_path,
            output_dir=args.output_dir,
            force_fast_embeddings=args.force_fast_embeddings,
        )
        if args.demo:
            demonstrate_sample_retrieval(hybrid)
    except Exception as e:
        logger.exception("Indexing failed: %s", e)
        print(f"\n[ERROR] Indexing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
