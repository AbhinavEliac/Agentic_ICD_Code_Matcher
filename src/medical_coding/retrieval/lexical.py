"""Lexical BM25 retrieval for local ICD-10-CM code matching."""

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from medical_coding.retrieval.base import BaseICDRetriever
from medical_coding.retrieval.tokenizer import (
    normalize_clinical_query,
    tokenize_clinical_text,
)
from medical_coding.schemas.icd import ICDCandidate, ICDCodeRecord
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class BM25ICDRetriever(BaseICDRetriever):
    """Lexical retriever utilizing BM25 over local ICD-10-CM descriptions and synonyms."""

    def __init__(self, index_path: Path | str | None = None) -> None:
        self.index_path = Path(index_path) if index_path else None
        self._bm25_index: BM25Okapi | None = None
        self._records: list[ICDCodeRecord] = []
        self._tokenized_corpus: list[list[str]] = []

    @property
    def is_indexed(self) -> bool:
        return self._bm25_index is not None and len(self._records) > 0

    def build_index(self, records: list[ICDCodeRecord]) -> None:
        """Tokenize and compile BM25 index from authoritative local ICD records."""
        if not records:
            raise ValueError("Cannot build BM25 index with empty ICD record list.")

        self._records = records
        self._tokenized_corpus = []

        for record in records:
            # Index code, formatted and unformatted
            code_tokens = [record.code.lower(), record.unformatted_code.lower()]
            # Index searchable clinical description, synonyms, and inclusion terms
            searchable_text = record.get_searchable_text()
            text_tokens = tokenize_clinical_text(searchable_text, expand_abbreviations=True)
            combined_tokens = code_tokens + text_tokens
            self._tokenized_corpus.append(combined_tokens)

        self._bm25_index = BM25Okapi(self._tokenized_corpus)
        logger.info("Compiled BM25 index over %d authoritative ICD records.", len(self._records))

    def retrieve(self, query: str, top_k: int = 15) -> list[ICDCandidate]:
        """Perform lexical BM25 search against local ICD-10-CM catalog.

        Scoring Calculation:
            BM25 scores are normalized to [0.0, 1.0] using a composite formula:
            score = 0.5 * (raw_bm25 / max_raw_bm25) + 0.5 * (term_overlap / query_length)
            where:
            - raw_bm25 / max_raw_bm25 represents relative term-frequency saturation
            - term_overlap / query_length represents exact query coverage fraction
        """
        if not self.is_indexed:
            raise RuntimeError("BM25 index has not been built or loaded.")

        normalized_query = normalize_clinical_query(query)
        query_tokens = tokenize_clinical_text(normalized_query, expand_abbreviations=True)
        if not query_tokens:
            return []

        assert self._bm25_index is not None
        raw_scores = self._bm25_index.get_scores(query_tokens)

        max_raw_score = float(max(raw_scores)) if len(raw_scores) > 0 else 0.0
        if max_raw_score <= 0.0:
            return []

        # Find indices with non-zero scores sorted descending
        indexed_scores = [
            (idx, float(score)) for idx, score in enumerate(raw_scores) if score > 0.0
        ]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        candidates: list[ICDCandidate] = []
        query_token_set = set(query_tokens)

        for idx, raw_score in indexed_scores[:top_k]:
            record = self._records[idx]
            doc_tokens = set(self._tokenized_corpus[idx])
            overlap = len(query_token_set & doc_tokens)
            coverage = overlap / len(query_token_set) if query_token_set else 0.0

            relative_bm25 = raw_score / max_raw_score
            # Composite normalized score
            normalized_score = 0.5 * relative_bm25 + 0.5 * coverage
            clamped_score = max(0.0, min(1.0, round(normalized_score, 4)))

            candidates.append(
                ICDCandidate(
                    code=record.code,
                    description=record.description,
                    retrieval_score=clamped_score,
                    retrieval_method="bm25",
                    is_valid_billable=record.is_valid_billable,
                    lexical_score=clamped_score,
                    category=record.category,
                    coding_system=record.coding_system,
                )
            )

        return candidates

    def save(self, output_dir: Path | str) -> None:
        """Persist tokenized BM25 corpus and record metadata to disk."""
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        save_file = target_dir / "bm25_index.pkl"

        data: dict[str, Any] = {
            "tokenized_corpus": self._tokenized_corpus,
            "records": [r.model_dump() for r in self._records],
        }
        with open(save_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Saved BM25 index to %s", save_file)

    def load(self, index_dir: Path | str) -> None:
        """Load BM25 corpus from persisted disk storage."""
        source_file = Path(index_dir) / "bm25_index.pkl"
        if not source_file.exists():
            raise FileNotFoundError(f"BM25 index file not found at {source_file}")

        with open(source_file, "rb") as f:
            data = pickle.load(f)

        self._tokenized_corpus = data["tokenized_corpus"]
        self._records = [ICDCodeRecord.model_validate(r) for r in data["records"]]
        self._bm25_index = BM25Okapi(self._tokenized_corpus)
        logger.info("Loaded BM25 index from %s (%d records)", source_file, len(self._records))
