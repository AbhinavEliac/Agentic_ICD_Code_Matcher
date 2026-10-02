"""Dense vector retrieval using local embeddings and FAISS index."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from medical_coding.models.base import BaseLocalEmbeddings
from medical_coding.retrieval.base import BaseICDRetriever
from medical_coding.retrieval.tokenizer import normalize_clinical_query
from medical_coding.schemas.icd import ICDCandidate, ICDCodeRecord
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class FAISSICDRetriever(BaseICDRetriever):
    """Semantic vector retriever using local offline embeddings and FAISS IndexFlatIP."""

    def __init__(
        self,
        embeddings: BaseLocalEmbeddings,
        index_path: Path | str | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.index_path = Path(index_path) if index_path else None
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._records: list[ICDCodeRecord] = []
        self._dimension: int = 0

    @property
    def is_indexed(self) -> bool:
        return self._faiss_index is not None and len(self._records) > 0

    def build_index(self, records: list[ICDCodeRecord], batch_size: int = 64) -> None:
        """Embed descriptions and build FAISS IndexFlatIP over authoritative records."""
        if not records:
            raise ValueError("Cannot build FAISS index with empty ICD record list.")

        self._records = records
        searchable_texts = [r.get_searchable_text() for r in records]

        logger.info("Generating embeddings for %d ICD records...", len(records))
        all_embeddings: list[list[float]] = []
        for i in range(0, len(searchable_texts), batch_size):
            batch = searchable_texts[i : i + batch_size]
            batch_vecs = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_vecs)

        matrix = np.array(all_embeddings, dtype=np.float32)
        # Ensure L2-normalized so inner product equals cosine similarity
        faiss.normalize_L2(matrix)

        self._dimension = matrix.shape[1]
        self._faiss_index = faiss.IndexFlatIP(self._dimension)
        self._faiss_index.add(matrix)

        logger.info(
            "Compiled FAISS IndexFlatIP with %d vectors (dimension %d).",
            self._faiss_index.ntotal,
            self._dimension,
        )

    def retrieve(self, query: str, top_k: int = 15) -> list[ICDCandidate]:
        """Embed query and search nearest semantic neighbors in local FAISS index.

        Scoring Calculation:
            Cosine similarity via L2-normalized Inner Product, clamped to [0.0, 1.0].
        """
        if not self.is_indexed:
            raise RuntimeError("FAISS vector index has not been built or loaded.")

        normalized_query = normalize_clinical_query(query)
        if not normalized_query.strip():
            return []

        raw_vec = self.embeddings.embed_query(normalized_query)
        query_matrix = np.array([raw_vec], dtype=np.float32)
        faiss.normalize_L2(query_matrix)

        assert self._faiss_index is not None
        k_search = min(top_k, len(self._records))
        distances, indices = self._faiss_index.search(query_matrix, k_search)

        candidates: list[ICDCandidate] = []
        for sim, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._records):
                continue

            record = self._records[idx]
            # Clamp cosine similarity to [0.0, 1.0]
            norm_sim = max(0.0, min(1.0, round(float(sim), 4)))

            candidates.append(
                ICDCandidate(
                    code=record.code,
                    description=record.description,
                    retrieval_score=norm_sim,
                    retrieval_method="faiss",
                    is_valid_billable=record.is_valid_billable,
                    semantic_score=norm_sim,
                    category=record.category,
                )
            )

        return candidates

    def save(self, output_dir: Path | str) -> None:
        """Persist FAISS index and record metadata to disk."""
        if not self.is_indexed:
            raise RuntimeError("Cannot save uninitialized FAISS index.")

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "faiss.index"
        records_file = target_dir / "faiss_records.json"

        assert self._faiss_index is not None
        faiss.write_index(self._faiss_index, str(index_file))

        metadata: dict[str, Any] = {
            "dimension": self._dimension,
            "total_records": len(self._records),
            "records": [r.model_dump() for r in self._records],
        }
        with open(records_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved FAISS index to %s and metadata to %s", index_file, records_file)

    def load(self, index_dir: Path | str) -> None:
        """Load FAISS index and metadata from disk."""
        target_dir = Path(index_dir)
        index_file = target_dir / "faiss.index"
        records_file = target_dir / "faiss_records.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found at {index_file}")
        if not records_file.exists():
            raise FileNotFoundError(f"FAISS records metadata file not found at {records_file}")

        self._faiss_index = faiss.read_index(str(index_file))
        self._dimension = self._faiss_index.d

        with open(records_file, encoding="utf-8") as f:
            metadata = json.load(f)

        self._records = [ICDCodeRecord.model_validate(r) for r in metadata["records"]]
        logger.info(
            "Loaded FAISS index from %s (%d vectors, dim %d)",
            index_file,
            self._faiss_index.ntotal,
            self._dimension,
        )
