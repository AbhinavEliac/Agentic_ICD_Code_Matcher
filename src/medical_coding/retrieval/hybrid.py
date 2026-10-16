"""Hybrid retrieval combining lexical (BM25) and dense semantic (FAISS) local search."""

from typing import Any

from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.retrieval.base import BaseICDRetriever
from medical_coding.schemas.icd import ICDCandidate
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class HybridICDRetriever(BaseICDRetriever):
    """Fuses lexical BM25 and dense semantic FAISS retrieval over the local ICD dataset.

    Score Calculation Formula:
        For each retrieved code candidate 'c':
            Score_hybrid(c) = (weight_semantic * S_semantic(c)) + (weight_lexical * S_lexical(c))

        Where:
            - S_semantic(c) in [0.0, 1.0]: Cosine similarity from FAISS dense vector search
            - S_lexical(c) in [0.0, 1.0]: Normalized BM25 score with query coverage weighting
            - weight_semantic + weight_lexical are normalized to sum to 1.0.

        Candidates appearing in both lexical and semantic pools receive reinforcement.
        Candidates scoring below min_score_threshold are excluded. If no candidates exceed
        the threshold, the retriever returns an empty list (explicit abstention state).
    """

    def __init__(
        self,
        lexical_retriever: BaseICDRetriever,
        vector_retriever: BaseICDRetriever,
        catalog: LocalICDCatalog | None = None,
        weight_semantic: float = 0.6,
        weight_lexical: float = 0.4,
        min_score_threshold: float = 0.35,
        default_top_k: int = 10,
    ) -> None:
        self.lexical_retriever = lexical_retriever
        self.vector_retriever = vector_retriever
        self.catalog = catalog

        # Normalize weights so they strictly sum to 1.0
        total_weight = weight_semantic + weight_lexical
        if total_weight <= 0.0:
            raise ValueError("Total weight (semantic + lexical) must be positive.")
        self.weight_semantic = weight_semantic / total_weight
        self.weight_lexical = weight_lexical / total_weight

        self.min_score_threshold = min_score_threshold
        self.default_top_k = default_top_k
        self._query_cache: dict[tuple[str, int, float, str | None], list[ICDCandidate]] = {}

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        system: str | None = None,
    ) -> list[ICDCandidate]:
        """Execute hybrid search, merge candidates, score, deduplicate, and rank.

        Args:
            query: Normalized clinical diagnosis or condition description.
            top_k: Maximum candidate pool size to return (defaults to configured top_k).
            min_score: Minimum threshold override; candidates below this are pruned.
            system: Optional filter to restrict candidates to a specific coding system ('ICD-10-CM', 'ICD-O', 'CPT').

        Returns:
            Ranked list of ICDCandidate objects sourced exclusively from the local catalog.
            Returns an empty list (abstention) if no candidate meets the threshold.
        """
        k = top_k if top_k is not None else self.default_top_k
        threshold = min_score if min_score is not None else self.min_score_threshold

        clean_query = query.strip()
        if not clean_query:
            return []

        # Check in-memory query cache for instant sub-millisecond retrieval
        cache_key = (clean_query.lower(), k, threshold, system)
        if hasattr(self, "_query_cache") and cache_key in self._query_cache:
            return [c.model_copy() for c in self._query_cache[cache_key]]

        # Expand clinical abbreviations for dense semantic query embedding
        from medical_coding.retrieval.tokenizer import expand_medical_abbreviations

        expanded_query = expand_medical_abbreviations(clean_query)
        dense_query = f"{clean_query} {expanded_query}".strip() if expanded_query.lower() != clean_query.lower() else clean_query

        # Retrieve candidates from both local engines with an expanded initial window
        search_window = max(k * 10, 350)
        lexical_hits = self.lexical_retriever.retrieve(clean_query, top_k=search_window)
        vector_hits = self.vector_retriever.retrieve(dense_query, top_k=search_window)

        # Merge candidate pools by authoritative code
        merged: dict[str, dict[str, Any]] = {}

        for hit in lexical_hits:
            # Enforce local catalog existence if catalog is attached
            if self.catalog and not self.catalog.is_valid_code(hit.code):
                continue
            cat_rec = self.catalog.get_by_code(hit.code) if self.catalog else None
            hit_system = getattr(hit, "coding_system", None) or (cat_rec.coding_system if cat_rec else "ICD-10-CM")
            if system and hit_system != system:
                continue

            merged[hit.code] = {
                "code": hit.code,
                "description": hit.description,
                "is_valid_billable": hit.is_valid_billable,
                "category": hit.category,
                "lexical_score": hit.retrieval_score,
                "semantic_score": 0.0,
                "coding_system": hit_system,
            }

        for hit in vector_hits:
            if self.catalog and not self.catalog.is_valid_code(hit.code):
                continue
            cat_rec = self.catalog.get_by_code(hit.code) if self.catalog else None
            hit_system = getattr(hit, "coding_system", None) or (cat_rec.coding_system if cat_rec else "ICD-10-CM")
            if system and hit_system != system:
                continue

            if hit.code in merged:
                merged[hit.code]["semantic_score"] = hit.retrieval_score
                # Keep most descriptive text
                if len(hit.description) > len(merged[hit.code]["description"]):
                    merged[hit.code]["description"] = hit.description
            else:
                merged[hit.code] = {
                    "code": hit.code,
                    "description": hit.description,
                    "is_valid_billable": hit.is_valid_billable,
                    "category": hit.category,
                    "lexical_score": 0.0,
                    "semantic_score": hit.retrieval_score,
                    "coding_system": hit_system,
                }

        if not merged:
            logger.info(
                "Hybrid retrieval: No candidates found for query '%s' (abstaining).", clean_query
            )
            return []

        # Compute combined hybrid score
        candidate_pool: list[ICDCandidate] = []
        for code, info in merged.items():
            sem = info["semantic_score"]
            lex = info["lexical_score"]

            if sem > 0.0 and lex > 0.0:
                hybrid_score = (self.weight_semantic * sem) + (self.weight_lexical * lex)
            elif sem > 0.0:
                hybrid_score = sem * 0.85
            else:
                hybrid_score = lex * 0.85

            rounded_score = max(0.0, min(1.0, round(hybrid_score, 4)))

            # Filter against relevance threshold
            if rounded_score < threshold:
                continue

            # Determine retrieval method provenance
            if sem > 0.0 and lex > 0.0:
                method = "hybrid"
            elif sem > 0.0:
                method = "faiss"
            else:
                method = "bm25"

            candidate_pool.append(
                ICDCandidate(
                    code=code,
                    description=info["description"],
                    retrieval_score=rounded_score,
                    retrieval_method=method,
                    is_valid_billable=info["is_valid_billable"],
                    semantic_score=round(sem, 4),
                    lexical_score=round(lex, 4),
                    category=info["category"],
                    coding_system=info.get("coding_system", "ICD-10-CM"),
                )
            )

        # Sort descending by hybrid retrieval score
        candidate_pool.sort(key=lambda c: c.retrieval_score, reverse=True)

        top_candidates = candidate_pool[:k]

        if not top_candidates:
            logger.info(
                "Hybrid retrieval: %d candidates found but none met min_score threshold %.2f for query '%s' (abstaining).",
                len(candidate_pool),
                threshold,
                clean_query,
            )

        if hasattr(self, "_query_cache"):
            if len(self._query_cache) >= 1000:
                self._query_cache.clear()
            self._query_cache[cache_key] = [c.model_copy() for c in top_candidates]

        return top_candidates

    def retrieve_multi_system(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> dict[str, list[ICDCandidate]]:
        """Retrieve candidates partitioned by coding system (ICD-10-CM, ICD-O, CPT).

        Returns:
            Dictionary with keys 'ICD-10-CM', 'ICD-O', and 'CPT' containing top matching candidates.
        """
        return {
            "ICD-10-CM": self.retrieve(query, top_k=top_k, min_score=min_score, system="ICD-10-CM"),
            "ICD-O": self.retrieve(query, top_k=top_k, min_score=min_score, system="ICD-O"),
            "CPT": self.retrieve(query, top_k=top_k, min_score=min_score, system="CPT"),
        }

    def retrieve_with_retry(
        self,
        query: str,
        evidence: str = "",
        top_k: int | None = None,
        min_score: float | None = None,
        system: str | None = None,
        max_rounds: int = 3,
    ) -> list[ICDCandidate]:
        """Multi-round iterative BM25 lexical and dense vector search across query variations.

        Re-checks candidates for conditions with positive evidence that were not identified
        on the first pass.

        Round 1: Standard query retrieval.
        Round 2: Normalized query variations (core clinical entities, synonyms, expanded acronyms).
        Round 3: Evidence-anchored keywords and relaxed threshold matching.
        """
        k = top_k if top_k is not None else self.default_top_k
        threshold = min_score if min_score is not None else self.min_score_threshold

        from medical_coding.reasoning.concept_matcher import (
            compute_concept_match_score,
            normalize_for_concept_matching,
        )
        from medical_coding.utils.text import extract_search_query_variations

        # Round 1: Standard retrieval with original query
        round1_hits = self.retrieve(query, top_k=k, min_score=threshold, system=system)
        merged_candidates: dict[str, ICDCandidate] = {c.code: c for c in round1_hits}

        # Check if Round 1 already contains an exact match
        has_exact = False
        for c in round1_hits:
            _, _, is_exact = compute_concept_match_score(
                candidate_code=c.code,
                candidate_description=c.description,
                concept_name=query,
                evidence_text=evidence,
            )
            if is_exact:
                has_exact = True
                break

        # If no exact match in Round 1, expand across normalized query variations
        variations = extract_search_query_variations(query, evidence=evidence)

        # Round 2: Query variations (core clinical entities, synonyms, expanded acronyms)
        if (not has_exact or len(merged_candidates) < k) and max_rounds >= 2 and variations:
            relaxed_threshold_r2 = max(0.20, threshold - 0.10)
            for var in variations:
                if var.lower().strip() == query.lower().strip():
                    continue
                var_hits = self.retrieve(var, top_k=k, min_score=relaxed_threshold_r2, system=system)
                for h in var_hits:
                    if h.code not in merged_candidates or h.retrieval_score > merged_candidates[h.code].retrieval_score:
                        merged_candidates[h.code] = h
                        _, _, is_exact = compute_concept_match_score(
                            candidate_code=h.code,
                            candidate_description=h.description,
                            concept_name=query,
                            evidence_text=evidence,
                        )
                        if is_exact:
                            has_exact = True

        # Round 3: Evidence-anchored search if still below top_k or no exact match
        if (not has_exact or len(merged_candidates) < 5) and max_rounds >= 3 and evidence:
            relaxed_threshold_r3 = max(0.15, threshold - 0.15)
            ev_hits = self.retrieve(evidence[:120], top_k=k, min_score=relaxed_threshold_r3, system=system)
            for h in ev_hits:
                if h.code not in merged_candidates or h.retrieval_score > merged_candidates[h.code].retrieval_score:
                    merged_candidates[h.code] = h

        # Rescore all candidates based on clinical concept match to prevent returning first good-enough candidate
        scored_candidates: list[tuple[float, ICDCandidate]] = []
        for cand in merged_candidates.values():
            c_score, _, is_ex = compute_concept_match_score(
                candidate_code=cand.code,
                candidate_description=cand.description,
                concept_name=query,
                evidence_text=evidence,
            )
            composite_score = round(max(cand.retrieval_score, (0.80 * c_score) + (0.20 * cand.retrieval_score)), 4)
            if is_ex:
                composite_score = max(composite_score, 0.98)
            cand.retrieval_score = composite_score
            scored_candidates.append((composite_score, cand))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_candidates[:k]]

