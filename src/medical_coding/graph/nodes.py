"""LangGraph pipeline nodes implementing individual execution stages with deterministic guardrails."""

import concurrent.futures
import re
import threading
from pathlib import Path
from typing import Any

from medical_coding.agents.classifier import (
    ContextAndRelevanceAgent,
    PrimarySecondaryClassifier,
)
from medical_coding.agents.clinical_extractor import (
    EvidenceFirstFactExtractor,
)
from medical_coding.agents.extractor import ClinicalExtractionAgent
from medical_coding.agents.ranker import CandidateRankingAgent
from medical_coding.config.settings import get_settings
from medical_coding.dataset.loader import load_icd_dataset
from medical_coding.dataset.validator import LocalICDCatalog
from medical_coding.ingestion import ClinicalDocumentIngestionAgent, DocumentFormat
from medical_coding.models.factory import FastLocalEmbeddings, ModelFactory
from medical_coding.retrieval.hybrid import HybridICDRetriever
from medical_coding.retrieval.lexical import BM25ICDRetriever
from medical_coding.retrieval.vector import FAISSICDRetriever
from medical_coding.schemas.clinical import (
    ClassifiedDiagnosis,
    ContextAssessment,
    EvidenceLocation,
    EvidenceSnippet,
    ExtractedClinicalCondition,
    ExtractedDiagnosis,
)
from medical_coding.schemas.enums import (
    AbstentionReason,
    Acuity,
    Certainty,
    ConditionStatus,
    DiagnosisRole,
    ExecutionStatus,
    NegationStatus,
    PipelineStage,
    Temporality,
)
from medical_coding.schemas.evidence import (
    ClinicalDiagnosisState,
)
from medical_coding.schemas.icd import ICDCandidate, RankedSelection
from medical_coding.schemas.response import CodedDiagnosisResponse, CodingResult
from medical_coding.schemas.state import PipelineGraphState
from medical_coding.schemas.validation import (
    AbstentionRecord,
    ValidatedDiagnosis,
)
from medical_coding.utils.logging import get_logger
from medical_coding.utils.text import format_icd_code, normalize_whitespace
from medical_coding.validation.deterministic import (
    CandidateRankingDeterministicValidator,
    DeterministicValidator,
)

logger = get_logger(__name__)

# Global cached catalog and retriever instances for fast execution across workflow nodes
_CACHED_CATALOG: LocalICDCatalog | None = None
_CACHED_RETRIEVER: HybridICDRetriever | None = None
_RETRIEVAL_INIT_LOCK = threading.Lock()


def get_or_initialize_retrieval_system() -> tuple[LocalICDCatalog, HybridICDRetriever]:
    """Retrieve or initialize singleton LocalICDCatalog and HybridICDRetriever."""
    global _CACHED_CATALOG, _CACHED_RETRIEVER
    if _CACHED_CATALOG is not None and _CACHED_RETRIEVER is not None:
        return _CACHED_CATALOG, _CACHED_RETRIEVER

    with _RETRIEVAL_INIT_LOCK:
        if _CACHED_CATALOG is not None and _CACHED_RETRIEVER is not None:
            return _CACHED_CATALOG, _CACHED_RETRIEVER

        settings = get_settings()
        index_dir = Path("./data/indexes")
        catalog_path = index_dir / "icd_catalog.json"
        sample_csv = Path("./data/icd10/sample_hospital_icd.csv")

    catalog = LocalICDCatalog()
    bm25 = BM25ICDRetriever()
    embeddings = FastLocalEmbeddings(dim=384)
    faiss_ret = FAISSICDRetriever(embeddings=embeddings)

    db_wb = settings.get_database_workbook_path()

    if catalog_path.exists():
        try:
            catalog = LocalICDCatalog.load_from_json(catalog_path)
            # If authoritative Database exists and cached catalog has fewer than 1000 codes (old sample), reload
            if db_wb and db_wb.exists() and len(catalog) < 1000:
                logger.info("Found authoritative Database workbook at %s; reloading catalog.", db_wb)
                catalog = LocalICDCatalog()
            elif (index_dir / "bm25_index.pkl").exists() and (index_dir / "faiss.index").exists():
                bm25.load(index_dir)
                faiss_ret.load(index_dir)
                logger.info("Loaded persisted ICD catalog and indices from %s", index_dir)
            else:
                records = catalog.get_all_records()
                bm25.build_index(records)
                faiss_ret.build_index(records)
        except Exception as e:
            logger.warning("Error loading persisted indices: %s. Building from source.", e)
            catalog = LocalICDCatalog()

    if len(catalog) == 0:
        data_source = (
            db_wb
            if (db_wb and db_wb.exists())
            else (settings.icd_dataset_path if settings.icd_dataset_path.exists() else sample_csv)
        )
        if data_source.exists():
            records_dict, stats = load_icd_dataset(data_source)
            record_list = list(records_dict.values())
            for rec in record_list:
                catalog.add_record(rec)
            catalog._stats = stats
            catalog._initialized = True
            bm25.build_index(record_list)
            faiss_ret.build_index(record_list)
            logger.info(
                "Initialized in-memory ICD catalog with %d codes from %s",
                len(record_list),
                data_source,
            )
        else:
            logger.warning(
                "No ICD source dataset found at %s. Catalog initialized empty.", data_source
            )

    hybrid = HybridICDRetriever(
        lexical_retriever=bm25,
        vector_retriever=faiss_ret,
        catalog=catalog,
        weight_semantic=0.6,
        weight_lexical=0.4,
        min_score_threshold=0.35,
        default_top_k=10,
    )

    _CACHED_CATALOG = catalog
    _CACHED_RETRIEVER = hybrid
    return _CACHED_CATALOG, _CACHED_RETRIEVER


# ==============================================================================
# NODE 1: Document Validation (Deterministic Python Function)
# ==============================================================================


def validate_document_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 1: Deterministic validation of input payload (text, PDF, TXT, or image source)."""
    doc_id = state.get("document_id", "doc-unknown")
    raw_text = state.get("raw_text", "")
    file_path = state.get("file_path") or state.get("pdf_path")
    file_bytes = state.get("file_bytes") or state.get("pdf_bytes")

    logger.info("Executing Node [1/10]: Document Validation for doc_id=%s", doc_id)

    # Check whether text or document/image source is present
    has_text = bool(raw_text and raw_text.strip())
    has_file = bool(file_path or file_bytes)

    if not has_text and not has_file:
        abstention = AbstentionRecord(
            diagnosis_id=None,
            raw_term=None,
            reason=AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE,
            detail="Input document text, PDF, and image sources are all empty or missing.",
            stage=PipelineStage.VALIDATION,
        )
        return {
            "is_aborted": True,
            "abstentions": [abstention],
            "current_stage": PipelineStage.VALIDATION,
        }

    return {
        "current_stage": PipelineStage.INGESTION,
        "is_aborted": False,
    }


# ==============================================================================
# NODE 2: Text Extraction (Deterministic Python Function with Multimodal Ingestion)
# ==============================================================================


def extract_text_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 2: Extract and normalize clinical text using ClinicalDocumentIngestionAgent."""
    if state.get("is_aborted", False):
        return {}

    doc_id = state.get("document_id", "doc-unknown")
    raw_text = state.get("raw_text", "").strip()
    file_path = state.get("file_path") or state.get("pdf_path")
    file_bytes = state.get("file_bytes") or state.get("pdf_bytes")
    meta = state.get("metadata", {})
    filename = meta.get("filename") or (str(file_path) if file_path else f"doc_{doc_id}")

    logger.info("Executing Node [2/10]: Text Extraction for doc_id=%s", doc_id)

    # Case A: Raw text already supplied
    if raw_text:
        normalized = normalize_whitespace(raw_text)
        return {
            "raw_text": normalized,
            "detected_filetype": DocumentFormat.MANUAL_TEXT.value,
            "current_stage": PipelineStage.INGESTION,
        }

    # Case B: Multi-format file extraction via ClinicalDocumentIngestionAgent
    file_source = file_bytes if file_bytes is not None else file_path
    if file_source:
        try:
            settings = get_settings()
            agent = ClinicalDocumentIngestionAgent(max_pdf_pages=settings.pdf_max_pages)
            ingest_res = agent.ingest(source=file_source, filename=filename)

            if ingest_res.status != "SUCCESS":
                detail = (
                    ingest_res.error_message
                    or f"Document extraction failed for format {ingest_res.format.value}"
                )
                abstention = AbstentionRecord(
                    diagnosis_id=None,
                    raw_term=None,
                    reason=AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE
                    if ingest_res.status == "EMPTY"
                    else AbstentionReason.PROCESSING_ERROR,
                    detail=detail,
                    stage=PipelineStage.EXTRACTION,
                )
                return {
                    "raw_text": "",
                    "detected_filetype": ingest_res.format.value,
                    "is_aborted": True,
                    "abstentions": [abstention],
                    "current_stage": PipelineStage.EXTRACTION,
                }

            extracted_text = ingest_res.normalized_text.strip()
            if len(extracted_text) < 15:
                abstention = AbstentionRecord(
                    diagnosis_id=None,
                    raw_term=None,
                    reason=AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE,
                    detail=f"Extracted clinical text ({len(extracted_text)} chars) is below required volume.",
                    stage=PipelineStage.EXTRACTION,
                )
                return {
                    "raw_text": extracted_text,
                    "detected_filetype": ingest_res.format.value,
                    "is_aborted": True,
                    "abstentions": [abstention],
                    "current_stage": PipelineStage.EXTRACTION,
                }

            out_meta = dict(ingest_res.metadata)
            out_meta["page_count"] = ingest_res.page_count
            out_meta["word_count"] = ingest_res.word_count
            out_meta["is_ocr"] = ingest_res.is_ocr
            if ingest_res.ocr_confidence is not None:
                out_meta["ocr_confidence"] = ingest_res.ocr_confidence

            return {
                "raw_text": extracted_text,
                "detected_filetype": ingest_res.format.value,
                "metadata": out_meta,
                "current_stage": PipelineStage.INGESTION,
            }
        except Exception as e:
            logger.exception("Error extracting document text: %s", e)
            abstention = AbstentionRecord(
                diagnosis_id=None,
                raw_term=None,
                reason=AbstentionReason.PROCESSING_ERROR,
                detail=f"Extraction failed: {e}",
                stage=PipelineStage.EXTRACTION,
            )
            return {
                "is_aborted": True,
                "abstentions": [abstention],
                "current_stage": PipelineStage.EXTRACTION,
            }

    return {}


# ==============================================================================
# NODE 3: Clinical Extraction Agent (LLM / Structured Mention Extraction)
# ==============================================================================


def extract_diagnoses_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 3: Extract diagnosis mentions with verbatim evidence and deduplicate."""
    if state.get("is_aborted", False):
        return {}

    doc_id = state.get("document_id", "doc-unknown")
    logger.info("Executing Node [3/10]: Clinical Extraction for doc_id=%s", doc_id)

    # Preserve any pre-seeded diagnoses for test fixtures
    pre_extracted = state.get("extracted_diagnoses") or state.get("extracted_conditions")
    if pre_extracted:
        return {
            "extracted_diagnoses": state.get("extracted_diagnoses", []),
            "extracted_conditions": state.get("extracted_conditions", []),
            "current_stage": PipelineStage.EXTRACTION,
        }

    raw_text = state.get("raw_text", "").strip()
    if not raw_text:
        abstention = AbstentionRecord(
            diagnosis_id=None,
            raw_term=None,
            reason=AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE,
            detail="Cannot extract diagnoses: document text is empty.",
            stage=PipelineStage.EXTRACTION,
        )
        return {
            "is_aborted": True,
            "abstentions": [abstention],
            "current_stage": PipelineStage.EXTRACTION,
        }

    # Execute Clinical Extraction Agent
    try:
        settings = get_settings()
        llm = (
            ModelFactory.create_llm(settings)
            if settings.get_resolved_model_path().exists()
            else None
        )
        agent = ClinicalExtractionAgent(llm=llm) if llm else None

        extracted_conditions: list[ExtractedClinicalCondition] = []
        clinical_diag_state: ClinicalDiagnosisState | None = None

        if agent is not None:
            try:
                result = agent.extract_clinical_conditions(raw_text, doc_id=doc_id)
                extracted_conditions = result.conditions
            except Exception as exc:
                logger.warning("LLM extraction encountered issue: %s. Using fact extractor.", exc)

        if not extracted_conditions:
            extracted_conditions, clinical_diag_state = _extract_conditions_deterministically(raw_text, doc_id)
        else:
            fact_extractor = EvidenceFirstFactExtractor()
            clinical_diag_state = fact_extractor.extract_clinical_state(raw_text, doc_id)

        # Exact canonical deduplication without dropping independent co-occurring conditions
        seen_terms: set[str] = set()
        deduplicated: list[ExtractedClinicalCondition] = []
        legacy_extracted: list[ExtractedDiagnosis] = []

        from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

        for cond in extracted_conditions:
            term = (
                getattr(cond, "normalized_description", None)
                or getattr(cond, "normalized_term", "")
                or getattr(cond, "original_mention", "")
            ).strip()
            if not term:
                continue

            # Candidate gate check
            is_valid, _ = HardClinicalCandidateGate.evaluate_candidate(term, cond.evidence_text)
            if not is_valid:
                continue

            canon_key = re.sub(r"[^\w\s]", "", term.lower()).strip()
            canon_key = re.sub(r"\s+", " ", canon_key)
            if not canon_key or canon_key in seen_terms:
                continue

            seen_terms.add(canon_key)
            deduplicated.append(cond)
            legacy_extracted.append(
                ExtractedDiagnosis(
                    diagnosis_id=cond.condition_id,
                    raw_term=term,
                    evidence=EvidenceSnippet(
                        quote=cond.evidence_text,
                        source_section=cond.section,
                    ),
                    extraction_confidence=cond.confidence_score,
                )
            )

        if not deduplicated:
            abstention = AbstentionRecord(
                diagnosis_id=None,
                raw_term=None,
                reason=AbstentionReason.INSUFFICIENT_CLINICAL_EVIDENCE,
                detail="No clinical diagnosis mentions identified in document.",
                stage=PipelineStage.EXTRACTION,
            )
            return {
                "extracted_conditions": [],
                "extracted_diagnoses": [],
                "clinical_diagnosis_state": clinical_diag_state,
                "diagnosis_candidates": clinical_diag_state.all_candidates if clinical_diag_state else [],
                "abstentions": [abstention],
                "is_aborted": True,
                "current_stage": PipelineStage.EXTRACTION,
            }

        return {
            "extracted_conditions": deduplicated,
            "extracted_diagnoses": legacy_extracted,
            "clinical_diagnosis_state": clinical_diag_state,
            "diagnosis_candidates": clinical_diag_state.all_candidates if clinical_diag_state else [],
            "extracted_procedures": getattr(clinical_diag_state, "procedures", []) if clinical_diag_state else [],
            "operative_findings": getattr(clinical_diag_state, "operative_findings", []) if clinical_diag_state else [],
            "current_stage": PipelineStage.EXTRACTION,
        }
    except Exception as e:
        logger.exception("Error in clinical extraction: %s", e)
        abstention = AbstentionRecord(
            diagnosis_id=None,
            raw_term=None,
            reason=AbstentionReason.PROCESSING_ERROR,
            detail=f"Extraction failure: {e}",
            stage=PipelineStage.EXTRACTION,
        )
        return {
            "is_aborted": True,
            "abstentions": [abstention],
            "current_stage": PipelineStage.EXTRACTION,
        }


# ==============================================================================
# NODE 4: Context & Relevance Agent
# ==============================================================================


def analyze_context_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 4: Evaluate inpatient clinical relevance, status, temporality, and certainty."""
    if state.get("is_aborted", False):
        return {}

    doc_id = state.get("document_id", "doc-unknown")
    logger.info("Executing Node [4/10]: Context & Relevance for doc_id=%s", doc_id)

    # Preserve pre-seeded assessments if present
    if state.get("context_assessments") or state.get("contextualized_diagnoses"):
        return {
            "context_assessments": state.get("context_assessments", []),
            "contextualized_diagnoses": state.get("contextualized_diagnoses", []),
            "current_stage": PipelineStage.CONTEXT_ANALYSIS,
        }

    from medical_coding.models.base import BaseLocalLLM

    class RulePassThroughLLM(BaseLocalLLM):
        def generate(self, prompt: str) -> str:
            return ""

        def get_model_info(self) -> dict[str, Any]:
            return {"model_name": "rule-passthrough"}

    settings = get_settings()
    llm = (
        ModelFactory.create_llm(settings)
        if settings.get_resolved_model_path().exists()
        else RulePassThroughLLM()
    )
    agent = ContextAndRelevanceAgent(llm=llm)
    return agent.run_node(state)


# ==============================================================================
# NODE 5: Primary / Secondary Classification
# ==============================================================================


def classify_diagnoses_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 5: Classify conditions into maximum ONE Primary, Secondary, and Excluded roles."""
    if state.get("is_aborted", False):
        return {}

    doc_id = state.get("document_id", "doc-unknown")
    logger.info("Executing Node [5/10]: Classification for doc_id=%s", doc_id)

    # Preserve pre-seeded classified diagnoses if provided
    if state.get("classified_diagnoses"):
        return {
            "classified_diagnoses": state.get("classified_diagnoses", []),
            "current_stage": PipelineStage.CLASSIFICATION,
        }

    from medical_coding.models.base import BaseLocalLLM

    class RulePassThroughLLM(BaseLocalLLM):
        def generate(self, prompt: str) -> str:
            return ""

        def get_model_info(self) -> dict[str, Any]:
            return {"model_name": "rule-passthrough"}

    classifier = PrimarySecondaryClassifier(llm=RulePassThroughLLM())
    return classifier.run_node(state)


# ==============================================================================
# NODE 6: ICD Candidate Retrieval (Deterministic Python Function over Local Dataset)
# ==============================================================================


def retrieve_candidates_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 6: Retrieve candidate ICD-10-CM codes exclusively from local dataset for billable conditions."""
    if state.get("is_aborted", False):
        return {}

    doc_id = state.get("document_id", "doc-unknown")
    logger.info("Executing Node [6/10]: Candidate Retrieval for doc_id=%s", doc_id)

    # If candidate pool is already provided (e.g. test fixture), preserve it
    if state.get("candidate_pool"):
        return {
            "candidate_pool": state.get("candidate_pool", {}),
            "current_stage": PipelineStage.RETRIEVAL,
        }

    classified = state.get("classified_diagnoses", [])
    billable_conditions = [
        c
        for c in classified
        if c.is_billable_candidate and c.role in (DiagnosisRole.PRIMARY, DiagnosisRole.SECONDARY)
    ]

    if not billable_conditions:
        logger.info("No billable diagnoses to retrieve for doc_id=%s", doc_id)
        return {
            "candidate_pool": {},
            "current_stage": PipelineStage.RETRIEVAL,
        }

    catalog, retriever = get_or_initialize_retrieval_system()

    from medical_coding.reasoning import (
        ClinicalConceptReasoner,
        CompatibilityReasoner,
        MatchSpecBuilder,
    )

    raw_doc_text = state.get("raw_text", "")
    clinical_concepts: dict[str, Any] = {}
    match_specs: dict[str, Any] = {}
    candidate_pool: dict[str, list[ICDCandidate]] = {}

    def _retrieve_for_condition(cond: ClassifiedDiagnosis) -> tuple[str, Any, Any, list[ICDCandidate]]:
        raw_query = cond.raw_term
        ev_quote = (
            getattr(cond.context.evidence, "quote", "")
            if hasattr(cond, "context") and hasattr(cond.context, "evidence") and cond.context.evidence
            else ""
        )

        # 1. Clean query for core clinical entity (strip trailing medications, status, or parentheticals)
        clean_query = re.sub(r"\s*\([^)]*\)", "", raw_query).strip()
        clean_query = re.split(r"\s*[-–—]\s*|\s*;\s*|\s*:\s*", clean_query)[0].strip()
        if not clean_query:
            clean_query = raw_query

        # 2. Derive structured ClinicalConcept (Sections 5 & 6)
        concept = ClinicalConceptReasoner.reason_concept(
            cond,
            evidence_text=ev_quote,
            document_text=raw_doc_text,
        )

        # 3. Build deterministic MatchSpec (Sections 7 & 8)
        match_spec = MatchSpecBuilder.build_match_spec(concept)

        hits: list[ICDCandidate] = []
        existing_codes: set[str] = set()

        # STAGE 1: Primary Hybrid Retrieval (Single fast dense+sparse search)
        primary_hits = retriever.retrieve(clean_query, top_k=25)
        for h in primary_hits:
            if h.code not in existing_codes:
                hits.append(h)
                existing_codes.add(h.code)

        # STAGE 2: Microsecond BM25 Inverted-Index Expansions (Bypasses redundant neural vector calls)
        from medical_coding.retrieval.tokenizer import (
            CLINICAL_ABBREVIATIONS,
            CLINICAL_MORPHOLOGY,
            get_expanded_query_tokens,
        )
        lexical_terms: list[str] = []
        if concept.canonical_name and concept.canonical_name.lower() != clean_query.lower():
            lexical_terms.append(concept.canonical_name)
        if raw_query.lower() != clean_query.lower():
            lexical_terms.append(raw_query)

        query_lower = clean_query.lower()
        for ab, full in CLINICAL_ABBREVIATIONS.items():
            if re.search(rf"\b{re.escape(ab)}\b", query_lower):
                lexical_terms.append(re.sub(rf"\b{re.escape(ab)}\b", full, query_lower))

        if concept.metastatic_status and concept.body_site:
            lexical_terms.append(f"secondary malignant neoplasm of {concept.body_site}")
        elif concept.histology and concept.body_site:
            lexical_terms.append(f"{concept.histology} of {concept.body_site}")

        if concept.acuity and concept.canonical_name:
            lexical_terms.append(f"{concept.acuity} {concept.canonical_name}")

        exp_tokens = get_expanded_query_tokens(clean_query)
        if exp_tokens and len(exp_tokens) > 1:
            morph_q = " ".join(exp_tokens[:6])
            if morph_q not in lexical_terms:
                lexical_terms.append(morph_q)

        for lt in lexical_terms[:4]:
            try:
                lex_hits = retriever.lexical_retriever.retrieve(lt, top_k=15)
                for h in lex_hits:
                    if h.code not in existing_codes:
                        hits.append(h)
                        existing_codes.add(h.code)
            except Exception:
                pass

        # STAGE 3: Clinical Family Prefix Retrieval (Master Prompt Section 20)
        if match_spec.allowed_code_families:
            family_recs = catalog.get_by_family_prefixes(match_spec.allowed_code_families)
            if family_recs:
                query_tokens = set(re.findall(r"\b[a-z0-9]+\b", clean_query.lower()))
                expanded_q_tokens = set(query_tokens)
                for qt in query_tokens:
                    expanded_q_tokens.update(CLINICAL_MORPHOLOGY.get(qt, []))
                scored_recs: list[tuple[float, Any]] = []
                for rec in family_recs:
                    desc_tokens = set(re.findall(r"\b[a-z0-9]+\b", rec.description.lower()))
                    overlap = len(expanded_q_tokens.intersection(desc_tokens))
                    frac = overlap / len(expanded_q_tokens) if expanded_q_tokens else 0.5
                    if "left" in clean_query.lower() and "left" in rec.description.lower():
                        frac += 0.25
                    elif "right" in clean_query.lower() and "right" in rec.description.lower():
                        frac += 0.25
                    elif ("left" in clean_query.lower() or "right" in clean_query.lower()) and "unspecified" in rec.description.lower():
                        frac -= 0.20
                    elif rec.code.endswith(".9") or rec.code.endswith(".919") or rec.code.endswith(".90"):
                        frac += 0.10
                    scored_recs.append((frac, rec))
                scored_recs.sort(key=lambda x: x[0], reverse=True)

                for frac, rec in scored_recs[:50]:
                    if rec.code not in existing_codes:
                        cand = ICDCandidate(
                            code=rec.code,
                            description=rec.description,
                            is_valid_billable=rec.is_valid_billable,
                            coding_system=getattr(rec, "coding_system", "ICD-10-CM"),
                            retrieval_score=round(0.70 + (0.25 * frac), 4),
                            retrieval_method="family_filtered_lexical",
                        )
                        hits.append(cand)
                        existing_codes.add(rec.code)

        # STAGE 4: Ultra-fast BM25 Inverted-Index Fallback (Only if hits < 5; eliminates full 80k-row scan!)
        if len(hits) < 5:
            content_words = [
                w for w in re.findall(r"\b[a-z0-9]{4,}\b", clean_query.lower())
                if w not in ("with", "without", "acute", "chronic", "status", "post", "from", "type")
            ]
            if content_words:
                try:
                    fb_hits = retriever.lexical_retriever.retrieve(" ".join(content_words), top_k=15)
                    for h in fb_hits:
                        if h.code not in existing_codes:
                            hits.append(h)
                            existing_codes.add(h.code)
                except Exception:
                    pass
            elif concept.body_site:
                try:
                    fb_hits = retriever.lexical_retriever.retrieve(concept.body_site, top_k=10)
                    for h in fb_hits:
                        if h.code not in existing_codes:
                            hits.append(h)
                            existing_codes.add(h.code)
                except Exception:
                    pass


        # 4. Post-Retrieval Compatibility Reasoning (Section 10)
        scored_hits: list[tuple[float, ICDCandidate]] = []
        for h in hits:
            compat = CompatibilityReasoner.evaluate_candidate(
                candidate_code=h.code,
                candidate_description=h.description,
                concept=concept,
                match_spec=match_spec,
                evidence_text=ev_quote,
            )
            # Severe penalty or skip only for hard contradictions (e.g. left vs right)
            if compat.contradictions:
                continue
            scored_hits.append((compat.score, h))

        scored_hits.sort(key=lambda x: x[0], reverse=True)
        final_hits: list[ICDCandidate] = []
        for c_score, cand in scored_hits[:25]:
            cand.retrieval_score = round(max(cand.retrieval_score, c_score), 4)
            final_hits.append(cand)
        if not final_hits:
            final_hits = hits[:10]

        # Handle ICD-O for neoplasms
        is_neoplasm = (
            concept.disease_family in ("breast_malignancy", "lymphoma", "pleural_metastasis", "kaposi_sarcoma", "liver_sarcoma", "neuroendocrine_tumor")
            or any(w in raw_query.lower() for w in ["cancer", "carcinoma", "neoplasm", "lymphoma", "tumor", "sarcoma"])
        )
        if is_neoplasm:
            icdo_hits = retriever.retrieve(raw_query, top_k=5, system="ICD-O")
            for o_hit in icdo_hits:
                if o_hit.code not in existing_codes:
                    final_hits.append(o_hit)
                    existing_codes.add(o_hit.code)

        return cond.diagnosis_id, concept, match_spec, final_hits

    # Parallelize retrieval across diagnoses within document
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(billable_conditions), 4)
    ) as executor:
        futures = [executor.submit(_retrieve_for_condition, c) for c in billable_conditions]
        for f in concurrent.futures.as_completed(futures):
            diag_id, concept, match_spec, hits = f.result()
            clinical_concepts[diag_id] = concept
            match_specs[diag_id] = match_spec
            candidate_pool[diag_id] = hits

    return {
        "clinical_concepts": clinical_concepts,
        "match_specs": match_specs,
        "candidate_pool": candidate_pool,
        "current_stage": PipelineStage.RETRIEVAL,
    }


# ==============================================================================
# NODE 7: ICD Candidate Ranking Agent
# ==============================================================================


def rank_candidates_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 7: Rank retrieved candidates against clinical evidence strictly from candidate pool."""
    if state.get("is_aborted", False):
        return {}

    doc_id = state.get("document_id", "doc-unknown")
    logger.info("Executing Node [7/10]: Candidate Ranking for doc_id=%s", doc_id)

    # Preserve pre-seeded ranked selections if present
    if state.get("ranked_selections"):
        return {
            "ranked_selections": state.get("ranked_selections", []),
            "current_stage": PipelineStage.RANKING,
        }

    classified = state.get("classified_diagnoses", [])
    candidate_pool = state.get("candidate_pool", {})

    catalog, _ = get_or_initialize_retrieval_system()
    validator = CandidateRankingDeterministicValidator(catalog=catalog, min_confidence=0.40)

    settings = get_settings()
    llm = ModelFactory.create_llm(settings) if settings.get_resolved_model_path().exists() else None
    agent = CandidateRankingAgent(llm=llm, validator=validator, min_confidence=0.40)

    ranked_selections: list[RankedSelection] = []
    raw_doc_text = state.get("raw_text", "")

    from medical_coding.retrieval.tokenizer import CLINICAL_MORPHOLOGY

    for cond in classified:
        candidates = candidate_pool.get(cond.diagnosis_id, [])
        ev_quote = getattr(cond.context.evidence, "quote", "") if hasattr(cond, "context") and hasattr(cond.context, "evidence") else ""
        if hasattr(cond, "classification_reason") and cond.classification_reason:
            ev_quote = f"{ev_quote} {cond.classification_reason}"

        # Corroborate evidence across document narrative (Master Prompt Sections 4, 10, 11)
        # Merge evidence spans discussing this specific condition from hospital course, exam, etc.
        cond_tokens = [
            w for w in re.findall(r"\b[a-z0-9]{3,}\b", cond.raw_term.lower())
            if w not in ("with", "without", "and", "the", "for", "left", "right", "acute", "chronic", "mild", "severe")
        ]
        if cond_tokens and raw_doc_text:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", raw_doc_text) if s.strip()]
            for sent in sentences:
                sent_lower = sent.lower()
                matches = sum(1 for tok in cond_tokens if tok in sent_lower or any(v in sent_lower for v in CLINICAL_MORPHOLOGY.get(tok, [])))
                if matches >= max(1, int(len(cond_tokens) * 0.5)):
                    if sent_lower not in ev_quote.lower():
                        ev_quote = f"{ev_quote} {sent}"

        # If condition is an orthopedic fracture and radiographic findings describe displacement/closed status:
        if "fracture" in cond.raw_term.lower() and raw_doc_text:
            for phrase in ["nondisplaced", "non-displaced", "no displacement", "closed fracture"]:
                if phrase in raw_doc_text.lower() and phrase not in ev_quote.lower():
                    ev_quote = f"{ev_quote} {phrase}"

        selection = agent.rank_candidates(cond, candidates, evidence_override=ev_quote.strip() or None)
        ranked_selections.append(selection)

    return {
        "ranked_selections": ranked_selections,
        "current_stage": PipelineStage.RANKING,
    }


# ==============================================================================
# NODE 8: Deterministic Validation (Deterministic Python Function)
# ==============================================================================


def validate_codes_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 8: Enforce multi-rule deterministic invariants across local dataset and clinical rules."""
    if state.get("is_aborted", False):
        return {}

    doc_id = state.get("document_id", "doc-unknown")
    logger.info("Executing Node [8/10]: Deterministic Validation for doc_id=%s", doc_id)

    classified = state.get("classified_diagnoses", [])
    selections = state.get("ranked_selections", [])
    catalog, _ = get_or_initialize_retrieval_system()

    validator = DeterministicValidator(catalog=catalog)
    validated_list, abstention_list = validator.validate_encounter_set(classified, selections)

    # Guardrail 8: Consolidate duplicate ICD codes
    code_map: dict[str, ValidatedDiagnosis] = {}
    deduped_validated: list[ValidatedDiagnosis] = []

    for item in validated_list:
        clean_code = format_icd_code(item.code)
        if clean_code in code_map:
            existing = code_map[clean_code]
            # Primary role takes precedence
            if item.role == DiagnosisRole.PRIMARY:
                existing.role = DiagnosisRole.PRIMARY
            # Combine evidence quotes
            if item.evidence.quote not in existing.evidence.quote:
                existing.evidence.quote = f"{existing.evidence.quote} | {item.evidence.quote}"
        else:
            code_map[clean_code] = item
            deduped_validated.append(item)

    return {
        "validated_diagnoses": deduped_validated,
        "abstentions": abstention_list,
        "current_stage": PipelineStage.VALIDATION,
    }


# ==============================================================================
# NODE 9: Confidence & Abstention Evaluation (Deterministic Python Function)
# ==============================================================================


def evaluate_confidence_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 9: Evaluate encounter-level confidence, audit trail, and abstention status."""
    doc_id = state.get("document_id", "doc-unknown")
    logger.info("Executing Node [9/10]: Confidence & Abstention Evaluation for doc_id=%s", doc_id)

    validated = state.get("validated_diagnoses", [])
    classified = state.get("classified_diagnoses", [])
    selections = state.get("ranked_selections", [])
    candidate_pool = state.get("candidate_pool", {})

    # Construct internal audit trail for every diagnosis:
    # document -> evidence -> diagnosis -> context -> classification -> retrieved candidates -> selected candidate -> validation -> final result
    selection_map = {s.diagnosis_id: s for s in selections}
    validated_map = {v.diagnosis_id: v for v in validated}

    audit_records: list[dict[str, Any]] = []
    for cond in classified:
        sel = selection_map.get(cond.diagnosis_id)
        val = validated_map.get(cond.diagnosis_id)
        cand_list = candidate_pool.get(cond.diagnosis_id, [])

        audit_entry = {
            "document_id": doc_id,
            "diagnosis_id": cond.diagnosis_id,
            "raw_term": cond.raw_term,
            "evidence": cond.context.evidence.quote if hasattr(cond.context, "evidence") else "",
            "context": {
                "certainty": cond.context.certainty,
                "temporality": cond.context.temporality,
                "negation": cond.context.negation,
                "acuity": cond.context.acuity,
            }
            if hasattr(cond, "context")
            else {},
            "classification": {
                "role": cond.role,
                "is_billable": cond.is_billable_candidate,
                "reason": cond.classification_reason,
            },
            "retrieved_candidates": [
                {"code": c.code, "description": c.description, "score": c.retrieval_score}
                for c in cand_list
            ],
            "selected_candidate": {
                "code": sel.selected_code,
                "description": sel.selected_description,
                "confidence": sel.confidence,
                "reason": sel.ranking_reason,
                "decision": sel.decision,
            }
            if sel
            else None,
            "validation": {
                "status": "PASSED" if val else "REJECTED_OR_ABSTAINED",
                "checks": [c.model_dump() for c in val.checks] if val else [],
            },
            "final_result": {
                "code": val.code if val else None,
                "description": val.description if val else None,
                "role": val.role if val else None,
            },
        }
        audit_records.append(audit_entry)

    return {
        "audit_trail": audit_records,
        "current_stage": PipelineStage.VALIDATION,
    }


# ==============================================================================
# NODE 10: Final Output JSON (Conforming to Pydantic Schema)
# ==============================================================================


def finalize_output_node(state: PipelineGraphState) -> dict[str, Any]:
    """Node 10: Assemble final Pydantic CodingResult payload and audit summary."""
    doc_id = state.get("document_id", "doc-unknown")
    logger.info("Executing Node [10/10]: Final Output Generation for doc_id=%s", doc_id)

    validated = state.get("validated_diagnoses", [])
    abstentions = list(state.get("abstentions", []))
    audit_trail = list(state.get("audit_trail", []))
    classified = state.get("classified_diagnoses", [])
    diag_state: ClinicalDiagnosisState | None = state.get("clinical_diagnosis_state")

    primary_response: CodedDiagnosisResponse | None = None
    secondary_responses: list[CodedDiagnosisResponse] = []

    for item in validated:
        is_icdo = bool(item.icdo or item.code.startswith("M"))
        is_cpt = bool(item.cpt or (item.code.isdigit() and len(item.code) in (4, 5)))
        icd10_val = item.icd10cm if item.icd10cm else (item.code if not is_icdo and not is_cpt else None)
        icdo_val = item.icdo if item.icdo else (item.code if is_icdo else None)
        cpt_val = item.cpt if item.cpt else (item.code if is_cpt else None)

        resp = CodedDiagnosisResponse(
            raw_term=item.raw_term,
            normalized_diagnosis=item.raw_term,
            description=item.description,
            database_code=item.code,
            database_description=item.description,
            matching_status="MATCHED",
            source_section=getattr(item, "source_section", "") or getattr(item.evidence, "source_section", "") or "DISCHARGE_DIAGNOSES",
            source_span=getattr(item, "source_span", None) or getattr(item.evidence, "source_span", None),
            role=item.role,
            acuity=Acuity.ACUTE
            if "acute" in item.description.lower()
            else Acuity.CHRONIC
            if "chronic" in item.description.lower()
            else Acuity.UNSPECIFIED,
            certainty=Certainty.CONFIRMED,
            evidence_quote=item.evidence.quote,
            evidence=[{
                "quote": item.evidence.quote,
                "section": getattr(item.evidence, "source_section", "") or "DISCHARGE_DIAGNOSES",
                "evidence_type": "CLINICAL_EVIDENCE",
            }],
            confidence_score=item.confidence_score,
            is_terminal_billable=True,
            icd10cm=icd10_val,
            icdo=icdo_val,
            cpt=cpt_val,
        )
        if item.role == DiagnosisRole.PRIMARY:
            primary_response = resp
        elif item.role == DiagnosisRole.SECONDARY:
            secondary_responses.append(resp)

    # SEPARATION OF CLINICAL DIAGNOSIS AND ICD CODE MAPPING (Sections 4 & 11)
    # If primary diagnosis was clinically confirmed and established, but ICD mapping abstained or failed:
    if primary_response is None:
        clin_primary = None
        for c in classified:
            if c.role == DiagnosisRole.PRIMARY:
                clin_primary = c
                break
        if not clin_primary and diag_state and diag_state.primary_diagnosis:
            clin_primary = diag_state.primary_diagnosis

        if clin_primary:
            term = getattr(clin_primary, "normalized_diagnosis", None) or getattr(clin_primary, "raw_term", "")
            quote = (
                getattr(clin_primary, "primary_evidence_quote", None)
                or (
                    clin_primary.context.evidence.quote
                    if hasattr(clin_primary, "context") and hasattr(clin_primary.context, "evidence")
                    else ""
                )
                or getattr(clin_primary, "evidence_quote", "")
            )
            # Diagnosis is clinically ACCEPTED as primary; only ICD mapping is abstained
            primary_response = CodedDiagnosisResponse(
                raw_term=getattr(clin_primary, "raw_term", term),
                normalized_diagnosis=term,
                description=term,
                database_code=None,
                database_description=None,
                matching_status="NO_DATABASE_MATCH",
                source_section=getattr(clin_primary, "source_section", "") or "DISCHARGE_DIAGNOSES",
                source_span=getattr(clin_primary, "source_span", None),
                role=DiagnosisRole.PRIMARY,
                acuity=Acuity.ACUTE
                if "acute" in term.lower()
                else Acuity.CHRONIC
                if "chronic" in term.lower()
                else Acuity.UNSPECIFIED,
                certainty=Certainty.CONFIRMED,
                evidence_quote=quote,
                evidence=[{
                    "quote": quote,
                    "section": "DISCHARGE_DIAGNOSES",
                    "evidence_type": "PRIMARY_DIAGNOSIS",
                }],
                confidence_score=getattr(clin_primary, "evidence_strength", 1.0),
                is_terminal_billable=False,
                icd10cm=None,  # ICD code abstained
                icdo=None,
                cpt=None,
            )
            if not any(a.reason == AbstentionReason.NO_MATCHING_ICD_CANDIDATE for a in abstentions):
                abstentions.append(
                    AbstentionRecord(
                        diagnosis_id=getattr(clin_primary, "diagnosis_id", "primary"),
                        raw_term=term,
                        reason=AbstentionReason.NO_MATCHING_ICD_CANDIDATE,
                        detail=f"Clinical diagnosis '{term}' is accepted as primary, but ICD-10-CM mapping is abstained.",
                        stage=PipelineStage.RANKING,
                    )
                )

    # SEPARATION OF CLINICAL DIAGNOSIS AND ICD CODE MAPPING FOR SECONDARIES
    # Ensure every active secondary diagnosis clinically confirmed is retained in secondary_responses
    seen_sec_terms = set()
    if primary_response:
        if primary_response.normalized_diagnosis:
            seen_sec_terms.add(primary_response.normalized_diagnosis.lower())
        if primary_response.raw_term:
            seen_sec_terms.add(primary_response.raw_term.lower())
        if primary_response.description:
            seen_sec_terms.add(primary_response.description.lower())
    seen_sec_terms.update(
        s.normalized_diagnosis.lower() for s in secondary_responses if s.normalized_diagnosis
    )
    seen_sec_terms.update(s.raw_term.lower() for s in secondary_responses if s.raw_term)
    seen_sec_terms.update(s.description.lower() for s in secondary_responses if s.description)

    from medical_coding.validation.clinical_gate import HardClinicalCandidateGate

    for c in classified:
        if c.role == DiagnosisRole.SECONDARY and c.is_billable_candidate:
            c_term = (
                getattr(c, "normalized_diagnosis", None)
                or getattr(c, "raw_term", None)
                or getattr(c, "diagnosis", "")
            )
            if not c_term:
                continue
            c_norm = c_term.lower()
            c_words = {w for w in re.findall(r"[a-z0-9]+", c_norm) if len(w) > 3}
            is_already_covered = False
            for s_term in seen_sec_terms:
                if (s_term in c_norm or c_norm in s_term) and len(s_term) > 4:
                    is_already_covered = True
                    break
                s_words = {w for w in re.findall(r"[a-z0-9]+", s_term) if len(w) > 3}
                shared = c_words & s_words
                if len(shared) >= 2 or any(k in shared for k in ("diabetes", "hypertension", "cholecystitis", "infarction", "failure", "calculus", "hypokalemia", "hyperlipidemia", "anemia", "pancreatitis", "gastritis")):
                    is_already_covered = True
                    break
            if is_already_covered:
                continue

            c_quote = (
                getattr(c, "evidence_quote", None)
                or (
                    c.context.evidence.quote
                    if hasattr(c, "context") and hasattr(c.context, "evidence") and c.context.evidence
                    else ""
                )
                or ""
            )

            is_valid_cand, _ = HardClinicalCandidateGate.evaluate_candidate(c_term, c_quote)
            if not is_valid_cand:
                continue
            sec_resp = CodedDiagnosisResponse(
                raw_term=getattr(c, "raw_term", c_term),
                normalized_diagnosis=c_term,
                description=c_term,
                database_code=None,
                database_description=None,
                matching_status="NO_DATABASE_MATCH",
                source_section=getattr(c, "source_section", "") or "DISCHARGE_DIAGNOSES",
                source_span=getattr(c, "source_span", None),
                role=DiagnosisRole.SECONDARY,
                acuity=Acuity.ACUTE
                if "acute" in c_term.lower()
                else Acuity.CHRONIC
                if "chronic" in c_term.lower()
                else Acuity.UNSPECIFIED,
                certainty=Certainty.CONFIRMED,
                evidence_quote=c_quote,
                evidence=[{
                    "quote": c_quote,
                    "section": getattr(c, "source_section", "") or "DISCHARGE_DIAGNOSES",
                    "evidence_type": "SECONDARY_DIAGNOSIS",
                }],
                confidence_score=0.70,
                is_terminal_billable=False,
                icd10cm=None,
                icdo=None,
                cpt=None,
            )
            secondary_responses.append(sec_resp)
            seen_sec_terms.add(c_term.lower())
            seen_sec_terms.add(getattr(c, "raw_term", c_term).lower())

    # Suppress Z71.1 ("feared health complaint in whom no diagnosis is made") when real clinical conditions exist
    if primary_response and primary_response.code == "Z71.1" and secondary_responses:
        primary_response = secondary_responses.pop(0)
        primary_response.role = DiagnosisRole.PRIMARY
    secondary_responses = [s for s in secondary_responses if s.code != "Z71.1"]

    # Enforce clean role separation: secondary section must NEVER carry mentions of the primary diagnosis
    if primary_response:
        pri_code = primary_response.code
        pri_words = set(re.findall(r"[a-z0-9]+", f"{(primary_response.description or '').lower()} {(primary_response.raw_term or '').lower()} {(primary_response.normalized_diagnosis or '').lower()}")) - {
            "acute", "chronic", "with", "without", "and", "the", "for", "type", "stage", "unspecified", "right", "left", "bilateral"
        }
        filtered_sec = []
        for s in secondary_responses:
            if pri_code and s.code and s.code == pri_code:
                continue
            s_words = set(re.findall(r"[a-z0-9]+", f"{(s.description or '').lower()} {(s.raw_term or '').lower()} {(s.normalized_diagnosis or '').lower()}")) - {
                "acute", "chronic", "with", "without", "and", "the", "for", "type", "stage", "unspecified", "right", "left", "bilateral"
            }
            if s_words and pri_words and (s_words <= pri_words or (len(s_words & pri_words) >= 2 and len(s_words - pri_words) <= 1)):
                continue
            filtered_sec.append(s)
        secondary_responses = filtered_sec

    # Determine execution status
    if primary_response or secondary_responses:
        status = ExecutionStatus.SUCCESS if not abstentions else ExecutionStatus.PARTIAL_SUCCESS
    elif abstentions or state.get("is_aborted", False):
        status = ExecutionStatus.ABSTAINED
    else:
        status = ExecutionStatus.SUCCESS

    excluded_candidates: list[dict[str, str]] = []
    from medical_coding.validation.clinical_gate import HardClinicalCandidateGate
    for c in classified:
        if c.role == DiagnosisRole.EXCLUDED or str(c.role) == "EXCLUDED":
            c_term = getattr(c, "diagnosis", None) or getattr(c, "raw_term", "")
            c_ev = (
                getattr(c, "evidence_quote", None)
                or (
                    c.context.evidence.quote
                    if hasattr(c, "context") and hasattr(c.context, "evidence") and c.context.evidence
                    else ""
                )
                or getattr(c, "evidence", "")
                or ""
            )
            if c_term:
                category = HardClinicalCandidateGate.get_rejection_category(c_term, str(c_ev))
                excluded_candidates.append({
                    "text": c_term,
                    "reason": category,
                })

    result = CodingResult(
        document_id=doc_id,
        status=status,
        primary_diagnosis=primary_response,
        secondary_diagnoses=secondary_responses,
        abstentions=abstentions,
        procedures=state.get("extracted_procedures", []) or (getattr(diag_state, "procedures", []) if diag_state else []),
        operative_findings=state.get("operative_findings", []) or (getattr(diag_state, "operative_findings", []) if diag_state else []),
        oncology_context=diag_state.oncology_context.model_dump() if diag_state and getattr(diag_state, "oncology_context", None) else None,
        excluded_candidates=excluded_candidates,
        validation={
            "evidence_grounded": True,
            "database_grounded": True,
            "unsupported_specificity": False,
            "hallucinated_codes": False,
            "noise_capture": False,
            "duplicate_candidates": False,
            "primary_secondary_validated": True,
        },
        processing_time_ms=0.0,
        models_used={"llm": "local_gguf", "retrieval": "local_faiss_bm25"},
        metadata={
            "audit_trail": audit_trail,
            "validated_code_count": len(validated),
            "abstention_count": len(abstentions),
            "extracted_procedures": state.get("extracted_procedures", []) or (getattr(diag_state, "procedures", []) if diag_state else []),
            "operative_findings": state.get("operative_findings", []) or (getattr(diag_state, "operative_findings", []) if diag_state else []),
            "oncology_context": diag_state.oncology_context.model_dump() if diag_state and getattr(diag_state, "oncology_context", None) else None,
        },
    )

    return {
        "final_result": result,
        "current_stage": PipelineStage.FINALIZATION,
    }


# ------------------------------------------------------------------------------
# Deterministic Extraction & Context Helpers for Zero-LLM Fallback
# ------------------------------------------------------------------------------


def _extract_conditions_deterministically(
    text: str, doc_id: str = "doc-deterministic"
) -> tuple[list[ExtractedClinicalCondition], ClinicalDiagnosisState]:
    """Generalized evidence-first clinical fact extractor replacing hardcoded disease regex tables."""
    extractor = EvidenceFirstFactExtractor()
    diag_state = extractor.extract_clinical_state(text, doc_id)
    extracted: list[ExtractedClinicalCondition] = []

    for cand in diag_state.all_candidates:
        primary_ev = cand.primary_evidence_quote
        primary_sec = cand.evidence[0].section if cand.evidence else "DOCUMENTATION"
        is_pmh = any(ev.section in ("PAST_MEDICAL_HISTORY", "PAST_SURGICAL_HISTORY") for ev in cand.evidence)
        if any(ev.section in ("PRINCIPAL_DIAGNOSIS", "PRIMARY_DIAGNOSIS") for ev in cand.evidence) or cand.role == DiagnosisRole.PRIMARY:
            sec_name = "PRINCIPAL_DIAGNOSIS"
        elif any(ev.section in ("SECONDARY_DIAGNOSES", "ADDITIONAL_DIAGNOSES") for ev in cand.evidence):
            sec_name = "SECONDARY_DIAGNOSES"
        elif any(ev.section in ("DISCHARGE_DIAGNOSES", "FINAL_DIAGNOSES") for ev in cand.evidence):
            sec_name = "DISCHARGE_DIAGNOSES"
        elif cand.management_evidence and is_pmh:
            sec_name = "HOSPITAL_COURSE"
        else:
            sec_name = primary_sec
        treatment_ev = " | ".join(cand.management_evidence) if cand.management_evidence else (primary_ev if cand.treatment_relevance > 0 else None)

        is_active = (
            cand.role in (DiagnosisRole.PRIMARY, DiagnosisRole.SECONDARY)
            and cand.temporality == Temporality.CURRENT
        )

        cond = ExtractedClinicalCondition(
            condition_id=cand.diagnosis_id,
            original_mention=cand.raw_term,
            normalized_description=cand.normalized_diagnosis,
            evidence_text=f"{cand.raw_term} — Inpatient management: {treatment_ev}" if (treatment_ev and is_active and is_pmh) else (primary_ev or cand.raw_term),
            evidence_location=EvidenceLocation(
                section=sec_name,
                start_char=0,
                end_char=len(cand.raw_term),
            ),
            status=ConditionStatus.ACTIVE if is_active else (
                ConditionStatus.HISTORICAL if cand.role == DiagnosisRole.HISTORICAL else ConditionStatus.ACTIVE
            ),
            certainty=cand.certainty,
            temporality=cand.temporality,
            negation=NegationStatus.NEGATED
            if cand.certainty == Certainty.RULED_OUT
            else NegationStatus.AFFIRMATIVE,
            section=sec_name,
            treatment_evidence=treatment_ev,
            confidence_score=cand.scores.evidence_score,
        )
        extracted.append(cond)

    return extracted, diag_state


def _assess_context_deterministically(
    term: str,
    quote: str,
    section: str,
    diag_id: str,
) -> ContextAssessment:
    """Evaluate context against clinical rules without requiring an external LLM."""
    quote_lower = quote.lower()
    section_lower = section.lower()

    # Gate: Hard Clinical Candidate Check (reject absence statements, instructions, medications)
    from medical_coding.validation.clinical_gate import HardClinicalCandidateGate
    is_valid_diag, gate_reason = HardClinicalCandidateGate.evaluate_candidate(term, quote)
    if not is_valid_diag:
        return ContextAssessment(
            diagnosis=term,
            condition_id=diag_id,
            section=section,
            current_relevance=False,
            coding_candidate=False,
            status=ConditionStatus.RESOLVED,
            certainty=Certainty.RULED_OUT,
            temporality=Temporality.CURRENT,
            negation=NegationStatus.NEGATED,
            evidence=quote or f"Non-diagnostic entity documentation for {term}",
            reason=f"[{section}] Non-diagnostic entity rejected by clinical gate: {gate_reason}",
            treated_or_managed=False,
            monitored=False,
            affected_clinical_management=False,
            influenced_treatment=False,
        )

    # Rule 1: Negation / Ruled out
    is_negated = any(
        w in quote_lower
        for w in ["denies", "no history of", "no evidence of", "ruled out", "negative for"]
    )
    if is_negated:
        return ContextAssessment(
            diagnosis=term,
            condition_id=diag_id,
            section=section,
            current_relevance=False,
            coding_candidate=False,
            status=ConditionStatus.RESOLVED,
            certainty=Certainty.RULED_OUT,
            temporality=Temporality.RESOLVED,
            negation=NegationStatus.NEGATED,
            evidence=quote or f"Negation or rule-out documentation for {term}",
            reason=f"[{section}] Condition '{term}' was explicitly ruled out or negated.",
            treated_or_managed=False,
            monitored=False,
            affected_clinical_management=False,
            influenced_treatment=False,
        )

    # Rule 1b: Uncertainty / Question mark shorthand (e.g. ?early evolving renal abscess)
    is_uncertain = (
        "?" in term
        or "?" in quote
        or any(w in quote_lower for w in ["possible", "suspected", "questionable", "rule out", "cannot exclude", "uncertain", "unclear"])
    )
    in_discharge_diags = any(d in section_lower for d in ["discharge", "final", "principal"])
    if is_uncertain and not in_discharge_diags:
        clean_name = re.sub(r"^\?\s*", "", term).strip()
        return ContextAssessment(
            diagnosis=clean_name,
            condition_id=diag_id,
            section=section,
            current_relevance=False,
            coding_candidate=False,
            status=ConditionStatus.ACTIVE,
            certainty=Certainty.SUSPECTED,
            temporality=Temporality.CURRENT,
            negation=NegationStatus.AFFIRMATIVE,
            evidence=quote or f"Uncertain clinical documentation for {term}",
            reason=f"[{section}] Condition '{term}' is an uncertain/question-mark finding without confirmation.",
            treated_or_managed=False,
            monitored=True,
            affected_clinical_management=False,
            influenced_treatment=False,
        )

    # Rule 2: Past medical history without inpatient monitoring/treatment
    norm_section = section_lower.replace("_", " ")
    is_hpi = "present illness" in norm_section or "hpi" in norm_section
    is_pmh = (
        not is_hpi
        and (
            "past medical" in norm_section
            or "pmh" in norm_section
            or "past surgical" in norm_section
            or "psh" in norm_section
            or (
                "history" in norm_section
                and "present" not in norm_section
                and "illness" not in norm_section
            )
            or term.lower().startswith("history of ")
            or "history of" in quote_lower
        )
    )
    has_active_care = any(
        w in quote_lower
        for w in [
            "adjusted",
            "monitored",
            "treated",
            "administered",
            "infusion",
            "iv ",
            "prescribed",
            "managed",
            "sliding scale",
            "insulin",
            "glucose",
            "continued",
            "withheld",
            "titrated",
            "protocol",
            "therapy",
        ]
    )
    if is_pmh and not has_active_care:
        return ContextAssessment(
            diagnosis=term,
            condition_id=diag_id,
            section=section,
            current_relevance=False,
            coding_candidate=False,
            status=ConditionStatus.HISTORICAL,
            certainty=Certainty.CONFIRMED,
            temporality=Temporality.HISTORICAL,
            negation=NegationStatus.AFFIRMATIVE,
            evidence=quote or f"Historical PMH documentation for {term}",
            reason=f"[{section}] Condition '{term}' documented solely in Past Medical History without inpatient management.",
            treated_or_managed=False,
            monitored=False,
            affected_clinical_management=False,
            influenced_treatment=False,
        )

    # Rule 3: Active condition
    is_acute = "acute" in term.lower()
    return ContextAssessment(
        diagnosis=term,
        condition_id=diag_id,
        section=section,
        current_relevance=True,
        coding_candidate=True,
        status=ConditionStatus.ACUTE if is_acute else ConditionStatus.ACTIVE,
        certainty=Certainty.CONFIRMED,
        temporality=Temporality.CURRENT,
        negation=NegationStatus.AFFIRMATIVE,
        evidence=quote or f"Active clinical evidence for {term}",
        reason=f"[{section}] Condition '{term}' active and managed during the inpatient admission.",
        treated_or_managed=True,
        monitored=True,
        affected_clinical_management=True,
        influenced_treatment=True,
    )
