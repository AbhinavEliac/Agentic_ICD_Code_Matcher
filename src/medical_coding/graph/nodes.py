"""LangGraph pipeline nodes implementing individual execution stages with deterministic guardrails."""

import concurrent.futures
import threading
from pathlib import Path
from typing import Any

from medical_coding.agents.classifier import (
    ContextAndRelevanceAgent,
    PrimarySecondaryClassifier,
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
from medical_coding.schemas.icd import ICDCandidate, RankedSelection
from medical_coding.schemas.response import CodedDiagnosisResponse, CodingResult
from medical_coding.schemas.state import PipelineGraphState
from medical_coding.schemas.validation import (
    AbstentionRecord,
    ValidatedDiagnosis,
)
from medical_coding.utils.logging import get_logger
from medical_coding.utils.text import format_icd_code, normalize_whitespace
from medical_coding.agents.clinical_extractor import (
    ClinicalDocumentSection,
    EvidenceFirstFactExtractor,
    SectionSegmenter,
)
from medical_coding.schemas.evidence import (
    AuditTrailEntry,
    ClinicalDiagnosisCandidate,
    ClinicalDiagnosisState,
    ICDMappingState,
    MultiDimensionalScore,
    StructuredEvidence,
)
from medical_coding.validation.deterministic import (
    CandidateRankingDeterministicValidator,
    DeterministicValidator,
)
from medical_coding.validation.gates import ValidationGateEngine

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

        # Generalized subsumption check without hardcoded disease lists
        seen_terms: set[str] = set()
        deduplicated: list[ExtractedClinicalCondition] = []
        legacy_extracted: list[ExtractedDiagnosis] = []

        sorted_conditions = sorted(
            extracted_conditions,
            key=lambda c: len(
                getattr(c, "normalized_description", None) or getattr(c, "normalized_term", "")
            ),
            reverse=True,
        )

        for cond in sorted_conditions:
            term = (
                getattr(cond, "normalized_description", None)
                or getattr(cond, "normalized_term", "")
            ).strip()
            key = term.lower()
            if not key:
                continue

            is_subsumed = False
            for seen in seen_terms:
                if key == seen:
                    is_subsumed = True
                    break
                if len(key) < len(seen) and key in seen:
                    is_subsumed = True
                    break

            if is_subsumed:
                continue

            seen_terms.add(key)
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

    _, retriever = get_or_initialize_retrieval_system()

    candidate_pool: dict[str, list[ICDCandidate]] = {}

    def _retrieve_for_condition(cond: ClassifiedDiagnosis) -> tuple[str, list[ICDCandidate]]:
        query = cond.raw_term
        hits = retriever.retrieve(query, top_k=25)

        term_lower = query.lower()
        evidence_lower = (
            cond.context.evidence.quote.lower()
            if hasattr(cond, "context") and hasattr(cond.context, "evidence") and cond.context.evidence
            else ""
        )
        combined_text = f"{term_lower} {evidence_lower}"

        is_neoplasm = any(
            w in combined_text
            for w in [
                "cancer", "carcinoma", "sarcoma", "melanoma", "tumor", "tumour",
                "neoplasm", "lymphoma", "leukemia", "malignant", "infiltrating duct"
            ]
        )
        is_procedure = any(
            w in combined_text
            for w in [
                "biopsy", "excision", "resection", "catheterization", "infusion",
                "endoscopy", "surgery", "repair", "graft", "consultation", "evaluation",
                "treatment", "injection", "procedure"
            ]
        )

        existing_codes = {h.code for h in hits}
        if is_neoplasm:
            icdo_hits = retriever.retrieve(query, top_k=5, system="ICD-O")
            for o_hit in icdo_hits:
                if o_hit.code not in existing_codes:
                    hits.append(o_hit)
                    existing_codes.add(o_hit.code)

        if is_procedure:
            cpt_hits = retriever.retrieve(query, top_k=5, system="CPT")
            for c_hit in cpt_hits:
                if c_hit.code not in existing_codes:
                    hits.append(c_hit)
                    existing_codes.add(c_hit.code)

        return cond.diagnosis_id, hits

    # Parallelize retrieval across diagnoses within document
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(billable_conditions), 4)
    ) as executor:
        futures = [executor.submit(_retrieve_for_condition, c) for c in billable_conditions]
        for f in concurrent.futures.as_completed(futures):
            diag_id, hits = f.result()
            candidate_pool[diag_id] = hits

    return {
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

    for cond in classified:
        candidates = candidate_pool.get(cond.diagnosis_id, [])
        selection = agent.rank_candidates(cond, candidates)
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
        clin_primary = diag_state.primary_diagnosis if (diag_state and diag_state.primary_diagnosis) else None
        if not clin_primary:
            for c in classified:
                if c.role == DiagnosisRole.PRIMARY:
                    clin_primary = c
                    break

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

    # Determine execution status
    if primary_response or secondary_responses:
        status = ExecutionStatus.SUCCESS if not abstentions else ExecutionStatus.PARTIAL_SUCCESS
    elif abstentions or state.get("is_aborted", False):
        status = ExecutionStatus.ABSTAINED
    else:
        status = ExecutionStatus.SUCCESS

    result = CodingResult(
        document_id=doc_id,
        status=status,
        primary_diagnosis=primary_response,
        secondary_diagnoses=secondary_responses,
        abstentions=abstentions,
        processing_time_ms=0.0,
        models_used={"llm": "local_gguf", "retrieval": "local_faiss_bm25"},
        metadata={
            "audit_trail": audit_trail,
            "validated_code_count": len(validated),
            "abstention_count": len(abstentions),
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
        sec_name = cand.evidence[0].section if cand.evidence else "DOCUMENTATION"
        cond = ExtractedClinicalCondition(
            condition_id=cand.diagnosis_id,
            original_mention=cand.raw_term,
            normalized_description=cand.normalized_diagnosis,
            evidence_text=primary_ev or cand.raw_term,
            evidence_location=EvidenceLocation(
                section=sec_name,
                start_char=0,
                end_char=len(cand.raw_term),
            ),
            status=ConditionStatus.HISTORICAL
            if cand.role == DiagnosisRole.HISTORICAL
            else ConditionStatus.ACTIVE,
            certainty=cand.certainty,
            temporality=cand.temporality,
            negation=NegationStatus.NEGATED
            if cand.certainty == Certainty.RULED_OUT
            else NegationStatus.AFFIRMATIVE,
            section=sec_name,
            treatment_evidence=primary_ev if cand.treatment_relevance > 0 else None,
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

    # Rule 2: Past medical history without inpatient monitoring/treatment
    norm_section = section_lower.replace("_", " ")
    is_pmh = (
        "past medical" in norm_section
        or "pmh" in norm_section
        or "past surgical" in norm_section
        or "psh" in norm_section
        or "history" in norm_section
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
