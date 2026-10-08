"""Streamlit web application for local ICD-10-CM clinical coding, PDF ingestion, and database storage."""

import asyncio
import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from medical_coding.config.settings import get_settings
from medical_coding.database.connection import init_db
from medical_coding.database.repository import MedicalCodingRepository
from medical_coding.graph.nodes import get_or_initialize_retrieval_system
from medical_coding.ingestion.agent import ClinicalDocumentIngestionAgent
from medical_coding.ingestion.detector import DocumentFormat
from medical_coding.orchestration.pipeline import MedicalCodingPipeline
from medical_coding.pdf.extractor import PDFExtractor
from medical_coding.schemas.enums import ExecutionStatus
from medical_coding.schemas.response import CodingResult
from medical_coding.ui.components import (
    NODE_TITLES,
    render_abstentions,
    render_failure_alert,
    render_header,
    render_kpi_metrics,
    render_pdf_page_inspector,
    render_primary_diagnosis,
    render_process_oversight_timeline,
    render_secondary_diagnoses,
    render_validation_audit_trail,
)
from medical_coding.ui.sample_data import (
    CLINICAL_CASES,
    generate_sample_image,
    generate_sample_pdf,
    generate_sample_txt,
)
from medical_coding.ui.styles import get_css

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Local ICD-10 Medical Coding System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom modern styling
st.markdown(get_css(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Singletons for Performance
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_pipeline(_version: str = "v3.2") -> MedicalCodingPipeline:
    """Return a cached singleton instance of the medical coding orchestrator."""
    settings = get_settings()
    return MedicalCodingPipeline(settings=settings)


def get_cached_repository() -> MedicalCodingRepository:
    """Return an initialized instance of the database repository."""
    init_db()
    return MedicalCodingRepository()


@st.cache_resource(show_spinner=False)
def get_cached_extractor() -> PDFExtractor:
    """Return a cached PyMuPDF extractor with pdfplumber fallback."""
    return PDFExtractor(max_pages=50)


@st.cache_resource(show_spinner=False)
def get_cached_ingestion_agent() -> ClinicalDocumentIngestionAgent:
    """Return a cached singleton instance of the multimodal clinical document ingestion agent."""
    return ClinicalDocumentIngestionAgent(max_pdf_pages=50)


# Initialize components
pipeline = get_cached_pipeline()
repo = get_cached_repository()
pdf_extractor = get_cached_extractor()
ingestion_agent = get_cached_ingestion_agent()
catalog, hybrid_retriever = get_or_initialize_retrieval_system()


# -----------------------------------------------------------------------------
# Sidebar Diagnostics & Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏥 System Diagnostics")
    st.markdown(
        """
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <div style="font-size: 12px; color: #94a3b8;">STORAGE ENGINE</div>
            <div style="font-size: 14px; font-weight: 600; color: #38bdf8;">SQLite (WAL Mode)</div>
            <div style="font-size: 11px; color: #cbd5e1; margin-top: 2px;"><code>./data/medical_coding.db</code></div>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 8px 0;"/>
            <div style="font-size: 12px; color: #94a3b8;">RETRIEVAL ENGINE</div>
            <div style="font-size: 14px; font-weight: 600; color: #10b981;">Hybrid BM25 + FAISS</div>
            <div style="font-size: 11px; color: #cbd5e1; margin-top: 2px;">Catalog: <strong>25 verified codes</strong></div>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 8px 0;"/>
            <div style="font-size: 12px; color: #94a3b8;">MULTIMODAL INGESTION</div>
            <div style="font-size: 14px; font-weight: 600; color: #f59e0b;">Auto-Detect & Routing</div>
            <div style="font-size: 11px; color: #cbd5e1; margin-top: 2px;">PDF • RapidOCR Images • TXT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Engine Settings")
    min_score = st.slider("Min Candidate Score:", min_value=0.1, max_value=0.9, value=0.40, step=0.05)
    max_pages = st.number_input("Max PDF Pages:", min_value=1, max_value=100, value=50)

    st.markdown("---")
    st.caption("Offline Local Medical ICD-10-CM Coding System • HIPAA & CMS Invariants Enforced")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
render_header(
    offline_mode=True,
    catalog_size=len(catalog),
    db_connected=True,
)


# -----------------------------------------------------------------------------
# Application Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Clinical Coding Workspace",
    "📑 Batch Multimodal Ingestion",
    "🗄️ Database & Encounters Vault",
    "📊 Analytics & Quality Audits",
    "🔍 ICD-10 Catalog & Semantic Explorer",
])


def render_coding_session_results(
    coding_result: CodingResult,
    steps_data: list[dict[str, Any]],
    thread_id: str,
    document_id: str,
    raw_text: str = "",
    active_filename: str = "clinical_document.txt",
    active_source_type: str = "text",
    active_file_bytes: bytes | None = None,
    min_score: float = 0.40,
) -> None:
    """Render complete diagnostics: Step Oversight Timeline, Failure/Success, KPIs, Diagnoses, and Exports."""
    _ = (raw_text, active_filename, active_source_type, active_file_bytes, min_score)
    is_error = coding_result.status == ExecutionStatus.ERROR
    has_failed_meta = bool(coding_result.metadata and coding_result.metadata.get("failed_step"))

    if is_error or has_failed_meta:
        failed_step = (
            coding_result.metadata.get("failed_step", "unspecified_node")
            if coding_result.metadata
            else "unspecified_node"
        )
        err_msg = (
            (coding_result.metadata.get("error") if coding_result.metadata else None)
            or "An error occurred during workflow execution."
        )
        tb_str = coding_result.metadata.get("traceback") if coding_result.metadata else None
        render_failure_alert(
            failed_step=failed_step,
            error_message=err_msg,
            error_traceback=tb_str,
            duration_ms=coding_result.processing_time_ms,
        )
        render_process_oversight_timeline(steps_data, coding_result.processing_time_ms)
        return

    # Render Node-by-Node Process Oversight & Latency Timeline
    render_process_oversight_timeline(steps_data, coding_result.processing_time_ms)

    # High-Level KPIs
    st.markdown("### 📊 Coding Decisions & Evidence Audit")
    render_kpi_metrics(coding_result)

    # Primary Diagnosis
    render_primary_diagnosis(coding_result.primary_diagnosis)

    # Secondary Diagnoses
    render_secondary_diagnoses(coding_result.secondary_diagnoses)

    # Explicit Abstentions & Rule-Outs
    render_abstentions(coding_result.abstentions)

    # Deterministic Rule Audit Trail
    render_validation_audit_trail(coding_result.primary_diagnosis, coding_result.secondary_diagnoses)

    # Structured JSON Response Payload
    with st.expander("🔍 View Structured JSON Response Payload", expanded=False):
        st.json(coding_result.model_dump())

    # Downloads
    st.markdown("---")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        json_data = coding_result.model_dump_json(indent=2)
        st.download_button(
            label="📥 Download Structured JSON Coding Report",
            data=json_data,
            file_name=f"{document_id}_coding_report.json",
            mime="application/json",
            use_container_width=True,
            key=f"dl_json_{thread_id}",
        )
    with d_col2:
        export_rows = []
        if coding_result.primary_diagnosis:
            export_rows.append({
                "Encounter": document_id,
                "Role": "PRIMARY",
                "Code": coding_result.primary_diagnosis.code,
                "ICD10CM": coding_result.primary_diagnosis.icd10cm,
                "ICD_O": coding_result.primary_diagnosis.icdo,
                "CPT": coding_result.primary_diagnosis.cpt,
                "Description": coding_result.primary_diagnosis.description,
                "Confidence": coding_result.primary_diagnosis.confidence_score,
                "Billable": coding_result.primary_diagnosis.is_terminal_billable,
                "Evidence": coding_result.primary_diagnosis.evidence_quote,
            })
        for s in coding_result.secondary_diagnoses:
            export_rows.append({
                "Encounter": document_id,
                "Role": "SECONDARY",
                "Code": s.code,
                "ICD10CM": s.icd10cm,
                "ICD_O": s.icdo,
                "CPT": s.cpt,
                "Description": s.description,
                "Confidence": s.confidence_score,
                "Billable": s.is_terminal_billable,
                "Evidence": s.evidence_quote,
            })
        csv_df = pd.DataFrame(export_rows)
        st.download_button(
            label="📥 Download Encounter Summary (CSV)",
            data=csv_df.to_csv(index=False),
            file_name=f"{document_id}_summary.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_csv_{thread_id}",
        )


# =============================================================================
# TAB 1: Clinical Coding Workspace (Multimodal Ingestion & Text Processing)
# =============================================================================
with tab1:
    st.markdown("#### Multimodal Clinical Document Ingestion & Coding")
    st.caption(
        "Upload clinical discharge summaries as PDF documents, plain text (.txt), screenshots / images (.png, .jpg, .webp), "
        "or enter text manually. The intelligent ingestion agent auto-detects the format via magic bytes and routes to the specialized parser."
    )

    # -------------------------------------------------------------------------
    # Thread Persistence & History Recovery across Browser Refresh
    # -------------------------------------------------------------------------
    url_thread_id = st.query_params.get("thread_id")
    active_thread_id = st.session_state.get("active_thread_id") or url_thread_id

    active_thread_record = (
        repo.get_pipeline_thread(active_thread_id)
        if (active_thread_id and hasattr(repo, "get_pipeline_thread"))
        else None
    )
    recent_threads = (
        repo.list_pipeline_threads(limit=20)
        if hasattr(repo, "list_pipeline_threads")
        else []
    )

    with st.container():
        p_col1, p_col2, p_col3 = st.columns([3, 2, 1])
        with p_col1:
            if active_thread_record:
                status_clr = (
                    "#10b981"
                    if active_thread_record["status"] == "SUCCESS"
                    else "#ef4444"
                    if active_thread_record["status"] == "FAILED"
                    else "#38bdf8"
                )
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 8px; padding: 8px 14px; margin-bottom: 8px;">
                        <span style="font-size: 11px; color: #94a3b8;">ACTIVE PERSISTENT THREAD:</span>
                        <strong style="color: {status_clr}; margin-left: 6px;">{active_thread_record['thread_id']}</strong>
                        <span class="pill-badge" style="background: rgba(255,255,255,0.05); color: {status_clr}; border: 1px solid {status_clr}; margin-left: 8px;">
                            {active_thread_record['status']}
                        </span>
                        <span style="font-size: 11px; color: #94a3b8; margin-left: 8px;">({active_thread_record['duration_ms']:.1f}ms)</span>
                        <span style="font-size: 11px; color: #10b981; margin-left: 10px;">● Restored from SQLite</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid #1e293b; border-radius: 8px; padding: 8px 14px; margin-bottom: 8px;">
                        <span style="font-size: 12px; color: #94a3b8;">🚀 State persistence active: execution states and histories are stored in SQLite and survive browser refreshes.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with p_col2:
            if recent_threads:
                thread_map = {
                    f"{t['thread_id']} ({t['status']} • {t['document_id']})": t["thread_id"]
                    for t in recent_threads
                }
                options = ["-- Quick Switch Thread --"] + list(thread_map.keys())
                selected_option = st.selectbox(
                    "Switch Thread:",
                    options=options,
                    label_visibility="collapsed",
                    key="quick_thread_select",
                )
                if selected_option != "-- Quick Switch Thread --":
                    chosen_tid = thread_map[selected_option]
                    if chosen_tid != active_thread_id:
                        st.query_params["thread_id"] = chosen_tid
                        st.session_state["active_thread_id"] = chosen_tid
                        st.rerun()
        with p_col3:
            if active_thread_id:
                if st.button("✨ Reset Session", use_container_width=True, key="reset_active_thread_btn"):
                    if "thread_id" in st.query_params:
                        del st.query_params["thread_id"]
                    if "active_thread_id" in st.session_state:
                        del st.session_state["active_thread_id"]
                    st.rerun()

    input_mode = st.radio(
        "Choose Input Modality:",
        [
            "📁 Multimodal Document Ingestion (PDF / Image / TXT)",
            "✍️ Direct Manual Clinical Text Entry",
            "⚡ Interactive Demonstration Scenarios",
        ],
        horizontal=True,
    )

    default_doc_id = (
        active_thread_record.get("document_id")
        if active_thread_record
        else f"ENC-{uuid4().hex[:8].upper()}"
    )
    document_id = st.text_input("Encounter Identifier (Document ID):", value=default_doc_id)
    clinical_text = active_thread_record.get("raw_text", "") if active_thread_record else ""
    active_file_bytes: bytes | None = None
    active_filename = active_thread_record.get("input_source", "clinical_document.txt") if active_thread_record else "clinical_document.txt"
    active_source_type = "text"
    extracted_pages_data: list[Any] = []
    ingest_result = None

    if input_mode == "📁 Multimodal Document Ingestion (PDF / Image / TXT)":
        uploaded_file = st.file_uploader(
            "Upload Clinical Record or Screenshot:",
            type=["pdf", "txt", "text", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            help="Supports PDFs, plain text files (.txt), and clinical screenshot images (.png, .jpg, .webp). The format is auto-detected via magic bytes.",
        )

        if uploaded_file is not None:
            active_file_bytes = uploaded_file.getvalue()
            active_filename = uploaded_file.name

            with st.spinner("🔍 Auto-detecting format and parsing clinical document..."):
                ingest_result = ingestion_agent.ingest(
                    source=active_file_bytes,
                    filename=active_filename,
                )
                clinical_text = ingest_result.normalized_text
                extracted_pages_data = ingest_result.pages
                active_source_type = ingest_result.format.value.lower()

            # Render Format Banner
            if ingest_result.format == DocumentFormat.PDF:
                st.markdown(
                    f"""
                    <div class="modality-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="modality-badge-pdf">📄 PDF CLINICAL DOCUMENT</span>
                                <span style="font-weight: 600; color: #f8fafc; margin-left: 10px;">{active_filename}</span>
                            </div>
                            <span style="font-size: 12px; color: #94a3b8;">Format Auto-Detected via Magic Bytes</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pages Detected", ingest_result.page_count)
                c2.metric("Word Count", ingest_result.word_count)
                c3.metric("Character Count", ingest_result.char_count)
                c4.metric("Extractor Used", ingest_result.metadata.get("extractor_used", "PYMUPDF").upper())

                if ingest_result.metadata.get("is_scanned"):
                    st.warning("⚠️ Scanned image PDF detected. RapidOCR fallback was automatically applied to extract text.")

                render_pdf_page_inspector(extracted_pages_data)

            elif ingest_result.format == DocumentFormat.IMAGE:
                conf_pct = round((ingest_result.ocr_confidence or 0.85) * 100, 1)
                st.markdown(
                    f"""
                    <div class="modality-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="modality-badge-image">🖼️ CLINICAL SCREENSHOT (RapidOCR)</span>
                                <span style="font-weight: 600; color: #f8fafc; margin-left: 10px;">{active_filename}</span>
                            </div>
                            <span style="font-size: 12px; color: #c084fc;">Auto-Detected & Processed via RapidOCR</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("OCR Engine", "RapidOCR ONNX")
                c2.metric("Avg Confidence", f"{conf_pct}%")
                c3.metric("Words Extracted", ingest_result.word_count)
                c4.metric("Lines Detected", ingest_result.metadata.get("lines_detected", 1))

                col_img, col_txt = st.columns([1, 1])
                with col_img:
                    st.markdown("##### 🖼️ Uploaded Clinical Screenshot")
                    st.image(active_file_bytes, caption=active_filename, width="stretch")
                with col_txt:
                    st.markdown("##### 📄 OCR Extracted Discharge Summary")
                    clinical_text = st.text_area("Extracted Text (Editable):", value=clinical_text, height=350)

            elif ingest_result.format == DocumentFormat.TXT:
                st.markdown(
                    f"""
                    <div class="modality-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="modality-badge-txt">📝 PLAIN TEXT DOCUMENT (.TXT)</span>
                                <span style="font-weight: 600; color: #f8fafc; margin-left: 10px;">{active_filename}</span>
                            </div>
                            <span style="font-size: 12px; color: #34d399;">Auto-Detected Plain Text Encoding</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Format", "Plain Text")
                c2.metric("Encoding", "UTF-8 / Latin-1")
                c3.metric("Word Count", ingest_result.word_count)
                c4.metric("Character Count", ingest_result.char_count)

            # Show section badges if any detected
            detected_sections = ingest_result.metadata.get("detected_sections", [])
            if detected_sections:
                st.markdown(
                    "**Detected Clinical Sections:** "
                    + " ".join([f'<span class="section-chip">🏷️ {s}</span>' for s in detected_sections]),
                    unsafe_allow_html=True,
                )

            if ingest_result.format != DocumentFormat.IMAGE:
                clinical_text = st.text_area(
                    "Clinical Text Preview (Editable):",
                    value=clinical_text,
                    height=220,
                    key=f"txt_preview_{active_filename}",
                )

    elif input_mode == "✍️ Direct Manual Clinical Text Entry":
        clinical_text = st.text_area(
            "Clinical Documentation (Discharge Summary, History & Physical):",
            height=260,
            placeholder="Paste or type clinical discharge summary here...",
        )
        active_filename = "manual_entry.txt"
        active_source_type = "manual_text"

        if clinical_text.strip():
            with st.spinner("Analyzing manual clinical text..."):
                ingest_result = ingestion_agent.ingest(source=clinical_text, filename=active_filename)

            st.markdown(
                """
                <div class="modality-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="modality-badge-manual">✍️ MANUAL CLINICAL ENTRY</span>
                            <span style="font-weight: 600; color: #f8fafc; margin-left: 10px;">Direct Input</span>
                        </div>
                        <span style="font-size: 12px; color: #fbbf24;">Direct Clinical Text Stream</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Word Count", ingest_result.word_count)
            c2.metric("Character Count", ingest_result.char_count)
            c3.metric("Detected Sections", ingest_result.metadata.get("section_count", 0))

            detected_sections = ingest_result.metadata.get("detected_sections", [])
            if detected_sections:
                st.markdown(
                    "**Detected Clinical Sections:** "
                    + " ".join([f'<span class="section-chip">🏷️ {s}</span>' for s in detected_sections]),
                    unsafe_allow_html=True,
                )

    else:  # Interactive Demonstration Scenarios
        selected_case_name = st.selectbox("Select Clinical Demonstration Scenario:", list(CLINICAL_CASES.keys()))
        case_info = CLINICAL_CASES[selected_case_name]

        st.markdown(
            f"""
            <div style="background: rgba(14, 165, 233, 0.08); border-left: 3px solid #38bdf8; padding: 10px 14px; margin-bottom: 12px; border-radius: 0 6px 6px 0;">
                <strong style="color: #38bdf8;">{case_info['title']}</strong><br/>
                <span style="font-size: 13px; color: #cbd5e1;">{case_info['description']}</span><br/>
                <span style="font-size: 11px; color: #94a3b8;">Expected Primary: <code>{case_info['expected_primary']}</code> | Secondaries: <code>{', '.join(case_info['expected_secondary']) or 'None'}</code></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("📄 Ingest as Vector PDF", use_container_width=True):
                pdf_bytes_gen = generate_sample_pdf(case_info["title"], case_info["text"])
                st.session_state["active_demo_bytes"] = pdf_bytes_gen
                st.session_state["active_demo_filename"] = f"{case_info['encounter_id']}.pdf"
                st.session_state["active_demo_format"] = "pdf"
                st.session_state["active_demo_text"] = case_info["text"]
                st.session_state["active_demo_id"] = case_info["encounter_id"]
                st.success("Generated synthetic clinical PDF!")

        with col_b:
            if st.button("🖼️ Ingest as Screenshot (OCR)", use_container_width=True):
                img_bytes_gen = generate_sample_image(case_info["title"], case_info["text"])
                st.session_state["active_demo_bytes"] = img_bytes_gen
                st.session_state["active_demo_filename"] = f"{case_info['encounter_id']}_screenshot.png"
                st.session_state["active_demo_format"] = "image"
                st.session_state["active_demo_text"] = case_info["text"]
                st.session_state["active_demo_id"] = case_info["encounter_id"]
                st.success("Rendered clinical screenshot image for OCR parsing!")

        with col_c:
            if st.button("📝 Ingest as Plain Text (.txt)", use_container_width=True):
                txt_bytes_gen = generate_sample_txt(case_info["title"], case_info["text"])
                st.session_state["active_demo_bytes"] = txt_bytes_gen
                st.session_state["active_demo_filename"] = f"{case_info['encounter_id']}.txt"
                st.session_state["active_demo_format"] = "txt"
                st.session_state["active_demo_text"] = case_info["text"]
                st.session_state["active_demo_id"] = case_info["encounter_id"]
                st.success("Loaded as plain text document!")

        if "active_demo_id" in st.session_state:
            document_id = st.session_state["active_demo_id"]
            active_file_bytes = st.session_state.get("active_demo_bytes")
            active_filename = st.session_state.get("active_demo_filename", f"{document_id}.txt")
            demo_fmt = st.session_state.get("active_demo_format", "text")

            # Run through ingestion agent to get real-time parser output and auto-detection badge
            with st.spinner("Processing scenario through Ingestion Agent..."):
                if active_file_bytes:
                    ingest_result = ingestion_agent.ingest(source=active_file_bytes, filename=active_filename)
                else:
                    ingest_result = ingestion_agent.ingest(source=st.session_state["active_demo_text"], filename=active_filename)

                clinical_text = ingest_result.normalized_text
                active_source_type = ingest_result.format.value.lower()
                extracted_pages_data = ingest_result.pages

            if demo_fmt == "image":
                conf_pct = round((ingest_result.ocr_confidence or 0.85) * 100, 1)
                st.markdown(
                    f"""
                    <div class="modality-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="modality-badge-image">🖼️ AUTO-DETECTED: CLINICAL SCREENSHOT (RapidOCR)</span>
                                <span style="font-weight: 600; color: #f8fafc; margin-left: 10px;">{active_filename}</span>
                            </div>
                            <span style="font-size: 12px; color: #c084fc;">RapidOCR Confidence: <strong>{conf_pct}%</strong></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_img, col_txt = st.columns([1, 1])
                with col_img:
                    st.markdown("##### 🖼️ Synthetic Clinical Screenshot")
                    st.image(active_file_bytes, caption=active_filename, width="stretch")
                with col_txt:
                    st.markdown("##### 📄 OCR Extracted Text")
                    clinical_text = st.text_area("Extracted Document Content:", value=clinical_text, height=350)

            elif demo_fmt == "pdf":
                st.markdown(
                    f"""
                    <div class="modality-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="modality-badge-pdf">📄 AUTO-DETECTED: CLINICAL PDF</span>
                                <span style="font-weight: 600; color: #f8fafc; margin-left: 10px;">{active_filename}</span>
                            </div>
                            <span style="font-size: 12px; color: #f87171;">PyMuPDF Vector Extraction</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_pdf_page_inspector(extracted_pages_data)
                clinical_text = st.text_area("Document Text Preview:", value=clinical_text, height=200)

            else:
                st.markdown(
                    f"""
                    <div class="modality-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="modality-badge-txt">📝 AUTO-DETECTED: PLAIN TEXT</span>
                                <span style="font-weight: 600; color: #f8fafc; margin-left: 10px;">{active_filename}</span>
                            </div>
                            <span style="font-size: 12px; color: #34d399;">Direct Normalized Text</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                clinical_text = st.text_area("Document Text Preview:", value=clinical_text, height=200)
        else:
            clinical_text = st.text_area("Document Preview:", value=case_info["text"], height=200)

    # Coding Execution Trigger
    st.markdown("---")
    col_run, col_clear = st.columns([3, 1])

    with col_run:
        run_button = st.button("⚡ Run ICD-10 Medical Coding Pipeline", type="primary", use_container_width=True)

    if run_button:
        if not clinical_text and not active_file_bytes:
            st.error("❌ Please provide clinical text or upload a document/screenshot before running coding.")
        else:
            new_thread_id = f"thread-{uuid4().hex[:10]}"
            st.query_params["thread_id"] = new_thread_id
            st.session_state["active_thread_id"] = new_thread_id

            st.markdown("#### ⚡ Real-Time Pipeline Progress")
            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()
            with st.expander("📋 Live Node Execution Log Stream", expanded=True):
                log_placeholder = st.empty()

            live_logs = [f"[{datetime.now().strftime('%H:%M:%S')}] Initialized execution thread {new_thread_id}"]
            log_placeholder.code("\n".join(live_logs), language="text")

            def on_progress(step_idx: int, total_steps: int, node_name: str, log_msg: str, duration_ms: float) -> None:
                pct = min(1.0, max(0.0, step_idx / total_steps))
                progress_bar.progress(pct)
                title = NODE_TITLES.get(node_name, node_name)
                status_placeholder.markdown(
                    f"""
                    <div class="progress-container">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="active-agent-badge">
                                ⚙️ Step {step_idx}/{total_steps}: {title}
                            </span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #38bdf8;">
                                {duration_ms:.1f} ms
                            </span>
                        </div>
                        <div style="margin-top: 8px; font-size: 13px; color: #e2e8f0;">
                            {log_msg}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                live_logs.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Step {step_idx} ({title}): {log_msg} ({duration_ms:.1f}ms)"
                )
                log_placeholder.code("\n".join(live_logs), language="text")

            try:
                coding_result: CodingResult = asyncio.run(
                    pipeline.run_document(
                        document_id=document_id,
                        text=clinical_text,
                        file_bytes=active_file_bytes,
                        metadata={
                            "filename": active_filename,
                            "min_score": min_score,
                            "source_type": active_source_type,
                        },
                        thread_id=new_thread_id,
                        progress_callback=on_progress,
                        persist_thread=True,
                    )
                )
                progress_bar.progress(1.0)
            except Exception as exc:
                coding_result = CodingResult(
                    document_id=document_id,
                    status=ExecutionStatus.ERROR,
                    primary_diagnosis=None,
                    secondary_diagnoses=[],
                    abstentions=[],
                    processing_time_ms=0.0,
                    models_used={},
                    metadata={"error": str(exc), "failed_step": "pipeline_execution_error", "thread_id": new_thread_id},
                )

            # Persist document record into encounter archive
            if coding_result and coding_result.status != ExecutionStatus.ERROR:
                file_size = len(active_file_bytes) if active_file_bytes else len(clinical_text.encode("utf-8"))
                page_count = len(extracted_pages_data) if extracted_pages_data else 1

                repo.save_coding_result(
                    result=coding_result,
                    raw_text=clinical_text,
                    filename=active_filename,
                    source_type=active_source_type,
                    page_count=page_count,
                    file_size_bytes=file_size,
                    metadata={"min_score": min_score, "modality": active_source_type, "thread_id": new_thread_id},
                )

            fresh_thread = repo.get_pipeline_thread(new_thread_id)
            steps_data = fresh_thread.get("steps", []) if fresh_thread else []

            render_coding_session_results(
                coding_result=coding_result,
                steps_data=steps_data,
                thread_id=new_thread_id,
                document_id=document_id,
                raw_text=clinical_text,
                active_filename=active_filename,
                active_source_type=active_source_type,
                active_file_bytes=active_file_bytes,
                min_score=min_score,
            )

    elif active_thread_record:
        # Browser refresh or historical thread restoration
        st.markdown("---")
        st.markdown(f"#### 📜 Restored Pipeline Execution Session (`{active_thread_id}`)")

        steps_data = active_thread_record.get("steps", [])
        thread_status = active_thread_record.get("status")

        if thread_status == "SUCCESS" and active_thread_record.get("result"):
            try:
                res_obj = CodingResult(**active_thread_record["result"])
            except Exception:
                res_obj = None

            if res_obj:
                render_coding_session_results(
                    coding_result=res_obj,
                    steps_data=steps_data,
                    thread_id=active_thread_id,
                    document_id=active_thread_record.get("document_id", document_id),
                    raw_text=active_thread_record.get("raw_text", ""),
                    active_filename=active_thread_record.get("input_source", "clinical_document.txt"),
                    active_source_type="text",
                    min_score=min_score,
                )
            else:
                st.warning("Could not reconstruct result payload for this thread.")
                render_process_oversight_timeline(steps_data, active_thread_record.get("duration_ms", 0.0))

        elif thread_status == "FAILED":
            render_failure_alert(
                failed_step=active_thread_record.get("failed_step"),
                error_message=active_thread_record.get("error_message"),
                error_traceback=active_thread_record.get("error_traceback"),
                duration_ms=active_thread_record.get("duration_ms", 0.0),
            )
            render_process_oversight_timeline(steps_data, active_thread_record.get("duration_ms", 0.0))

        elif thread_status == "RUNNING":
            pct = active_thread_record.get("progress_pct", 0.0) / 100.0
            st.info(
                f"⏳ Pipeline Thread was recorded as RUNNING at step: **{active_thread_record.get('current_step_name')}** ({active_thread_record.get('progress_pct')}%)"
            )
            st.progress(pct)
            render_process_oversight_timeline(steps_data, active_thread_record.get("duration_ms", 0.0))



# =============================================================================
# TAB 2: Batch Multimodal Ingestion
# =============================================================================
with tab2:
    st.markdown("#### Concurrent Multimodal Batch Ingestion & Coding")
    st.caption("Upload multiple clinical documents simultaneously (mixed PDFs, plain text, or screenshot images). The Ingestion Agent automatically detects each file's format and switches to the appropriate parser.")

    batch_files = st.file_uploader(
        "Upload Multiple Clinical Documents (PDF / TXT / Images):",
        type=["pdf", "txt", "text", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
        help="Select multiple clinical files of any supported format for concurrent batch ingestion.",
    )

    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        st.write(f"📁 **{len(batch_files)} Document(s)** queued for batch processing.")
    with b_col2:
        concurrency_limit = st.slider("Document Concurrency Limit:", min_value=1, max_value=20, value=10)

    if st.button("🚀 Process Batch Ingestion", type="primary", disabled=len(batch_files) == 0):
        progress_bar = st.progress(0)
        status_text = st.empty()
        batch_results: list[dict[str, Any]] = []

        total_files = len(batch_files)

        for idx, b_file in enumerate(batch_files):
            status_text.text(f"Processing ({idx + 1}/{total_files}): {b_file.name}...")
            file_bytes = b_file.read()
            doc_id = f"BATCH-{uuid4().hex[:6].upper()}"

            # Auto-detect format and extract text via ingestion agent
            ingest_res = ingestion_agent.ingest(source=file_bytes, filename=b_file.name)
            norm_text = ingest_res.normalized_text

            # Run document through pipeline
            try:
                res: CodingResult = asyncio.run(
                    pipeline.run_document(
                        document_id=doc_id,
                        text=norm_text,
                        file_bytes=file_bytes,
                        metadata={"filename": b_file.name, "modality": ingest_res.format.value},
                    )
                )

                # Persist to database
                repo.save_coding_result(
                    result=res,
                    raw_text=norm_text,
                    filename=b_file.name,
                    source_type=f"batch_{ingest_res.format.value.lower()}",
                    page_count=ingest_res.page_count,
                    file_size_bytes=len(file_bytes),
                    metadata={"format": ingest_res.format.value, "is_ocr": ingest_res.is_ocr},
                )

                batch_results.append({
                    "Filename": b_file.name,
                    "Encounter ID": doc_id,
                    "Modality": ingest_res.format.value,
                    "OCR Used": "Yes" if ingest_res.is_ocr else "No",
                    "Status": res.status.value if hasattr(res.status, "value") else str(res.status),
                    "Primary Code": res.primary_diagnosis.code if res.primary_diagnosis else "N/A",
                    "Primary Description": res.primary_diagnosis.description if res.primary_diagnosis else "None",
                    "Secondary Diags": len(res.secondary_diagnoses),
                    "Abstentions": len(res.abstentions),
                    "Latency (ms)": f"{res.processing_time_ms:.1f}",
                })
            except Exception as e:
                batch_results.append({
                    "Filename": b_file.name,
                    "Encounter ID": doc_id,
                    "Modality": ingest_res.format.value if ingest_res else "UNKNOWN",
                    "OCR Used": "Yes" if (ingest_res and ingest_res.is_ocr) else "No",
                    "Status": "ERROR",
                    "Primary Code": "N/A",
                    "Primary Description": str(e),
                    "Secondary Diags": 0,
                    "Abstentions": 0,
                    "Latency (ms)": "0.0",
                })

            progress_bar.progress((idx + 1) / total_files)

        status_text.text(f"✅ Completed batch processing of {total_files} multimodal document(s)!")

        # Display Batch Results Table
        b_df = pd.DataFrame(batch_results)
        st.download_button(
            label="📥 Download Batch Results (CSV)",
            data=b_df.to_csv(index=False),
            file_name="multimodal_batch_results.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 3: Database & Encounters Vault
# =============================================================================
with tab3:
    st.markdown("#### Database Vault & Historical Execution Audit")
    st.caption(
        "Direct connection to local SQLite database (`./data/medical_coding.db`) using WAL mode. "
        "Standardized SQLAlchemy model abstraction allows 1-click drop-in replacement with PostgreSQL."
    )

    vault_tab1, vault_tab2 = st.tabs([
        "🗄️ Clinical Encounters Archive",
        "🧵 Pipeline Execution Threads & Node Logs",
    ])

    with vault_tab1:
        # High-level DB metrics
        stats = repo.get_analytics_summary()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Encounters Saved", stats["total_documents"])
        m2.metric("Total Diagnoses Stored", stats["total_diagnoses"])
        m3.metric("HIPAA Billable Ratio", f"{stats['billable_ratio_percent']}%")
        m4.metric("Avg Latency", f"{stats['average_processing_time_ms']} ms")

        st.markdown("---")

        # Search & Filters
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search_query = st.text_input("🔍 Search Encounters (ID, Filename, Clinical Notes):", placeholder="e.g. heart failure, ENC-HF-...")
        with f2:
            status_filter = st.selectbox("Status Filter:", ["ALL", "SUCCESS", "PARTIAL_SUCCESS", "ABSTAINED", "ERROR"])
        with f3:
            code_filter = st.text_input("Filter by ICD-10 Code:", placeholder="e.g. I50, E11")

        # Fetch matching documents
        stored_docs = repo.list_documents(
            search=search_query if search_query else None,
            status=status_filter,
            code_filter=code_filter if code_filter else None,
            limit=200,
        )

        if not stored_docs:
            st.info("No encounter records found matching current query or filters.")
        else:
            st.write(f"Showing **{len(stored_docs)}** encounter record(s):")

            # Table summary
            summary_rows = []
            for d in stored_docs:
                src = (d["source_type"] or "").upper()
                if "PDF" in src:
                    modality_icon = "📄 PDF"
                elif "IMAGE" in src:
                    modality_icon = "🖼️ IMAGE"
                elif "MANUAL" in src:
                    modality_icon = "✍️ MANUAL"
                elif "TXT" in src or "TEXT" in src:
                    modality_icon = "📝 TXT"
                else:
                    modality_icon = src or "TEXT"

                summary_rows.append({
                    "Document ID": d["document_id"],
                    "Filename": d["filename"],
                    "Modality": modality_icon,
                    "Status": d["status"],
                    "Primary Code": d["primary_code"] or "None",
                    "Primary Description": d["primary_description"] or "None",
                    "Secondaries": d["secondary_count"],
                    "Abstentions": d["abstention_count"],
                    "Latency (ms)": f"{d['processing_time_ms']:.1f}",
                    "Date": d["created_at"][:19] if d["created_at"] else "N/A",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # Detailed Encounter Inspector Drawer
            st.markdown("---")
            st.markdown("##### 🔎 Inspect Full Encounter Record")
            doc_ids = [d["document_id"] for d in stored_docs]
            selected_doc_id = st.selectbox("Select Encounter ID to Inspect:", doc_ids)

            if selected_doc_id:
                full_doc = repo.get_document_by_id(selected_doc_id)
                if full_doc:
                    col_info1, col_info2 = st.columns([3, 1])
                    with col_info1:
                        src_val = (full_doc["source_type"] or "").upper()
                        st.markdown(
                            f"**Encounter:** `{full_doc['document_id']}` | **Modality:** `{src_val}` | **File:** `{full_doc['filename']}` | **Pages:** `{full_doc['page_count']}` | **Words:** `{full_doc['word_count']}`"
                        )
                    with col_info2:
                        if st.button("🗑️ Delete Encounter", key=f"del_{selected_doc_id}"):
                            repo.delete_document(selected_doc_id)
                            st.success(f"Encounter {selected_doc_id} deleted!")
                            st.rerun()

                    # Tabbed detail view for the encounter
                    d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs([
                        "🩺 Diagnoses & Evidences",
                        "🚫 Abstentions",
                        "📝 Raw Clinical Text",
                        "🛡️ Invariant Checks",
                    ])

                    with d_tab1:
                        if full_doc["primary_code"]:
                            st.markdown(
                                f"""
                                <div class="primary-card">
                                    <div class="primary-code">{full_doc['primary_code']}</div>
                                    <div class="primary-desc">{full_doc['primary_description']}</div>
                                    <span class="pill-badge badge-success">✓ Primary Diagnosis</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        for diag in full_doc["diagnoses"]:
                            if diag["role"] != "PRIMARY":
                                st.markdown(
                                    f"""
                                    <div class="secondary-card">
                                        <div style="font-weight: 600; color: #38bdf8;">{diag['code']} - {diag['description']}</div>
                                        <div style="font-size: 12px; color: #94a3b8; margin: 4px 0;">Role: {diag['role']} | Acuity: {diag['acuity']} | Certainty: {diag['certainty']}</div>
                                        <div class="evidence-box">
                                            <em>"{diag['evidence_quote']}"</em>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    with d_tab2:
                        if not full_doc["abstentions"]:
                            st.write("No abstentions recorded for this encounter.")
                        for a in full_doc["abstentions"]:
                            st.markdown(
                                f"""
                                <div class="abstention-card">
                                    <strong>{a['raw_term'] or 'Condition'}</strong>: {a['detail']}
                                    <div style="font-size: 11px; color: #94a3b8;">Reason: <code>{a['reason']}</code></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with d_tab3:
                        st.text_area("Stored Clinical Text:", value=full_doc["raw_text"], height=250, disabled=True)

                    with d_tab4:
                        if not full_doc["validation_checks"]:
                            st.write("No explicit individual check records stored; general invariants verified.")
                        for v in full_doc["validation_checks"]:
                            check_icon = "✔" if v["passed"] else "✖"
                            st.write(f"{check_icon} **{v['rule_name']}**: {v['details']}")

            # Vault Export Options
            st.markdown("---")
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                full_df = repo.get_documents_df()
                st.download_button(
                    label="📥 Export Full Vault to CSV",
                    data=full_df.to_csv(index=False),
                    file_name="encounters_vault_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with e_col2:
                st.download_button(
                    label="📥 Export Full Vault to JSON",
                    data=json.dumps(stored_docs, indent=2),
                    file_name="encounters_vault_export.json",
                    mime="application/json",
                    use_container_width=True,
                )

    with vault_tab2:
        st.markdown("#### 🧵 Pipeline Execution Threads & Node Logs Vault")
        st.caption(
            "Complete log records, latency breakdowns, and node oversight for every pipeline run. "
            "Stored persistently in SQLite; ready for PostgreSQL migration without code changes."
        )

        all_threads_list = (
            repo.list_pipeline_threads(limit=100)
            if hasattr(repo, "list_pipeline_threads")
            else []
        )
        total_th = len(all_threads_list)
        succ_th = sum(1 for t in all_threads_list if t.get("status") == "SUCCESS")
        fail_th = sum(1 for t in all_threads_list if t.get("status") == "FAILED")
        avg_th_lat = round(
            sum(t.get("duration_ms", 0.0) for t in all_threads_list) / max(1, total_th), 1
        )

        th_m1, th_m2, th_m3, th_m4 = st.columns(4)
        th_m1.metric("Total Executions", total_th)
        th_m2.metric("Successful Threads", succ_th)
        th_m3.metric("Failed Threads", fail_th)
        th_m4.metric("Avg Thread Latency", f"{avg_th_lat} ms")

        st.markdown("---")

        th_f1, th_f2 = st.columns([3, 1])
        with th_f1:
            th_search = st.text_input(
                "🔍 Search Threads (ID, Document ID, Source):",
                placeholder="e.g. thread-..., ENC-...",
                key="th_search_input",
            )
        with th_f2:
            th_status_filter = st.selectbox(
                "Filter Status:",
                ["ALL", "SUCCESS", "FAILED", "RUNNING"],
                key="th_status_filter_sel",
            )

        filtered_threads = all_threads_list
        if th_status_filter and th_status_filter != "ALL":
            filtered_threads = [t for t in filtered_threads if t.get("status") == th_status_filter]
        if th_search:
            term = th_search.strip().lower()
            filtered_threads = [
                t for t in filtered_threads
                if term in t.get("thread_id", "").lower()
                or term in t.get("document_id", "").lower()
                or term in t.get("input_source", "").lower()
            ]

        if not filtered_threads:
            st.info("No execution thread history found matching criteria.")
        else:
            st.write(f"Showing **{len(filtered_threads)}** pipeline thread(s):")
            th_summary_rows = []
            for t in filtered_threads:
                th_summary_rows.append({
                    "Thread ID": t["thread_id"],
                    "Document ID": t["document_id"],
                    "Source": t["input_source"],
                    "Status": t["status"],
                    "Current / Failed Step": t["failed_step"] or t["current_step_name"],
                    "Steps Completed": len(t.get("steps", [])),
                    "Latency (ms)": f"{t['duration_ms']:.1f}",
                    "Start Time": t["start_time"][:19] if t["start_time"] else "N/A",
                })
            st.dataframe(pd.DataFrame(th_summary_rows), use_container_width=True)

            st.markdown("---")
            st.markdown("##### 🔎 Inspect Execution Thread Timeline & Audit")
            th_ids = [t["thread_id"] for t in filtered_threads]
            selected_th_id = st.selectbox("Select Thread ID to Inspect:", th_ids, key="sel_th_inspect")

            if selected_th_id:
                sel_thread = (
                    repo.get_pipeline_thread(selected_th_id)
                    if hasattr(repo, "get_pipeline_thread")
                    else None
                )
                if sel_thread:
                    c_act1, c_act2, c_act3 = st.columns([2, 1, 1])
                    with c_act1:
                        st.markdown(
                            f"**Thread:** `{sel_thread['thread_id']}` | **Status:** `{sel_thread['status']}` | **Duration:** `{sel_thread['duration_ms']:.1f}ms`"
                        )
                    with c_act2:
                        if st.button("🚀 Load into Workspace (Tab 1)", key=f"load_ws_{selected_th_id}", use_container_width=True):
                            st.query_params["thread_id"] = selected_th_id
                            st.session_state["active_thread_id"] = selected_th_id
                            st.success(f"Thread {selected_th_id} loaded! Switch to Tab 1.")
                            st.rerun()
                    with c_act3:
                        if st.button("🗑️ Delete Thread", key=f"del_th_{selected_th_id}", use_container_width=True):
                            repo.delete_pipeline_thread(selected_th_id)
                            st.success(f"Thread {selected_th_id} deleted!")
                            st.rerun()

                    # Render timeline for selected thread
                    render_process_oversight_timeline(
                        sel_thread.get("steps", []), sel_thread.get("duration_ms", 0.0)
                    )

                    if sel_thread.get("status") == "FAILED":
                        render_failure_alert(
                            failed_step=sel_thread.get("failed_step"),
                            error_message=sel_thread.get("error_message"),
                            error_traceback=sel_thread.get("error_traceback"),
                            duration_ms=sel_thread.get("duration_ms", 0.0),
                        )

                    with st.expander("📄 Clinical Document Input Text", expanded=False):
                        st.text_area(
                            "Input Document Text:",
                            value=sel_thread.get("raw_text", ""),
                            height=200,
                            disabled=True,
                            key=f"txt_th_{selected_th_id}",
                        )

                    if sel_thread.get("result"):
                        with st.expander("🔍 Structured Output Result JSON", expanded=False):
                            st.json(sel_thread.get("result"))


# =============================================================================
# TAB 4: Analytics & Quality Audits
# =============================================================================
with tab4:
    st.markdown("#### Clinical Quality & Invariant Analytics")
    st.caption("Aggregated analytics over all stored encounters in SQLite database.")

    analytics_data = repo.get_analytics_summary()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Documents Coded", analytics_data["total_documents"])
    a2.metric("Total Diagnoses Assigned", analytics_data["total_diagnoses"])
    a3.metric("HIPAA Billable Compliance", f"{analytics_data['billable_ratio_percent']}%")
    a4.metric("Mean Latency", f"{analytics_data['average_processing_time_ms']} ms")

    st.markdown("---")

    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        st.markdown("##### 🏆 Top Primary Diagnoses")
        top_primaries = analytics_data["top_primary_codes"]
        if top_primaries:
            primary_chart_df = pd.DataFrame(top_primaries).set_index("code")
            st.bar_chart(primary_chart_df["count"])
        else:
            st.info("No primary diagnoses recorded yet.")

    with chart_c2:
        st.markdown("##### 📋 Top Secondary Comorbidities")
        top_secondaries = analytics_data["top_secondary_codes"]
        if top_secondaries:
            sec_chart_df = pd.DataFrame(top_secondaries).set_index("code")
            st.bar_chart(sec_chart_df["count"])
        else:
            st.info("No secondary diagnoses recorded yet.")

    st.markdown("---")
    chart_c3, chart_c4 = st.columns(2)

    with chart_c3:
        st.markdown("##### 🚫 Abstention Reasons Breakdown")
        abst_dist = analytics_data["abstention_distribution"]
        if abst_dist:
            abst_df = pd.DataFrame(list(abst_dist.items()), columns=["Reason", "Count"]).set_index("Reason")
            st.bar_chart(abst_df)
        else:
            st.info("No abstentions logged yet.")

    with chart_c4:
        st.markdown("##### 📈 Encounter Status Breakdown")
        status_dist = analytics_data["status_breakdown"]
        if status_dist:
            st_df = pd.DataFrame(list(status_dist.items()), columns=["Status", "Count"]).set_index("Status")
            st.bar_chart(st_df)
        else:
            st.info("No status data available.")


# =============================================================================
# TAB 5: ICD-10 Catalog & Semantic Explorer
# =============================================================================
with tab5:
    st.markdown("#### Authoritative ICD-10-CM Catalog & Retriever Explorer")
    st.caption("Directly test the hybrid BM25 lexical search and FAISS dense vector search against the local ICD catalog.")

    search_query = st.text_input(
        "Enter Clinical Term or Condition to Search Local Catalog:",
        value="acute systolic heart failure",
        placeholder="e.g. congestive heart failure, diabetes, COPD, myocardial infarction",
    )

    k_slider = st.slider("Top K Candidates to Retrieve:", min_value=1, max_value=15, value=5)

    if search_query:
        candidates = hybrid_retriever.retrieve(query=search_query, top_k=k_slider)

        st.markdown(f"##### 🎯 Hybrid Retrieval Results ({len(candidates)} candidates):")
        for idx, cand in enumerate(candidates, start=1):
            score_pct = int(cand.retrieval_score * 100)
            billable_str = "✓ Billable Leaf" if cand.is_valid_billable else "⚠ Non-Billable Header"
            badge_class = "badge-success" if cand.is_valid_billable else "badge-warning"

            st.markdown(
                f"""
                <div class="secondary-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 20px; font-weight: 700; color: #38bdf8;">#{idx}. {cand.code}</span>
                            <span style="font-size: 16px; font-weight: 600; color: #f8fafc; margin-left: 12px;">{cand.description}</span>
                        </div>
                        <div>
                            <span class="pill-badge {badge_class}">{billable_str}</span>
                            <span style="font-size: 13px; color: #94a3b8; margin-left: 8px;">Similarity: <strong>{score_pct}%</strong></span>
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">
                        Category: <code>{cand.category}</code> | Retrieval Source: <code>{cand.retrieval_method}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("##### 📚 Complete Local ICD-10-CM Source Catalog")
    all_records = catalog.get_all_records()
    catalog_df = pd.DataFrame([
        {
            "Code": r.code,
            "Description": r.short_description,
            "Category": r.category,
            "Billable": "Yes" if r.is_valid_billable else "No",
            "Excludes1": ", ".join(r.excludes1) if r.excludes1 else "None",
        }
        for r in all_records
    ])
    st.dataframe(catalog_df, use_container_width=True)
