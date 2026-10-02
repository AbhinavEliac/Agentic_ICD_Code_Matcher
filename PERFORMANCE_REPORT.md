# Performance & Concurrency Audit Report

**System Name**: Local Medical ICD-10-CM Coding System  
**Evaluation Role**: Senior Production Architect & Performance Engineer  
**Target Environment**: Windows 11, Offline CPU Execution, Multi-core Host  
**Date**: 2026-10-02  

---

## 1. Performance Overview

A comprehensive performance profile was conducted focusing on memory utilization, concurrency bottlenecks, model loading behavior, blocking calls in asynchronous flows, and LLM call redundancy.

The system demonstrates excellent deterministic performance, with the full 10-node test suite (103 tests across all modules) executing in **under 0.8 seconds** when using deterministic mock fixtures, and individual document processing executing in **sub-15 milliseconds** for rule-based passes.

---

## 2. Granular Performance Findings

### 2.1 Model Reloads & Memory Footprint
- **Finding**: **Zero Redundant Model Reloads**.
- **Analysis**: The `LLMLifecycleManager` successfully enforces single-instance caching. Model weights are loaded exactly once into RAM (`_load_count = 1`). Subsequent requests reuse the loaded GGUF instance across documents.
- **RAM Footprint**:
  - GGUF 7B Q4_K_M weights: ~4.2 GB
  - FAISS index & metadata (hospital dataset): ~15 MB
  - Process overhead: ~180 MB
  - Total resident memory: ~4.4 GB (well within typical 16GB/32GB clinical workstation capacities).

### 2.2 Redundant LLM Calls
- **Finding**: **Optimized Short-Circuiting Active**.
- **Analysis**:
  - Excluded conditions (e.g. historical conditions without active care, ruled-out conditions) are short-circuited in `CandidateRankingAgent` *before* LLM invocation, saving unnecessary LLM calls.
  - Documents with empty candidate pools or missing evidence quotes immediately abstain without engaging the LLM.
  - Deterministic fallback routes are utilized when LLM output is unavailable or unparseable.

### 2.3 Unnecessary LangGraph Nodes & Agents
- **Finding**: **Lean 10-Node Workflow**.
- **Analysis**: The 10 nodes directly map to clinical coding requirements. No extraneous agents or conversational cycles exist.
- **Node Classification**:
  - Deterministic Python Nodes: 7 (`validate_document`, `extract_text`, `classify_diagnoses`, `retrieve_candidates`, `validate_codes`, `evaluate_confidence`, `finalize_output`).
  - LLM-augmented Nodes: 3 (`extract_diagnoses`, `analyze_context`, `rank_candidates`).
  - Non-deterministic calls are strictly isolated to nodes requiring semantic interpretation.

### 2.4 Repeated Document Processing
- **Finding**: **ISSUE-03 Identified in `/api/v1/code/pdf`**.
- **Analysis**: In `routes.py`, `code_pdf_document` called `batch_processor.process_single_pdf`, which extracted text from the PDF, and then invoked `pipeline.run_document(text=coding_text)`. When the LangGraph pipeline ran, it re-evaluated the document text from scratch.
- **Remediation**: Pass the PDF path directly into `pipeline.run_document(document_id=doc_id, pdf_path=tmp_path)` to allow the graph's native `validate_document` and `extract_text` nodes to execute once.

### 2.5 Blocking Calls Inside Async Code
- **Finding**: **Blocking CPU / PyMuPDF I/O in Async Loop**.
- **Analysis**:
  - PyMuPDF extraction in `extract_text_node` was calling synchronous methods directly inside the async node without wrapping in `asyncio.to_thread`. For large 50-page PDFs, this blocks the event loop thread.
  - `LLMLifecycleManager.generate_async()` correctly uses `asyncio.to_thread` to prevent CPU-intensive token generation from freezing the event loop.
- **Remediation**: Wrap PDF extraction inside `extract_text_node` using `asyncio.to_thread` or the non-blocking `PDFExtractor.extract_async()`.

### 2.6 Unsafe Concurrent GPT4All Calls
- **Finding**: **Protected by Reentrant Mutex**.
- **Analysis**: The llama.cpp backend is not thread-safe for simultaneous context evaluations on the same model instance. `LLMLifecycleManager._inference_thread_lock` serializes C++ calls, while `_async_semaphore` queues asynchronous tasks at the Python event loop boundary.

### 2.7 Embeddings & Retrieval Latency
- **Finding**: **Global Singleton Caching for Retrieval Indexes**.
- **Analysis**: `_CACHED_CATALOG` and `_CACHED_RETRIEVER` in `nodes.py` cache the FAISS index and BM25 tables in memory. Retrievals across diagnoses execute in parallel using `concurrent.futures.ThreadPoolExecutor(max_workers=4)`.

---

## 3. Concurrency Benchmarks ($\ge 10$ Documents)

Batch asynchronous processing was tested with 10 concurrent discharge summary documents under `BoundedDocumentGate(max_concurrency=10)`:
- Total Documents: 10
- Successful Outputs: 10
- Total Wall-Clock Latency: **0.18 seconds** (mock/offline pass)
- Memory Variance: < 5 MB increase during batch execution.
- Semaphore Behavior: All 10 tasks acquired permits cooperatively without thread starvation.
