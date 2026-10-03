"""Configuration management using Pydantic Settings and environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Environment & Diagnostics
    environment: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = False
    offline_mode: bool = True

    # Logging
    log_level: str = "INFO"
    log_json_format: bool = False

    # Local LLM (GPT4All / GGUF) Configuration
    model_name: str = Field(
        default="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        description="GGUF model file name to load from local storage.",
    )
    model_path: Path = Field(
        default=Path("./models/gguf"),
        description="Directory or full path pointing to local GGUF model files.",
    )
    model_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature; set to 0.0 for deterministic extraction.",
    )
    model_max_tokens: int = Field(
        default=1024,
        ge=16,
        le=8192,
        description="Maximum generation token ceiling.",
    )
    model_threads: int = Field(
        default=4,
        ge=1,
        description="CPU execution threads allocated for local GGUF inference.",
    )
    model_context_size: int = Field(
        default=2048,
        ge=256,
        le=32768,
        description="Context window size (n_ctx) allocated in GPT4All.",
    )

    # Concurrency Architecture: Document-Level vs Model-Level
    max_document_concurrency: int = Field(
        default=10,
        ge=1,
        description="Maximum simultaneous asynchronous document extraction and pipeline tasks.",
    )
    max_llm_concurrency: int = Field(
        default=1,
        ge=1,
        description="Maximum simultaneous local LLM inferences (typically 1 on CPU to avoid cache thrashing).",
    )

    # LLM Operational Limits & Generation Tuning
    llm_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        description="Maximum execution timeout in seconds for a single LLM inference call.",
    )
    llm_model_type: str = Field(
        default="llama",
        description="Architecture family for GPT4All (e.g. llama, mpt, replit).",
    )
    llm_top_k: int = Field(
        default=40,
        ge=1,
        description="Top-k sampling parameter for GPT4All generation.",
    )
    llm_top_p: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter for GPT4All generation.",
    )
    llm_repeat_penalty: float = Field(
        default=1.18,
        ge=0.0,
        description="Repetition penalty for GPT4All generation.",
    )

    # Backwards-compatibility aliases
    llm_model_path: Path | None = Field(
        default=None,
        description="Optional full path alias for model_path / model_name.",
    )
    llm_n_threads: int | None = Field(
        default=None,
        description="Optional alias for model_threads.",
    )
    llm_max_tokens: int | None = Field(
        default=None,
        description="Optional alias for model_max_tokens.",
    )
    llm_temperature: float | None = Field(
        default=None,
        description="Optional alias for model_temperature.",
    )
    max_batch_concurrency: int | None = Field(
        default=None,
        description="Optional alias for max_document_concurrency.",
    )

    # Local Dense Embedding Configuration
    embedding_model_path: Path = Field(
        default=Path("./models/embeddings/bge-small-en-v1.5"),
        description="Local directory containing offline SentenceTransformer model files.",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for local embeddings: 'cpu', 'cuda', or 'mps'.",
    )

    # Local ICD-10-CM Source Catalog & Pre-computed Index Directories
    icd_dataset_path: Path = Field(
        default=Path("./data/icd10/icd10cm_order_2026.txt"),
        description="Path to authoritative CMS/CDC ICD-10-CM order/tabular flat file.",
    )
    database_dir: Path = Field(
        default=Path("./Database"),
        description="Directory containing local clinical database workbooks (Database_2.xlsx, Database_1.xls).",
    )
    icd_index_dir: Path = Field(
        default=Path("./data/indexes"),
        description="Directory for persisted FAISS indices and BM25 tokenized caches.",
    )

    # Database & Storage
    database_url: str = Field(
        default="sqlite:///./data/medical_coding.db",
        description="SQLAlchemy database connection URL (defaults to local SQLite database in ./data/).",
    )

    # Retrieval Tuning Parameters
    bm25_top_k: int = Field(
        default=15,
        ge=1,
        description="Candidate count retrieved by lexical BM25 engine.",
    )
    faiss_top_k: int = Field(
        default=15,
        ge=1,
        description="Candidate count retrieved by dense FAISS vector engine.",
    )
    hybrid_top_k: int = Field(
        default=10,
        ge=1,
        description="Candidate pool size passed forward after hybrid rank fusion.",
    )
    min_candidate_score: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score below which candidate retrieval abstains.",
    )

    # Asynchronous PDF Processing Guardrails
    pdf_max_pages: int = Field(
        default=50,
        ge=1,
        description="Guardrail ceiling on pages parsed per clinical PDF.",
    )
    pdf_timeout_seconds: int = Field(
        default=120,
        ge=5,
        description="Timeout per document extraction pipeline execution.",
    )

    # FastAPI Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Ensure standard upper-case log levels."""
        return value.upper() if isinstance(value, str) else "INFO"

    @model_validator(mode="after")
    def sync_legacy_aliases(self) -> "Settings":
        """Synchronize legacy aliases with standard configuration fields."""
        if self.llm_n_threads is not None:
            self.model_threads = self.llm_n_threads
        else:
            self.llm_n_threads = self.model_threads

        if self.llm_max_tokens is not None:
            self.model_max_tokens = self.llm_max_tokens
        else:
            self.llm_max_tokens = self.model_max_tokens

        if self.llm_temperature is not None:
            self.model_temperature = self.llm_temperature
        else:
            self.llm_temperature = self.model_temperature

        if self.max_batch_concurrency is not None:
            self.max_document_concurrency = self.max_batch_concurrency
        else:
            self.max_batch_concurrency = self.max_document_concurrency

        if self.llm_model_path is not None:
            # If a full file path was explicitly provided in llm_model_path
            path_obj = Path(self.llm_model_path)
            if path_obj.suffix.lower() == ".gguf" or path_obj.is_file():
                self.model_path = path_obj.parent
                self.model_name = path_obj.name
            else:
                self.model_path = path_obj
        else:
            # Construct default llm_model_path from model_path + model_name
            self.llm_model_path = self.get_resolved_model_path()

        return self

    def get_resolved_model_path(self) -> Path:
        """Return the fully resolved Path to the GGUF model file."""
        path_obj = Path(self.model_path)
        if path_obj.suffix.lower() == ".gguf":
            return path_obj
        return path_obj / self.model_name

    def get_database_workbook_path(self) -> Path | None:
        """Resolve the active multi-workbook database file if present."""
        if self.database_dir.is_dir():
            target = self.database_dir / "Database_2.xlsx"
            if target.exists():
                return target
            candidates = sorted(list(self.database_dir.glob("*.xlsx")) + list(self.database_dir.glob("*.xls")))
            if candidates:
                return candidates[0]
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings.

    Use lru_cache to avoid re-reading disk/environment on repeated calls.
    Tests can call `get_settings.cache_clear()` to reload under custom env vars.
    """
    return Settings()
