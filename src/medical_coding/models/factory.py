"""Factory and lifecycle coordination for local offline GGUF and embedding models."""

from pathlib import Path
from typing import Any

from medical_coding.config.settings import Settings, get_settings
from medical_coding.models.base import BaseLocalEmbeddings, BaseLocalLLM
from medical_coding.models.langchain_llm import LocalGPT4AllLangChainLLM
from medical_coding.models.lifecycle import LLMHealthReport, LLMLifecycleManager
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class GPT4AllLocalLLM(BaseLocalLLM):
    """Local GGUF model wrapper delegating execution to the singleton LLMLifecycleManager."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_path: Path | None = None,
        model_type: str = "llama",
        n_threads: int = 4,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_path = (
            Path(model_path) if model_path else self.settings.get_resolved_model_path()
        )
        self.model_type = model_type
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self.temperature = temperature

    def load_model(self) -> None:
        """Trigger one-time loading of the local GGUF weights via lifecycle manager."""
        manager = LLMLifecycleManager.get_instance(settings=self.settings)
        manager.get_or_load_wrapper()

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Generate text from the local GGUF model using thread-safe execution."""
        manager = LLMLifecycleManager.get_instance(settings=self.settings)
        return manager.generate_sync(
            prompt=prompt,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            stop_sequences=stop_sequences,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Return diagnostic health and status metadata."""
        manager = LLMLifecycleManager.get_instance(settings=self.settings)
        report = manager.check_health()
        return {
            "model_name": report.model_name,
            "resolved_path": report.resolved_path,
            "file_exists": report.file_exists,
            "is_loaded": report.is_loaded,
            "load_count": report.load_count,
            "inference_count": report.inference_count,
            "n_threads": report.configured_threads,
            "max_llm_concurrency": report.max_llm_concurrency,
            "max_document_concurrency": report.max_document_concurrency,
            "offline_only": True,
        }


class FastLocalEmbeddings(BaseLocalEmbeddings):
    """Deterministic, zero-network offline dense embedding generator.

    Uses subword and token feature hashing with L2-normalized dense projections.
    Ideal for test fixtures, air-gapped environments, and rapid local evaluation.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _hash_token(self, token: str) -> tuple[int, float]:
        import hashlib

        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % self.dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        return idx, sign

    def _embed_single(self, text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()

        for w in words:
            idx, sign = self._hash_token(f"w_{w}")
            vec[idx] += sign * 1.5
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    trigram = w[i : i + 3]
                    t_idx, t_sign = self._hash_token(f"c_{trigram}")
                    vec[t_idx] += t_sign * 0.5

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(t) for t in texts]


class SentenceTransformerLocalEmbeddings(BaseLocalEmbeddings):
    """Local offline dense embedding generator using SentenceTransformers."""

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._model: Any = None

        if not self.model_path.exists():
            logger.warning(
                "Local embedding directory not found at %s. FastLocalEmbeddings will be used as fallback.",
                self.model_path,
            )

    def load_model(self) -> None:
        """Load local sentence transformer model from disk without remote network calls."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local embedding model not found at '{self.model_path}'. "
                "Ensure local sentence transformer weights are present offline."
            )

        from sentence_transformers import SentenceTransformer

        logger.info("Loading offline SentenceTransformer weights from %s", self.model_path)
        self._model = SentenceTransformer(
            str(self.model_path),
            device=self.device,
            local_files_only=True,
        )

    def embed_query(self, text: str) -> list[float]:
        self.load_model()
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.load_model()
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class ModelFactory:
    """Factory creating and providing local offline models configured via Settings."""

    @staticmethod
    def create_llm(settings: Settings | None = None) -> BaseLocalLLM:
        """Create a local GGUF/GPT4All LLM wrapper instance adhering to BaseLocalLLM."""
        cfg = settings or get_settings()
        return GPT4AllLocalLLM(settings=cfg)

    @staticmethod
    def create_langchain_llm(settings: Settings | None = None) -> LocalGPT4AllLangChainLLM:
        """Create a LangChain-compatible LLM instance bound to the lifecycle manager."""
        cfg = settings or get_settings()
        return LocalGPT4AllLangChainLLM(settings=cfg)

    @staticmethod
    def get_llm_manager(settings: Settings | None = None) -> LLMLifecycleManager:
        """Return the singleton LLMLifecycleManager instance."""
        return LLMLifecycleManager.get_instance(settings=settings)

    @staticmethod
    def check_llm_health(settings: Settings | None = None) -> LLMHealthReport:
        """Execute diagnostic health assessment of local LLM infrastructure."""
        manager = LLMLifecycleManager.get_instance(settings=settings)
        return manager.check_health()

    @staticmethod
    def create_embeddings(
        settings: Settings | None = None,
        force_local_fast: bool = False,
    ) -> BaseLocalEmbeddings:
        """Create a local dense embedding wrapper instance.

        If force_local_fast is True or the configured embedding path does not exist on disk,
        returns FastLocalEmbeddings for deterministic, zero-network local operation.
        """
        cfg = settings or get_settings()
        if force_local_fast or not cfg.embedding_model_path.exists():
            return FastLocalEmbeddings(dim=384)
        return SentenceTransformerLocalEmbeddings(
            model_path=cfg.embedding_model_path,
            device=cfg.embedding_device,
        )
