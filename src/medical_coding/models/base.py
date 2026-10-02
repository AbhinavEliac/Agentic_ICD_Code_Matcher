"""Abstract interfaces for local LLMs and dense embedding models."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLocalLLM(ABC):
    """Abstract base class for local GGUF / GPT4All model inference."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Execute text generation against the local offline model.

        Args:
            prompt: Formatted prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 for deterministic output).
            stop_sequences: Optional tokens where generation should halt.

        Returns:
            Generated text string.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return diagnostic metadata about the loaded model."""
        pass


class BaseLocalEmbeddings(ABC):
    """Abstract base class for local dense vector embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for semantic vector search."""
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document descriptions."""
        pass
