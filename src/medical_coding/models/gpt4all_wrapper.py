"""Local GPT4All model wrapper executing strictly offline inference."""

from pathlib import Path
from typing import Any

from gpt4all import GPT4All

from medical_coding.models.exceptions import LLMInferenceError, LLMInitializationError
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class GPT4AllWrapper:
    """Wrapper around gpt4all.GPT4All with offline enforcement and parameter binding."""

    def __init__(
        self,
        model_name: str,
        model_path: Path,
        model_type: str = "llama",
        n_threads: int = 4,
        n_ctx: int = 2048,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_k: int = 40,
        top_p: float = 0.4,
        repeat_penalty: float = 1.18,
    ) -> None:
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

        self._resolved_file = self._resolve_file_path()
        self._model: GPT4All | None = None

    def _resolve_file_path(self) -> Path:
        """Resolve the exact filesystem path to the GGUF file."""
        if self.model_path.suffix.lower() == ".gguf" or self.model_path.is_file():
            return self.model_path
        return self.model_path / self.model_name

    @property
    def resolved_file_path(self) -> Path:
        return self._resolved_file

    def is_weight_file_present(self) -> bool:
        """Check whether the configured GGUF model file exists on disk."""
        return self._resolved_file.exists() and self._resolved_file.is_file()

    def get_file_size_bytes(self) -> int:
        """Return the byte size of the model file on disk, or 0 if missing."""
        if self.is_weight_file_present():
            try:
                return self._resolved_file.stat().st_size
            except OSError:
                return 0
        return 0

    def load(self) -> None:
        """Load GGUF model into memory offline.

        CRITICAL ARCHITECTURAL GUARANTEE:
        allow_download is unconditionally False. No network download will ever be attempted.
        """
        if self._model is not None:
            return

        if not self.is_weight_file_present():
            raise LLMInitializationError(
                f"Local GGUF model not found at '{self._resolved_file}'. "
                "Per offline architectural constraints, automatic download is disabled. "
                "Ensure the required GGUF weights are placed in the configured directory."
            )

        logger.info(
            "Loading local GGUF model '%s' from '%s' (threads=%d, ctx=%d, device=cpu)...",
            self.model_name,
            self._resolved_file.parent,
            self.n_threads,
            self.n_ctx,
        )

        try:
            # GPT4All expects model_path as directory and model_name as filename
            search_dir = str(self._resolved_file.parent)
            file_name = self._resolved_file.name

            self._model = GPT4All(
                model_name=file_name,
                model_path=search_dir,
                model_type=self.model_type,
                allow_download=False,  # STRICT OFFLINE ENFORCEMENT
                n_threads=self.n_threads,
                n_ctx=self.n_ctx,
                device="cpu",
                verbose=False,
            )
            logger.info("Successfully loaded local GGUF model into memory.")
        except Exception as exc:
            logger.exception("Failed to initialize GPT4All backend: %s", exc)
            raise LLMInitializationError(f"Failed to initialize GPT4All runtime: {exc}") from exc

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate response text from local GGUF model synchronously.

        Args:
            prompt: Text prompt for generation.
            max_tokens: Override max tokens (defaults to configured max_tokens).
            temperature: Override temperature (defaults to configured temperature).
            stop_sequences: List of strings where generation must stop.
            **kwargs: Additional parameters passed to GPT4All.generate().

        Returns:
            Generated response string.
        """
        if self._model is None:
            self.load()

        assert self._model is not None, "Model must be loaded before generation"

        tokens_to_gen = max_tokens if max_tokens is not None else self.max_tokens
        temp_val = temperature if temperature is not None else self.temperature

        try:
            raw_output = self._model.generate(
                prompt=prompt,
                max_tokens=tokens_to_gen,
                temp=temp_val,
                top_k=kwargs.get("top_k", self.top_k),
                top_p=kwargs.get("top_p", self.top_p),
                repeat_penalty=kwargs.get("repeat_penalty", self.repeat_penalty),
                streaming=False,
            )

            result_text = str(raw_output)

            # Apply stop sequences if requested
            if stop_sequences:
                for stop_token in stop_sequences:
                    if stop_token in result_text:
                        result_text = result_text.split(stop_token)[0]

            return result_text.strip()

        except Exception as exc:
            logger.error("Error during GPT4All inference generation: %s", exc)
            raise LLMInferenceError(f"GPT4All generation failed: {exc}") from exc

    def get_diagnostic_info(self) -> dict[str, Any]:
        """Return diagnostic state of the wrapper."""
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "resolved_file": str(self._resolved_file),
            "file_exists": self.is_weight_file_present(),
            "file_size_bytes": self.get_file_size_bytes(),
            "is_loaded": self._model is not None,
            "threads": self.n_threads,
            "context_size": self.n_ctx,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "offline_only": True,
        }
