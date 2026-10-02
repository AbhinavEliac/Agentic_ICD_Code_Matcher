"""Thread-safe lifecycle management, caching, and controlled concurrency for local LLM inference."""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from medical_coding.config.settings import Settings, get_settings
from medical_coding.models.exceptions import LLMInferenceError, LLMTimeoutError
from medical_coding.models.gpt4all_wrapper import GPT4AllWrapper
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LLMHealthReport:
    """Diagnostic health assessment of local LLM infrastructure."""

    status: Literal["healthy", "degraded", "unhealthy"]
    model_name: str
    model_path: str
    resolved_path: str
    file_exists: bool
    file_size_bytes: int
    is_loaded: bool
    load_count: int
    inference_count: int
    max_llm_concurrency: int
    max_document_concurrency: int
    configured_threads: int
    context_size: int
    details: str


class LLMLifecycleManager:
    """Singleton managing offline GGUF model lifecycle, single-load caching, and concurrency gates.

    ARCHITECTURAL PRINCIPLES:
    1. Single-Load Guarantee: The GGUF weights are loaded exactly once into RAM.
    2. Concurrency Separation:
       - Document-level concurrency (e.g. 10 PDFs) is handled by the async event loop / batch processor.
       - Model-level concurrency is strictly gated by an async semaphore and thread lock to protect
         CPU cache lines and avoid C++ llama.cpp context corruption.
    3. Thread-Safe: Uses double-checked reentrant locking across concurrent threads.
    """

    _instance: "LLMLifecycleManager | None" = None
    _singleton_lock: threading.RLock = threading.RLock()

    def __init__(
        self,
        settings: Settings | None = None,
        custom_wrapper: GPT4AllWrapper | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._load_lock = threading.RLock()
        self._inference_thread_lock = threading.Lock()

        # Concurrency gates
        self._async_semaphore: asyncio.Semaphore | None = None
        self._async_semaphore_lock = threading.Lock()

        # Lifecycle state
        self._wrapper: GPT4AllWrapper | None = custom_wrapper
        self._load_count: int = 1 if custom_wrapper is not None else 0
        self._inference_count: int = 0
        self._total_inference_time_ms: float = 0.0

    @classmethod
    def get_instance(
        cls,
        settings: Settings | None = None,
        custom_wrapper: GPT4AllWrapper | None = None,
    ) -> "LLMLifecycleManager":
        """Retrieve or create the singleton lifecycle manager instance."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls(settings=settings, custom_wrapper=custom_wrapper)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (used for testing or clean restarts)."""
        with cls._singleton_lock:
            if cls._instance is not None:
                logger.info("Resetting LLMLifecycleManager singleton instance.")
                cls._instance._wrapper = None
                cls._instance = None

    def _get_async_semaphore(self) -> asyncio.Semaphore:
        """Lazy-initialize asyncio.Semaphore bound to the active event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if self._async_semaphore is None or (
            loop is not None and getattr(self._async_semaphore, "_loop", None) is not loop
        ):
            with self._async_semaphore_lock:
                if self._async_semaphore is None or (
                    loop is not None and getattr(self._async_semaphore, "_loop", None) is not loop
                ):
                    limit = self.settings.max_llm_concurrency
                    self._async_semaphore = asyncio.Semaphore(limit)
        return self._async_semaphore

    def get_or_load_wrapper(self) -> GPT4AllWrapper:
        """Return the loaded model wrapper, loading it once if not already initialized."""
        if self._wrapper is not None:
            return self._wrapper

        with self._load_lock:
            if self._wrapper is not None:
                return self._wrapper

            logger.info("Initializing model wrapper for '%s'...", self.settings.model_name)
            wrapper = GPT4AllWrapper(
                model_name=self.settings.model_name,
                model_path=self.settings.model_path,
                model_type=self.settings.llm_model_type,
                n_threads=self.settings.model_threads,
                n_ctx=self.settings.model_context_size,
                temperature=self.settings.model_temperature,
                max_tokens=self.settings.model_max_tokens,
                top_k=self.settings.llm_top_k,
                top_p=self.settings.llm_top_p,
                repeat_penalty=self.settings.llm_repeat_penalty,
            )

            wrapper.load()
            self._wrapper = wrapper
            self._load_count += 1
            logger.info("Model loaded successfully (load_count=%d).", self._load_count)
            return self._wrapper

    def generate_sync(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute synchronous model generation protected by the inference thread lock."""
        wrapper = self.get_or_load_wrapper()

        # Thread lock ensures llama.cpp C++ context is accessed sequentially
        with self._inference_thread_lock:
            start_time = time.perf_counter()
            self._inference_count += 1

            logger.debug(
                "Starting sync LLM inference (call #%d, prompt_chars=%d)...",
                self._inference_count,
                len(prompt),
            )

            result = wrapper.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                **kwargs,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._total_inference_time_ms += elapsed_ms

            logger.debug("Completed sync LLM inference in %.2f ms", elapsed_ms)
            return result

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute asynchronous inference with concurrency gating and timeout handling.

        Protects CPU by queuing calls through the async semaphore (max_llm_concurrency)
        and offloading execution to worker threads without blocking the asyncio loop.
        """
        semaphore = self._get_async_semaphore()
        timeout_seconds = self.settings.llm_timeout_seconds

        async with semaphore:
            try:
                # Offload CPU execution to background thread while enforcing timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.generate_sync,
                        prompt,
                        max_tokens,
                        temperature,
                        stop_sequences,
                        **kwargs,
                    ),
                    timeout=timeout_seconds,
                )
                return result
            except TimeoutError as exc:
                logger.error(
                    "LLM inference timed out after %.1f seconds for prompt (%d chars)",
                    timeout_seconds,
                    len(prompt),
                )
                raise LLMTimeoutError(f"LLM inference timed out after {timeout_seconds}s") from exc
            except Exception as exc:
                if isinstance(exc, LLMInferenceError | LLMTimeoutError):
                    raise
                logger.exception("Unexpected error during async LLM inference: %s", exc)
                raise LLMInferenceError(f"Async LLM inference failed: {exc}") from exc

    def check_health(self) -> LLMHealthReport:
        """Run non-destructive diagnostic health check on local LLM infrastructure."""
        resolved = self.settings.get_resolved_model_path()
        file_exists = resolved.exists() and resolved.is_file()
        file_size = 0
        if file_exists:
            try:
                file_size = resolved.stat().st_size
            except OSError:
                file_size = 0

        is_loaded = self._wrapper is not None

        if file_exists and is_loaded:
            status = "healthy"
            details = "Local GGUF model is present on disk and loaded in memory."
        elif file_exists:
            status = "healthy"
            details = "Local GGUF model is present on disk and ready for on-demand loading."
        else:
            status = "degraded"
            details = (
                f"Model file not found at '{resolved}'. Offline mode is active; "
                "inference calls will raise FileNotFoundError until weights are placed on disk."
            )

        return LLMHealthReport(
            status=status,
            model_name=self.settings.model_name,
            model_path=str(self.settings.model_path),
            resolved_path=str(resolved),
            file_exists=file_exists,
            file_size_bytes=file_size,
            is_loaded=is_loaded,
            load_count=self._load_count,
            inference_count=self._inference_count,
            max_llm_concurrency=self.settings.max_llm_concurrency,
            max_document_concurrency=self.settings.max_document_concurrency,
            configured_threads=self.settings.model_threads,
            context_size=self.settings.model_context_size,
            details=details,
        )


def check_llm_health(settings: Settings | None = None) -> LLMHealthReport:
    """Convenience functional interface for LLM health check."""
    manager = LLMLifecycleManager.get_instance(settings=settings)
    return manager.check_health()
