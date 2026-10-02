"""LangChain-compatible local LLM interface wrapping GPT4All with controlled concurrency."""

from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from pydantic import Field, PrivateAttr

from medical_coding.config.settings import Settings, get_settings
from medical_coding.models.lifecycle import LLMLifecycleManager
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class LocalGPT4AllLangChainLLM(LLM):
    """LangChain-compatible LLM wrapping local offline GPT4All inference.

    Features:
    - Thread-safe and shared memory: delegates to the singleton LLMLifecycleManager.
    - Concurrency-controlled: prevents multi-thread C++ crashes on CPU inference.
    - Native async support (_acall) for LangChain and LangGraph workflows.
    - Zero cloud dependencies: 100% offline local execution.
    """

    model_name: str = Field(
        default="",
        description="Name of the GGUF model file.",
    )
    model_path: str = Field(
        default="",
        description="Path to local GGUF model directory or file.",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature for generation.",
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens to generate.",
    )
    threads: int = Field(
        default=4,
        description="CPU execution threads allocated.",
    )
    context_size: int = Field(
        default=2048,
        description="Context window size (n_ctx).",
    )
    timeout_seconds: float = Field(
        default=120.0,
        description="Generation timeout ceiling.",
    )

    _settings: Settings | None = PrivateAttr(default=None)

    def __init__(self, settings: Settings | None = None, **kwargs: Any) -> None:
        cfg = settings or get_settings()
        kwargs.setdefault("model_name", cfg.model_name)
        kwargs.setdefault("model_path", str(cfg.model_path))
        kwargs.setdefault("temperature", cfg.model_temperature)
        kwargs.setdefault("max_tokens", cfg.model_max_tokens)
        kwargs.setdefault("threads", cfg.model_threads)
        kwargs.setdefault("context_size", cfg.model_context_size)
        kwargs.setdefault("timeout_seconds", cfg.llm_timeout_seconds)
        super().__init__(**kwargs)
        self._settings = cfg

    @property
    def _llm_type(self) -> str:
        """Return type identifier for LangChain telemetry."""
        return "gpt4all_local"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters for tracking and serialization."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "threads": self.threads,
            "context_size": self.context_size,
            "timeout_seconds": self.timeout_seconds,
            "offline_only": True,
        }

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously execute local generation through the lifecycle manager."""
        manager = LLMLifecycleManager.get_instance(settings=self._settings)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        return manager.generate_sync(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop,
            **kwargs,
        )

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously execute local generation through the concurrency gate."""
        manager = LLMLifecycleManager.get_instance(settings=self._settings)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        return await manager.generate_async(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop,
            **kwargs,
        )
