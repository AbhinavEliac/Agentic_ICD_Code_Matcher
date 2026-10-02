"""Model exports for offline GGUF inference, LangChain integration, and lifecycle management."""

from medical_coding.models.base import BaseLocalEmbeddings, BaseLocalLLM
from medical_coding.models.exceptions import (
    LLMError,
    LLMInferenceError,
    LLMInitializationError,
    LLMTimeoutError,
)
from medical_coding.models.factory import (
    GPT4AllLocalLLM,
    ModelFactory,
    SentenceTransformerLocalEmbeddings,
)
from medical_coding.models.gpt4all_wrapper import GPT4AllWrapper
from medical_coding.models.langchain_llm import LocalGPT4AllLangChainLLM
from medical_coding.models.lifecycle import (
    LLMHealthReport,
    LLMLifecycleManager,
    check_llm_health,
)

__all__ = [
    "BaseLocalEmbeddings",
    "BaseLocalLLM",
    "GPT4AllLocalLLM",
    "GPT4AllWrapper",
    "LLMError",
    "LLMHealthReport",
    "LLMInferenceError",
    "LLMInitializationError",
    "LLMLifecycleManager",
    "LLMTimeoutError",
    "LocalGPT4AllLangChainLLM",
    "ModelFactory",
    "SentenceTransformerLocalEmbeddings",
    "check_llm_health",
]
