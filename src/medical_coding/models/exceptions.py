"""Exceptions for local LLM lifecycle, initialization, and inference."""


class LLMError(Exception):
    """Base exception for local LLM failures."""

    pass


class LLMInitializationError(LLMError):
    """Raised when local GGUF model fails to initialize or weight file is missing."""

    pass


class LLMInferenceError(LLMError):
    """Raised when model inference fails during generation."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when model generation exceeds configured timeout."""

    pass
