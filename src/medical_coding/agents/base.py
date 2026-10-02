"""Base agent interface for local LLM-driven clinical reasoning."""

from abc import ABC, abstractmethod
from typing import Any

from medical_coding.models.base import BaseLocalLLM


class BaseAgent(ABC):
    """Abstract base class for modular clinical reasoning agents."""

    def __init__(self, llm: BaseLocalLLM) -> None:
        self.llm = llm

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute agent reasoning step."""
        pass
