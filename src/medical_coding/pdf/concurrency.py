"""Bounded concurrency manager separating document-level tasks from model-level inference."""

import asyncio
from typing import Any

from medical_coding.config.settings import Settings, get_settings
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)


class BoundedDocumentGate:
    """Manages asynchronous document-level concurrency ceilings and lifecycle cancellation.

    ARCHITECTURAL PRINCIPLE:
    Document-level concurrency allows multiple PDFs (e.g. >= 10 documents) to be ingested,
    parsed with PyMuPDF, normalized, and partitioned in parallel, without spawning multiple LLMs.
    """

    def __init__(
        self, max_concurrency: int | None = None, settings: Settings | None = None
    ) -> None:
        cfg = settings or get_settings()
        self.max_concurrency = max_concurrency or cfg.max_document_concurrency
        self._semaphore: asyncio.Semaphore | None = None
        self._active_tasks: set[asyncio.Task[Any]] = set()

    def get_semaphore(self) -> asyncio.Semaphore:
        """Lazy-initialize asyncio.Semaphore bound to current active event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if self._semaphore is None or (
            loop is not None and getattr(self._semaphore, "_loop", None) is not loop
        ):
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def run_bounded(self, coro: Any) -> Any:
        """Run coroutine within the document-level concurrency semaphore."""
        sem = self.get_semaphore()
        async with sem:
            return await coro
