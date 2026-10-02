"""Unit and integration tests for local GPT4All LLM infrastructure."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from medical_coding.config.settings import Settings
from medical_coding.models.exceptions import LLMInitializationError
from medical_coding.models.factory import ModelFactory
from medical_coding.models.gpt4all_wrapper import GPT4AllWrapper
from medical_coding.models.langchain_llm import LocalGPT4AllLangChainLLM
from medical_coding.models.lifecycle import (
    LLMLifecycleManager,
    check_llm_health,
)


@pytest.fixture(autouse=True)
def clean_lifecycle_manager() -> None:
    """Ensure singleton manager is fresh before and after each test."""
    LLMLifecycleManager.reset_instance()
    yield
    LLMLifecycleManager.reset_instance()


@pytest.fixture
def mock_gpt4all_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Fixture providing a mock GPT4All backend with a dummy GGUF weight file."""
    model_file = tmp_path / "test-model.gguf"
    model_file.write_bytes(b"GGUF_TEST_HEADER_BYTES_1234567890")

    mock_backend = MagicMock()
    mock_backend.generate.return_value = "Trivial clinical test response."

    mock_class = MagicMock(return_value=mock_backend)
    monkeypatch.setattr("medical_coding.models.gpt4all_wrapper.GPT4All", mock_class)

    test_settings = Settings(
        model_name=model_file.name,
        model_path=tmp_path,
        model_temperature=0.0,
        model_max_tokens=256,
        model_threads=4,
        model_context_size=1024,
        max_document_concurrency=10,
        max_llm_concurrency=1,
        llm_timeout_seconds=30.0,
    )

    return {
        "model_file": model_file,
        "mock_class": mock_class,
        "mock_backend": mock_backend,
        "settings": test_settings,
    }


def test_configuration_loads_model_and_concurrency_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 1: Verify all required MODEL_* and CONCURRENCY configuration parameters load from environment."""
    monkeypatch.setenv("MODEL_NAME", "custom_model.gguf")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.2")
    monkeypatch.setenv("MODEL_MAX_TOKENS", "512")
    monkeypatch.setenv("MODEL_THREADS", "8")
    monkeypatch.setenv("MODEL_CONTEXT_SIZE", "4096")
    monkeypatch.setenv("MAX_DOCUMENT_CONCURRENCY", "12")
    monkeypatch.setenv("MAX_LLM_CONCURRENCY", "2")

    cfg = Settings()
    assert cfg.model_name == "custom_model.gguf"
    assert cfg.model_path == tmp_path
    assert cfg.model_temperature == 0.2
    assert cfg.model_max_tokens == 512
    assert cfg.model_threads == 8
    assert cfg.model_context_size == 4096
    assert cfg.max_document_concurrency == 12
    assert cfg.max_llm_concurrency == 2
    assert cfg.get_resolved_model_path() == tmp_path / "custom_model.gguf"


def test_model_can_be_initialized_offline(mock_gpt4all_env: dict) -> None:
    """Test 2: Verify model can be initialized offline with allow_download=False."""
    settings = mock_gpt4all_env["settings"]
    mock_class = mock_gpt4all_env["mock_class"]

    wrapper = GPT4AllWrapper(
        model_name=settings.model_name,
        model_path=settings.model_path,
        n_threads=settings.model_threads,
        n_ctx=settings.model_context_size,
    )

    assert wrapper.is_weight_file_present() is True
    assert wrapper.get_file_size_bytes() > 0

    wrapper.load()
    assert wrapper._model is not None

    # Verify allow_download was strictly False
    mock_class.assert_called_once()
    _, kwargs = mock_class.call_args
    assert kwargs.get("allow_download") is False
    assert kwargs.get("device") == "cpu"
    assert kwargs.get("n_threads") == 4
    assert kwargs.get("n_ctx") == 1024


def test_model_can_produce_trivial_response_sync_and_async(mock_gpt4all_env: dict) -> None:
    """Test 3: Verify model can produce a trivial response synchronously and asynchronously."""
    settings = mock_gpt4all_env["settings"]
    manager = LLMLifecycleManager.get_instance(settings=settings)

    # Synchronous test
    sync_resp = manager.generate_sync("Ping")
    assert sync_resp == "Trivial clinical test response."

    # Asynchronous test
    async def run_async() -> str:
        return await manager.generate_async("Ping async")

    async_resp = asyncio.run(run_async())
    assert async_resp == "Trivial clinical test response."
    assert manager.check_health().inference_count == 2


def test_model_is_not_reloaded_unnecessarily(mock_gpt4all_env: dict) -> None:
    """Test 4: Verify model is loaded once and reused across multiple sequential and parallel calls."""
    settings = mock_gpt4all_env["settings"]
    mock_class = mock_gpt4all_env["mock_class"]
    manager = LLMLifecycleManager.get_instance(settings=settings)

    # Execute 5 inferences
    for i in range(5):
        resp = manager.generate_sync(f"Query {i}")
        assert resp == "Trivial clinical test response."

    # Crucial assertion: Model was initialized exactly once
    assert mock_class.call_count == 1
    assert manager.check_health().load_count == 1
    assert manager.check_health().inference_count == 5


def test_langchain_interface_invoke_and_ainvoke(mock_gpt4all_env: dict) -> None:
    """Verify LangChain-compatible LocalGPT4AllLangChainLLM interface works with invoke and ainvoke."""
    settings = mock_gpt4all_env["settings"]
    llm = LocalGPT4AllLangChainLLM(settings=settings)

    assert llm._llm_type == "gpt4all_local"

    # Synchronous LangChain invoke
    output = llm.invoke("Identify diagnosis")
    assert output == "Trivial clinical test response."

    # Asynchronous LangChain ainvoke
    async def run_langchain_async() -> str:
        return await llm.ainvoke("Identify diagnosis async")

    async_output = asyncio.run(run_langchain_async())
    assert async_output == "Trivial clinical test response."


def test_controlled_concurrency_handles_multiple_concurrent_requests(
    mock_gpt4all_env: dict,
) -> None:
    """Verify that multiple concurrent async calls (e.g. 10 PDFs) are handled safely under MAX_LLM_CONCURRENCY."""
    settings = mock_gpt4all_env["settings"]
    manager = LLMLifecycleManager.get_instance(settings=settings)

    async def execute_batch() -> list[str]:
        # Launch 10 simultaneous async calls simulating 10 PDFs requesting LLM inference
        tasks = [manager.generate_async(f"PDF doc #{i}") for i in range(10)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(execute_batch())
    assert len(results) == 10
    assert all(r == "Trivial clinical test response." for r in results)
    assert manager.check_health().inference_count == 10
    # Model should still only have been loaded once!
    assert manager.check_health().load_count == 1


def test_health_check_reports_accurate_state(mock_gpt4all_env: dict) -> None:
    """Verify health check diagnostics before and after loading."""
    settings = mock_gpt4all_env["settings"]
    report = check_llm_health(settings=settings)

    assert report.file_exists is True
    assert report.status == "healthy"
    assert report.max_document_concurrency == 10
    assert report.max_llm_concurrency == 1
    assert report.configured_threads == 4


def test_missing_weight_file_raises_initialization_error(tmp_path: Path) -> None:
    """Verify that attempting to load a non-existent model raises an informative offline error."""
    settings = Settings(
        model_name="nonexistent.gguf",
        model_path=tmp_path,
    )
    wrapper = GPT4AllWrapper(
        model_name=settings.model_name,
        model_path=settings.model_path,
    )

    assert wrapper.is_weight_file_present() is False
    with pytest.raises(LLMInitializationError, match="Per offline architectural constraints"):
        wrapper.load()


def test_model_factory_methods(mock_gpt4all_env: dict) -> None:
    """Verify that ModelFactory factory methods correctly instantiate components."""
    settings = mock_gpt4all_env["settings"]
    base_llm = ModelFactory.create_llm(settings=settings)
    langchain_llm = ModelFactory.create_langchain_llm(settings=settings)
    manager = ModelFactory.get_llm_manager(settings=settings)
    health = ModelFactory.check_llm_health(settings=settings)

    assert base_llm is not None
    assert langchain_llm is not None
    assert manager is not None
    assert health.status == "healthy"
