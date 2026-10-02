"""FastAPI application factory for the local medical coding service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medical_coding.api.routes import router
from medical_coding.config.settings import Settings, get_settings
from medical_coding.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle management."""
    settings: Settings = app.state.settings
    configure_logging(settings)
    logger.info("Initializing Local Medical ICD-10-CM Coding Service...")
    logger.info("Operating Mode: OFFLINE (Cloud APIs strictly prohibited)")
    logger.info("Configured LLM path: %s", settings.llm_model_path)
    logger.info("Configured ICD dataset: %s", settings.icd_dataset_path)

    yield

    logger.info("Shutting down Medical Coding Service.")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    cfg = settings or get_settings()

    application = FastAPI(
        title="Local Medical ICD-10-CM Coding System",
        description=(
            "Offline clinical coding pipeline using local GGUF models, hybrid retrieval, "
            "LangGraph state orchestration, and deterministic validation."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    application.state.settings = cfg

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application


app = create_app()
