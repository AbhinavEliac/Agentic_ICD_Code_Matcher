"""Logging configuration and structured logger utilities."""

import logging
import sys

from medical_coding.config.settings import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logger format and level based on application settings."""
    cfg = settings or get_settings()

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    level = getattr(logging, cfg.log_level, logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
        root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance."""
    return logging.getLogger(name)
