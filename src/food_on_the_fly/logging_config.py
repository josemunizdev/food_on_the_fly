"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_RICH_FORMAT = "%(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: LogLevel = "INFO", fmt: str | None = None) -> None:
    """Configure the root logger for the application.

    Uses ``rich.logging.RichHandler`` when available for colorized,
    tracebacks-with-locals output, and falls back to a plain
    ``StreamHandler`` otherwise. Idempotent: safe to call multiple times.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            show_path=False,
            markup=False,
        )
        handler.setFormatter(
            logging.Formatter(fmt=fmt or _RICH_FORMAT, datefmt=_DEFAULT_DATEFMT)
        )
    except ImportError:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(fmt=fmt or _DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        )

    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
