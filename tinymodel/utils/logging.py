"""Structured logging utilities."""

import logging

from rich.logging import RichHandler


def get_logger(name: str = "tinymodel") -> logging.Logger:
    """Return a configured logger with rich formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
