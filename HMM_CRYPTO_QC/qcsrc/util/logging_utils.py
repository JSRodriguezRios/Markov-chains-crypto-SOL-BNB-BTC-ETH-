"""Logging utilities for QuantConnect HMM project."""
from __future__ import annotations

import logging


def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with sane defaults.

    Parameters
    ----------
    name: str
        Logger name, usually ``__name__`` of the caller.
    level: int, default logging.INFO
        Desired logging level. INFO is used to match QC cloud defaults.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_phase_checkpoint(logger: logging.Logger, message: str) -> None:
    """Emit a standardized checkpoint log message.

    This helper keeps debugging output consistent with the CODING_YAML
    requirement to surface phase transitions or milestones.
    """
    if not message:
        return
    logger.info("[checkpoint] %s", message)
