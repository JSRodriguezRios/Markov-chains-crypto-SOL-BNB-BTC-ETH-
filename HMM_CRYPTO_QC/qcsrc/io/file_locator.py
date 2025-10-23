"""Path resolution helpers for project datasets."""
from __future__ import annotations

from pathlib import Path

from ..util.logging_utils import create_logger

LOGGER = create_logger(__name__)


def project_root() -> Path:
    """Return the project root directory."""

    return Path(__file__).resolve().parents[2]


def external_source_directory(source: str, symbol: str) -> Path:
    """Resolve the folder where external ``source`` data for ``symbol`` lives."""

    directory = project_root() / "data" / "external" / source / symbol
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def interim_output_path(symbol: str) -> Path:
    """Return the canonical parquet path for ``symbol`` interim data."""

    directory = project_root() / "data" / "interim"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{symbol}.parquet"


def processed_output_path(symbol: str) -> Path:
    """Return the canonical path for processed feature matrices."""

    directory = project_root() / "data" / "processed"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{symbol}_features.parquet"


def scaler_output_path(symbol: str) -> Path:
    """Return the storage path for serialized feature scalers."""

    directory = project_root() / "data" / "models" / "hmm"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{symbol}_scaler.pkl"
