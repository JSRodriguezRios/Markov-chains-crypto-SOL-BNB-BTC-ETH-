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


def hmm_model_output_path(symbol: str) -> Path:
    """Return the canonical persistence path for trained HMM models."""

    directory = project_root() / "data" / "models" / "hmm"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{symbol}_hmm.pkl"


def diagnostics_output_path(symbol: str) -> Path:
    """Return the diagnostics report location for ``symbol``."""

    directory = project_root() / "data" / "models" / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{symbol}_report.json"


def metrics_output_path(symbol: str) -> Path:
    """Return the metrics report location for ``symbol``."""

    directory = project_root() / "data" / "models" / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{symbol}_metrics.json"


def _custom_data_directory(category: str) -> Path:
    """Return the QuantConnect custom data directory for ``category``."""

    directory = project_root() / "qc_data" / "custom" / category
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def hmm_probability_export_path(symbol: str) -> Path:
    """Return the export path for HMM probability time series."""

    filename = f"{symbol.lower()}.csv"
    return _custom_data_directory("HMMStateProba") / filename


def liquidity_export_path(symbol: str) -> Path:
    """Return the export path for liquidity custom data."""

    filename = f"{symbol.lower()}.csv"
    return _custom_data_directory("LiquidityBitAsk") / filename


def sentiment_export_path(symbol: str) -> Path:
    """Return the export path for sentiment custom data."""

    filename = f"{symbol.lower()}.csv"
    return _custom_data_directory("MarketSentiment") / filename
