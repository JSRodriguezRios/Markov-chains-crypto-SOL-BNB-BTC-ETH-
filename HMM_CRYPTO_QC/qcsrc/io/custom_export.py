"""QuantConnect custom data export helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..util.logging_utils import create_logger, log_phase_checkpoint
from ..util.timezones import ensure_utc
from .file_locator import (
    hmm_probability_export_path,
    liquidity_export_path,
    sentiment_export_path,
)

try:  # pragma: no cover - optional pandas dependency
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

LOGGER = create_logger(__name__)


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover - defensive guard
        raise ImportError(
            "pandas is required to export custom data. Install pandas to continue."
        )


def _normalise_time_column(frame: "pd.DataFrame") -> "pd.Series":
    if "time" in frame.columns:
        series = frame["time"]
    elif "timestamp" in frame.columns:
        series = frame["timestamp"]
    else:
        raise KeyError("Frame must contain a 'time' or 'timestamp' column for export.")

    series = ensure_utc(series)
    return series.dt.tz_convert("UTC").dt.tz_localize(None)


def _prepare_frame(
    frame: "pd.DataFrame",
    column_order: Sequence[str],
    required_columns: Iterable[str],
) -> "pd.DataFrame":
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns for export: {missing}")

    output = frame.copy()
    output["time"] = _normalise_time_column(output)
    output = output.sort_values("time").drop_duplicates("time", keep="last")
    return output.loc[:, column_order]


def _export_csv(frame: "pd.DataFrame", destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    LOGGER.info("Exported custom data to %s", destination)
    return destination


def export_probability_timeseries(symbol: str, frame: "pd.DataFrame") -> Path:
    """Persist posterior probabilities for ``symbol`` to QuantConnect format."""

    _require_pandas()
    ordered_columns = ("time", "prob_bullish", "prob_bearish", "prob_consolidation")
    prepared = _prepare_frame(frame, ordered_columns, ordered_columns)
    path = hmm_probability_export_path(symbol)
    exported = _export_csv(prepared, path)
    log_phase_checkpoint(
        LOGGER,
        f"Exported {len(prepared)} probability rows for {symbol} to custom data store",
    )
    return exported


def export_liquidity_timeseries(symbol: str, frame: "pd.DataFrame") -> Path:
    """Persist liquidity metrics for ``symbol`` to QuantConnect format."""

    _require_pandas()
    ordered_columns = (
        "time",
        "best_bid",
        "best_ask",
        "bid_ask_spread",
        "depth_pct_1",
        "depth_pct_5",
    )
    prepared = _prepare_frame(frame, ordered_columns, ordered_columns)
    return _export_csv(prepared, liquidity_export_path(symbol))


def export_sentiment_timeseries(symbol: str, frame: "pd.DataFrame") -> Path:
    """Persist sentiment metrics for ``symbol`` to QuantConnect format."""

    _require_pandas()
    ordered_columns = ("time", "fear_greed_score", "confidence")
    prepared = _prepare_frame(frame, ordered_columns, ordered_columns)
    return _export_csv(prepared, sentiment_export_path(symbol))
