"""Align external data sources into a unified hourly frame."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

from ..io import schemas
from ..io.file_locator import external_source_directory, interim_output_path
from ..util.config_loader import load_assets_config
from ..util.logging_utils import create_logger, log_phase_checkpoint
from ..util.timezones import ensure_utc, floor_to_hour, hourly_range

LOGGER = create_logger(__name__)


class AlignmentError(RuntimeError):
    """Raised when alignment prerequisites are not satisfied."""


def _load_csv_group(directory: Path, start: datetime, end: datetime) -> pd.DataFrame:
    """Read all CSV files under ``directory`` and filter to [start, end)."""

    frames: List[pd.DataFrame] = []
    for csv_path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        if "timestamp" not in frame.columns:
            LOGGER.debug("Skipping %s because it lacks a timestamp column", csv_path)
            continue
        frame["timestamp"] = ensure_utc(frame["timestamp"])
        mask = (frame["timestamp"] >= start) & (frame["timestamp"] < end)
        filtered = frame.loc[mask]
        if not filtered.empty:
            frames.append(filtered)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _prepare_ohlcv(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        raise AlignmentError("OHLCV frame is empty; cannot align data.")

    schemas.validate_cryptoquant(frame)
    ohlcv = frame.sort_values("timestamp").drop_duplicates("timestamp")
    ohlcv = ohlcv.set_index("timestamp").reindex(index)
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    ohlcv[numeric_cols] = ohlcv[numeric_cols].apply(pd.to_numeric, errors="coerce")
    ohlcv[numeric_cols] = ohlcv[numeric_cols].fillna(method="ffill", limit=1)
    ohlcv = ohlcv.dropna(subset=["close"])
    return ohlcv


def _prepare_liquidity(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        LOGGER.warning("Liquidity frame is empty; filling with zeros.")
        columns = ["best_bid", "best_ask", "bid_ask_spread", "depth_pct_1", "depth_pct_5"]
        return pd.DataFrame(0.0, index=index, columns=columns)

    schemas.validate_binance_orderbook(frame)
    liquidity = frame.sort_values("timestamp").drop_duplicates("timestamp")
    liquidity = liquidity.set_index("timestamp").reindex(index)
    numeric_cols = ["best_bid", "best_ask", "bid_ask_spread", "depth_pct_1", "depth_pct_5"]
    for column in numeric_cols:
        liquidity[column] = pd.to_numeric(liquidity[column], errors="coerce")
    liquidity[["depth_pct_1", "depth_pct_5"]] = liquidity[["depth_pct_1", "depth_pct_5"]].fillna(0.0)
    liquidity[["best_bid", "best_ask", "bid_ask_spread"]] = liquidity[["best_bid", "best_ask", "bid_ask_spread"]].fillna(method="ffill")
    liquidity.fillna(0.0, inplace=True)
    return liquidity


def _prepare_sentiment(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        LOGGER.warning("Sentiment frame is empty; forward-filling zeros.")
        columns = ["fear_greed_score", "confidence"]
        return pd.DataFrame(0.0, index=index, columns=columns)

    schemas.validate_coinstats(frame)
    sentiment = frame.sort_values("timestamp").drop_duplicates("timestamp")
    sentiment = sentiment.set_index("timestamp").reindex(index)
    for column in ["fear_greed_score", "confidence"]:
        sentiment[column] = pd.to_numeric(sentiment[column], errors="coerce")
    sentiment.fillna(method="ffill", inplace=True)
    sentiment.fillna(0.0, inplace=True)
    return sentiment


def _persist_interim(symbol: str, df: pd.DataFrame) -> Path:
    output_path = interim_output_path(symbol)
    try:
        df.to_parquet(output_path, index=True)
    except (ImportError, ValueError) as exc:
        fallback_path = output_path.with_suffix(".csv")
        LOGGER.warning("%s while writing parquet; falling back to CSV at %s", exc, fallback_path)
        df.to_csv(fallback_path)
        return fallback_path
    return output_path


def align_and_merge(symbol: str, date: datetime) -> pd.DataFrame:
    """Align external datasets for ``symbol`` on ``date`` (UTC day granularity)."""

    assets = load_assets_config()
    if symbol not in assets:
        raise KeyError(f"Symbol {symbol} is not configured in assets.yaml")

    start = floor_to_hour(date)
    end = start + timedelta(days=1)
    index = hourly_range(start, end)

    cryptoquant_dir = external_source_directory("cryptoquant", symbol)
    binance_dir = external_source_directory("binance_orderbook", symbol)
    coinstats_dir = external_source_directory("coinstats_sentiment", symbol)

    ohlcv_raw = _load_csv_group(cryptoquant_dir, start, end)
    liquidity_raw = _load_csv_group(binance_dir, start, end)
    sentiment_raw = _load_csv_group(coinstats_dir, start, end)

    ohlcv = _prepare_ohlcv(ohlcv_raw, index)
    if ohlcv.empty:
        raise AlignmentError("Aligned OHLCV frame is empty after preprocessing.")

    liquidity = _prepare_liquidity(liquidity_raw, ohlcv.index)
    sentiment = _prepare_sentiment(sentiment_raw, ohlcv.index)

    combined = pd.concat([ohlcv, liquidity, sentiment], axis=1, join="inner").sort_index()
    combined.index.name = "timestamp"
    schemas.validate_interim(combined.reset_index())

    output_path = _persist_interim(symbol, combined)
    log_phase_checkpoint(LOGGER, f"Aligned data for {symbol} on {start.date()} -> {output_path}")
    return combined.reset_index()
