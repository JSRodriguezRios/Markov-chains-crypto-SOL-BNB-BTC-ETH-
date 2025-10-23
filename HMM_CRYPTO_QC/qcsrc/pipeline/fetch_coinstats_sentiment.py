"""Fetch market sentiment data from CoinStats."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ..io import schemas
from ..util.config_loader import load_assets_config
from ..util.logging_utils import create_logger, log_phase_checkpoint
from ..util.secrets import get_api_key
from .http import request_with_backoff

LOGGER = create_logger(__name__)
COINSTATS_URL = "https://openapiv1.coinstats.app/insights/fear-and-greed"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_dir(qc_symbol: str) -> Path:
    directory = _project_root() / "data" / "external" / "coinstats_sentiment" / qc_symbol
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _persist_daily_csv(df: pd.DataFrame, qc_symbol: str) -> None:
    if df.empty:
        LOGGER.warning("CoinStats returned no data for %s; skipping export.", qc_symbol)
        return
    export_dir = _output_dir(qc_symbol)
    grouped = df.groupby(df["timestamp"].dt.date)
    for date_value, group in grouped:
        output_path = export_dir / f"{date_value}.csv"
        group.sort_values("timestamp").to_csv(output_path, index=False)
        LOGGER.info("Saved CoinStats sentiment data for %s to %s", qc_symbol, output_path)


def _interpolate_to_hourly(frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    hourly_index = pd.date_range(start=start, end=end, freq="H", tz="UTC")
    frame = frame.set_index("timestamp").reindex(hourly_index, method="ffill")
    frame = frame.reset_index().rename(columns={"index": "timestamp"})
    return frame


def fetch_coinstats_sentiment(
    qc_symbol: str,
    start: datetime,
    end: datetime,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch fear and greed sentiment series for the requested period."""
    if start >= end:
        raise ValueError("start must be earlier than end")

    assets = load_assets_config()
    asset_config = assets.get(qc_symbol)
    if not asset_config:
        raise KeyError(f"Symbol {qc_symbol} is not configured in assets.yaml")

    api_key = get_api_key("coinstats")
    params = {"symbol": asset_config["coinstats_symbol"]}
    headers = {"X-API-KEY": api_key}

    client = session or requests.Session()
    response = request_with_backoff(client, COINSTATS_URL, params=params, headers=headers, timeout=30)
    if response.status_code >= 400:
        LOGGER.error("CoinStats error %s: %s", response.status_code, response.text)
        response.raise_for_status()

    payload = response.json().get("data", [])
    frame = pd.DataFrame(payload)
    if frame.empty:
        LOGGER.warning("CoinStats returned empty payload for %s", qc_symbol)
        return frame

    frame.rename(columns={"score": "fear_greed_score"}, inplace=True)
    if "timestamp" not in frame.columns:
        frame["timestamp"] = pd.Timestamp.utcnow().tz_localize("UTC")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["fear_greed_score"] = pd.to_numeric(frame["fear_greed_score"], errors="coerce")
    if "confidence" in frame.columns:
        frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    else:
        frame["confidence"] = float("nan")

    frame = frame[(frame["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (frame["timestamp"] <= pd.Timestamp(end, tz="UTC"))]
    if frame.empty:
        LOGGER.warning("CoinStats payload did not include entries within range for %s", qc_symbol)
        return frame
    frame = _interpolate_to_hourly(frame, start, end)
    schemas.validate_coinstats(frame)
    _persist_daily_csv(frame, qc_symbol)
    log_phase_checkpoint(LOGGER, f"Fetched CoinStats sentiment for {qc_symbol} from {start} to {end}")
    return frame
