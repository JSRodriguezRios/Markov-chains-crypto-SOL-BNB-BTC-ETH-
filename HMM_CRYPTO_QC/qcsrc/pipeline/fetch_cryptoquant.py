"""Fetch hourly OHLCV data from CryptoQuant."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

from ..io import schemas
from ..util.config_loader import load_assets_config
from ..util.logging_utils import create_logger, log_phase_checkpoint
from ..util.secrets import get_api_key
from .http import request_with_backoff

LOGGER = create_logger(__name__)
CRYPTOQUANT_BASE_URL = "https://api.cryptoquant.com/v1"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_dir(qc_symbol: str) -> Path:
    directory = _project_root() / "data" / "external" / "cryptoquant" / qc_symbol
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _coin_from_symbol(cryptoquant_symbol: str) -> str:
    return cryptoquant_symbol.split("_")[0]


def _normalize_payload(records: Iterable[Dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return frame
    frame.rename(columns={"timestamp": "timestamp"}, inplace=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "quote_volume" not in frame.columns:
        frame["quote_volume"] = frame["close"] * frame["volume"]
    else:
        frame["quote_volume"] = pd.to_numeric(frame["quote_volume"], errors="coerce")
    return frame


def _persist_daily_csv(df: pd.DataFrame, qc_symbol: str) -> None:
    if df.empty:
        LOGGER.warning("CryptoQuant returned no data for %s; skipping export.", qc_symbol)
        return
    export_dir = _output_dir(qc_symbol)
    grouped = df.groupby(df["timestamp"].dt.date)
    for date_value, group in grouped:
        output_path = export_dir / f"{date_value}.csv"
        group.sort_values("timestamp").to_csv(output_path, index=False)
        LOGGER.info("Saved CryptoQuant data for %s to %s", qc_symbol, output_path)


def fetch_cryptoquant(
    qc_symbol: str,
    start: datetime,
    end: datetime,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch hourly OHLCV data for ``qc_symbol`` between ``start`` and ``end``."""
    if start >= end:
        raise ValueError("start must be earlier than end")

    assets = load_assets_config()
    asset_config = assets.get(qc_symbol)
    if not asset_config:
        raise KeyError(f"Symbol {qc_symbol} is not configured in assets.yaml")

    cryptoquant_symbol = asset_config["cryptoquant_symbol"]
    coin = _coin_from_symbol(cryptoquant_symbol)
    api_key = get_api_key("cryptoquant")

    params = {
        "window": "hour",
        "market": "spot",
        "exchange": "all_exchange",
        "symbol": cryptoquant_symbol,
        "from": int(start.timestamp()),
        "to": int(end.timestamp()),
    }
    url = f"{CRYPTOQUANT_BASE_URL}/{coin}/market-data/price-ohlcv"

    client = session or requests.Session()
    headers = {"x-api-key": api_key}
    response = request_with_backoff(client, url, params=params, headers=headers, timeout=30)
    if response.status_code >= 400:
        LOGGER.error("CryptoQuant error %s: %s", response.status_code, response.text)
        response.raise_for_status()

    payload = response.json().get("data", [])
    frame = _normalize_payload(payload)
    frame = frame[(frame["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (frame["timestamp"] <= pd.Timestamp(end, tz="UTC"))]
    if frame.empty:
        LOGGER.warning("CryptoQuant payload did not include entries within range for %s", qc_symbol)
        return frame
    schemas.validate_cryptoquant(frame)
    _persist_daily_csv(frame, qc_symbol)
    log_phase_checkpoint(LOGGER, f"Fetched CryptoQuant data for {qc_symbol} from {start} to {end}")
    return frame
