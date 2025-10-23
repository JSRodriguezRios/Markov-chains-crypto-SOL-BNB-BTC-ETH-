"""Fetch order book liquidity metrics from Binance."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ..io import schemas
from ..util.config_loader import load_assets_config
from ..util.logging_utils import create_logger, log_phase_checkpoint
from .http import request_with_backoff

LOGGER = create_logger(__name__)
BINANCE_BASE_URL = "https://api3.binance.com/api/v3/depth"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_dir(qc_symbol: str) -> Path:
    directory = _project_root() / "data" / "external" / "binance_orderbook" / qc_symbol
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _orders_to_frame(orders: list) -> pd.DataFrame:
    frame = pd.DataFrame(orders, columns=["price", "quantity"])
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame.dropna(inplace=True)
    return frame


def _calculate_depth(bids: pd.DataFrame, asks: pd.DataFrame, mid_price: float, pct: float) -> float:
    lower_bound = mid_price * (1 - pct)
    upper_bound = mid_price * (1 + pct)
    bid_depth = bids.loc[bids["price"] >= lower_bound, "quantity"].sum()
    ask_depth = asks.loc[asks["price"] <= upper_bound, "quantity"].sum()
    return float(bid_depth + ask_depth)


def _persist_daily_csv(df: pd.DataFrame, qc_symbol: str) -> None:
    if df.empty:
        LOGGER.warning("Binance returned no data for %s; skipping export.", qc_symbol)
        return
    export_dir = _output_dir(qc_symbol)
    grouped = df.groupby(df["timestamp"].dt.date)
    for date_value, group in grouped:
        output_path = export_dir / f"{date_value}.csv"
        group.sort_values("timestamp").to_csv(output_path, index=False)
        LOGGER.info("Saved Binance order book data for %s to %s", qc_symbol, output_path)


def fetch_binance_orderbook(
    qc_symbol: str,
    start: datetime,
    end: datetime,
    session: Optional[requests.Session] = None,
    depth_limit: int = 1000,
) -> pd.DataFrame:
    """Fetch hourly Binance order book snapshots."""
    if start >= end:
        raise ValueError("start must be earlier than end")

    assets = load_assets_config()
    asset_config = assets.get(qc_symbol)
    if not asset_config:
        raise KeyError(f"Symbol {qc_symbol} is not configured in assets.yaml")

    symbol = asset_config["binance_symbol"]

    timestamps = pd.date_range(start=start, end=end, freq="H", inclusive="left")
    client = session or requests.Session()
    rows = []
    for timestamp in timestamps:
        response = request_with_backoff(
            client,
            BINANCE_BASE_URL,
            params={"symbol": symbol, "limit": depth_limit},
            timeout=30,
        )
        if response.status_code >= 400:
            LOGGER.error("Binance error %s: %s", response.status_code, response.text)
            response.raise_for_status()
        payload = response.json()
        bids = _orders_to_frame(payload.get("bids", []))
        asks = _orders_to_frame(payload.get("asks", []))
        if bids.empty or asks.empty:
            LOGGER.warning("Empty order book snapshot for %s at %s", qc_symbol, timestamp)
            continue
        best_bid = bids.iloc[0]["price"]
        best_ask = asks.iloc[0]["price"]
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        depth_pct_1 = _calculate_depth(bids, asks, mid_price, 0.01)
        depth_pct_5 = _calculate_depth(bids, asks, mid_price, 0.05)
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp, tz="UTC"),
                "best_bid": float(best_bid),
                "best_ask": float(best_ask),
                "bid_ask_spread": float(spread),
                "depth_pct_1": depth_pct_1,
                "depth_pct_5": depth_pct_5,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        LOGGER.warning("No Binance order book rows produced for %s", qc_symbol)
        return frame
    schemas.validate_binance_orderbook(frame)
    _persist_daily_csv(frame, qc_symbol)
    log_phase_checkpoint(LOGGER, f"Fetched Binance order book for {qc_symbol} between {start} and {end}")
    return frame
