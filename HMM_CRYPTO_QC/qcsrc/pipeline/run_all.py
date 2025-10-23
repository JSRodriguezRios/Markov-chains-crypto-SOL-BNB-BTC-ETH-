"""Orchestrate external data ingestion for all assets."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Optional

import requests

from ..util.config_loader import load_assets_config
from ..util.logging_utils import create_logger, log_phase_checkpoint
from .fetch_binance_orderbook import fetch_binance_orderbook
from .fetch_coinstats_sentiment import fetch_coinstats_sentiment
from .fetch_cryptoquant import fetch_cryptoquant

LOGGER = create_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external data ingestion pipeline")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours")
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.utcnow().strftime("%Y-%m-%dT%H:00:00"),
        help="End timestamp in ISO format (UTC)",
    )
    return parser.parse_args()


def _determine_window(end_iso: str, hours: int) -> tuple[datetime, datetime]:
    end_time = datetime.fromisoformat(end_iso)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=None)
    start_time = end_time - timedelta(hours=hours)
    return start_time, end_time


def run_pipeline(start: datetime, end: datetime, session: Optional[requests.Session] = None) -> None:
    assets = load_assets_config()
    log_phase_checkpoint(LOGGER, f"Starting ingestion for window {start} -> {end}")
    client = session or requests.Session()

    for qc_symbol in assets.keys():
        try:
            fetch_cryptoquant(qc_symbol, start, end, session=client)
            fetch_binance_orderbook(qc_symbol, start, end, session=client)
            fetch_coinstats_sentiment(qc_symbol, start, end, session=client)
        except Exception as exc:  # pragma: no cover - orchestrator logging
            LOGGER.exception("Failed to ingest data for %s: %s", qc_symbol, exc)
            continue

    log_phase_checkpoint(LOGGER, "Completed ingestion run")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    args = _parse_args()
    start_dt, end_dt = _determine_window(args.end, args.hours)
    run_pipeline(start_dt, end_dt)
