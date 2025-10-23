from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

try:  # pragma: no cover - guard against minimal environments without pandas
    import pandas as pd  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - skip entire module if pandas unavailable
    pytest.skip("pandas is required for pipeline fetcher tests", allow_module_level=True)

import responses

from qcsrc.pipeline.fetch_binance_orderbook import fetch_binance_orderbook
from qcsrc.pipeline.fetch_coinstats_sentiment import fetch_coinstats_sentiment
from qcsrc.pipeline.fetch_cryptoquant import fetch_cryptoquant


@pytest.fixture(autouse=True)
def _clear_directories():
    yield
    for subdir in [
        PROJECT_ROOT / "data" / "external" / "cryptoquant",
        PROJECT_ROOT / "data" / "external" / "binance_orderbook",
        PROJECT_ROOT / "data" / "external" / "coinstats_sentiment",
    ]:
        for csv_file in subdir.glob("**/*.csv"):
            csv_file.unlink()


@responses.activate
def test_fetch_cryptoquant(monkeypatch):
    monkeypatch.setenv("CRYPTOQUANT_API_KEY", "dummy")
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 1, 3)
    url = "https://api.cryptoquant.com/v1/btc/market-data/price-ohlcv"
    responses.add(
        responses.GET,
        url,
        json={
            "data": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "open": "100",
                    "high": "110",
                    "low": "90",
                    "close": "105",
                    "volume": "12",
                },
                {
                    "timestamp": "2024-01-01T01:00:00Z",
                    "open": "105",
                    "high": "115",
                    "low": "95",
                    "close": "110",
                    "volume": "15",
                },
            ]
        },
        status=200,
    )

    frame = fetch_cryptoquant("BTCUSD", start, end)
    assert not frame.empty
    assert {"open", "quote_volume"}.issubset(frame.columns)
    output_dir = PROJECT_ROOT / "data" / "external" / "cryptoquant" / "BTCUSD"
    assert any(output_dir.glob("*.csv"))


@responses.activate
def test_fetch_binance_orderbook():
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 1, 2)
    payload = {
        "bids": [["100", "1.5"], ["99", "2"]],
        "asks": [["101", "1.0"], ["102", "1.8"]],
    }
    responses.add(
        responses.GET,
        "https://api3.binance.com/api/v3/depth",
        json=payload,
        status=200,
    )
    responses.add(
        responses.GET,
        "https://api3.binance.com/api/v3/depth",
        json=payload,
        status=200,
    )

    frame = fetch_binance_orderbook("BTCUSD", start, end)
    assert not frame.empty
    assert "bid_ask_spread" in frame.columns
    output_dir = PROJECT_ROOT / "data" / "external" / "binance_orderbook" / "BTCUSD"
    assert any(output_dir.glob("*.csv"))


@responses.activate
def test_fetch_coinstats_sentiment(monkeypatch):
    monkeypatch.setenv("COINSTATS_API_KEY", "dummy")
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 1, 1)
    responses.add(
        responses.GET,
        "https://openapiv1.coinstats.app/insights/fear-and-greed",
        json={
            "data": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "score": "45",
                    "confidence": "0.6",
                },
                {
                    "timestamp": "2024-01-01T01:00:00Z",
                    "score": "55",
                    "confidence": "0.7",
                },
            ]
        },
        status=200,
    )

    frame = fetch_coinstats_sentiment("BTCUSD", start, end)
    assert not frame.empty
    assert {"fear_greed_score", "confidence"}.issubset(frame.columns)
    output_dir = PROJECT_ROOT / "data" / "external" / "coinstats_sentiment" / "BTCUSD"
    assert any(output_dir.glob("*.csv"))
