"""Tests for QuantConnect custom data export helpers."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from qcsrc.io import custom_export


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_export_probability_timeseries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "probabilities.csv"
    monkeypatch.setattr(
        custom_export,
        "hmm_probability_export_path",
        lambda symbol: destination,
    )

    frame = pd.DataFrame(
        {
            "time": ["2024-01-01T01:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "prob_bullish": [0.7, 0.2, 0.8],
            "prob_bearish": [0.2, 0.5, 0.1],
            "prob_consolidation": [0.1, 0.3, 0.1],
        }
    )

    path = custom_export.export_probability_timeseries("BTCUSD", frame)
    rows = _read_csv(path)

    assert rows[0] == ["time", "prob_bullish", "prob_bearish", "prob_consolidation"]
    assert rows[1][0] == "2024-01-01 00:00:00"
    # The duplicate timestamp retains the latest observation after sorting/dedup.
    assert rows[2][1:] == ["0.8", "0.1", "0.1"]


def test_export_liquidity_timeseries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "liquidity.csv"
    monkeypatch.setattr(custom_export, "liquidity_export_path", lambda symbol: destination)

    frame = pd.DataFrame(
        {
            "timestamp": ["2024-02-01T00:00:00Z", "2024-02-01T01:00:00Z"],
            "best_bid": [100.0, 101.0],
            "best_ask": [100.5, 101.4],
            "bid_ask_spread": [0.5, 0.4],
            "depth_pct_1": [1500.0, 1400.0],
            "depth_pct_5": [4500.0, 4300.0],
        }
    )

    path = custom_export.export_liquidity_timeseries("ETHUSD", frame)
    rows = _read_csv(path)

    assert rows[0] == [
        "time",
        "best_bid",
        "best_ask",
        "bid_ask_spread",
        "depth_pct_1",
        "depth_pct_5",
    ]
    assert rows[1][0] == "2024-02-01 00:00:00"
    assert rows[2][1] == "101.0"


def test_export_sentiment_timeseries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "sentiment.csv"
    monkeypatch.setattr(custom_export, "sentiment_export_path", lambda symbol: destination)

    frame = pd.DataFrame(
        {
            "timestamp": ["2024-03-01T00:00:00Z"],
            "fear_greed_score": [45.0],
            "confidence": [0.7],
        }
    )

    path = custom_export.export_sentiment_timeseries("SOLUSD", frame)
    rows = _read_csv(path)

    assert rows[0] == ["time", "fear_greed_score", "confidence"]
    assert rows[1] == ["2024-03-01 00:00:00", "45.0", "0.7"]
