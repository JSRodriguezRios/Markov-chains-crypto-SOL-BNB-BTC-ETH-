from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

try:  # pragma: no cover - guard minimal environments
    import pandas as pd  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("pandas is required for alignment tests", allow_module_level=True)

from qcsrc.pipeline.align_merge import AlignmentError, align_and_merge
from qcsrc.io.file_locator import external_source_directory, interim_output_path


def _write_sample(directory: Path, filename: str, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_csv(directory / filename, index=False)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    interim_file = interim_output_path("BTCUSD")
    if interim_file.exists():
        interim_file.unlink()
    csv_fallback = interim_file.with_suffix(".csv")
    if csv_fallback.exists():
        csv_fallback.unlink()
    for source in ["cryptoquant", "binance_orderbook", "coinstats_sentiment"]:
        directory = external_source_directory(source, "BTCUSD")
        for file_path in directory.glob("*.csv"):
            file_path.unlink()


def test_align_and_merge_success():
    start = datetime(2024, 1, 1)
    rows = [
        {"timestamp": "2024-01-01T00:00:00Z", "open": 100, "high": 110, "low": 95, "close": 105, "volume": 12, "quote_volume": 1260},
        {"timestamp": "2024-01-01T01:00:00Z", "open": 105, "high": 112, "low": 98, "close": 109, "volume": 14, "quote_volume": 1526},
    ]
    _write_sample(external_source_directory("cryptoquant", "BTCUSD"), "2024-01-01.csv", rows)

    liquidity_rows = [
        {"timestamp": "2024-01-01T00:00:00Z", "best_bid": 104, "best_ask": 106, "bid_ask_spread": 2, "depth_pct_1": 1200, "depth_pct_5": 3400},
        {"timestamp": "2024-01-01T01:00:00Z", "best_bid": 108, "best_ask": 110, "bid_ask_spread": 2, "depth_pct_1": 1500, "depth_pct_5": 3800},
    ]
    _write_sample(external_source_directory("binance_orderbook", "BTCUSD"), "2024-01-01.csv", liquidity_rows)

    sentiment_rows = [
        {"timestamp": "2024-01-01T00:00:00Z", "fear_greed_score": 45, "confidence": 0.6},
        {"timestamp": "2024-01-01T01:00:00Z", "fear_greed_score": 55, "confidence": 0.7},
    ]
    _write_sample(external_source_directory("coinstats_sentiment", "BTCUSD"), "2024-01-01.csv", sentiment_rows)

    frame = align_and_merge("BTCUSD", start)
    assert len(frame) == 2
    assert set(frame.columns) == {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "best_bid",
        "best_ask",
        "bid_ask_spread",
        "depth_pct_1",
        "depth_pct_5",
        "fear_greed_score",
        "confidence",
    }
    assert frame["timestamp"].dt.tz is not None

    interim_path = interim_output_path("BTCUSD")
    fallback_path = interim_path.with_suffix(".csv")
    assert interim_path.exists() or fallback_path.exists()


def test_align_and_merge_missing_ohlcv():
    start = datetime(2024, 1, 1)
    with pytest.raises(AlignmentError):
        align_and_merge("BTCUSD", start)
