"""Tests for QuantConnect custom data classes."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from qcsrc.custom_data import HMMStateProba, LiquidityBitAsk, MarketSentiment


@pytest.fixture(name="config")
def fixture_config() -> SimpleNamespace:
    symbol = SimpleNamespace(Value="BTCUSD")
    return SimpleNamespace(Symbol=symbol)


def test_liquidity_reader_parses_line(config: SimpleNamespace) -> None:
    line = "2024-02-01 00:00:00,100.0,101.0,1.0,1500.0,4500.0"
    parser = LiquidityBitAsk()

    parsed = parser.Reader(config, line, datetime.now(timezone.utc), False)

    assert parsed is not None
    assert parsed.best_bid == pytest.approx(100.0)
    assert parsed.best_ask == pytest.approx(101.0)
    assert parsed.bid_ask_spread == pytest.approx(1.0)
    assert parsed.Time.tzinfo is timezone.utc


def test_sentiment_reader_parses_optional_columns(config: SimpleNamespace) -> None:
    line = "2024-03-01 00:00:00,45.0,0.7"
    parser = MarketSentiment()

    parsed = parser.Reader(config, line, datetime.now(timezone.utc), False)

    assert parsed is not None
    assert parsed.fear_greed_score == pytest.approx(45.0)
    assert parsed.confidence == pytest.approx(0.7)
    assert parsed.z_score == pytest.approx(0.0)


def test_probability_signal_thresholds(config: SimpleNamespace) -> None:
    line = "2024-04-01 00:00:00,0.75,0.1,0.15"
    parser = HMMStateProba()

    parsed = parser.Reader(config, line, datetime.now(timezone.utc), False)
    assert parsed is not None
    assert parsed.GetSignal(0.7) == "bullish"

    neutral_line = "2024-04-01 01:00:00,0.4,0.3,0.3"
    neutral = parser.Reader(config, neutral_line, datetime.now(timezone.utc), False)
    assert neutral is not None
    assert neutral.GetSignal(0.7) == "neutral"


def test_probability_reader_skips_headers(config: SimpleNamespace) -> None:
    parser = HMMStateProba()
    header = "time,prob_bullish,prob_bearish,prob_consolidation"

    result = parser.Reader(config, header, datetime.now(timezone.utc), False)

    assert result is None
