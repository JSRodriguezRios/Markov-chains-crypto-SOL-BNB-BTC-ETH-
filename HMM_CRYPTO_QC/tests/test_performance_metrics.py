from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json

import pytest

from qcsrc.analysis import performance
from qcsrc.analysis.performance import (
    extract_statistics,
    group_trades_by_symbol,
    persist_metrics_report,
    summarize_closed_trades,
)
from qcsrc.analysis.performance import TradeSnapshot


class DummyTrade:
    def __init__(self, symbol: str, profit_loss: float, fees: float = 0.0) -> None:
        self.Symbol = type("Symbol", (), {"Value": symbol})()
        self.ProfitLoss = profit_loss
        self.TotalFees = fees


def test_group_trades_by_symbol_groups_expected_counts():
    trades = [DummyTrade("BTCUSD", 10), DummyTrade("BTCUSD", -5), DummyTrade("ETHUSD", 2)]
    grouped = group_trades_by_symbol(trades)
    assert set(grouped.keys()) == {"BTCUSD", "ETHUSD"}
    assert len(grouped["BTCUSD"]) == 2
    assert isinstance(grouped["BTCUSD"][0], TradeSnapshot)


def test_summarize_closed_trades_returns_win_loss_metrics():
    trades = [DummyTrade("BTCUSD", 20, 2), DummyTrade("BTCUSD", -10, 1)]
    summary = summarize_closed_trades(trades)
    assert summary["total_trades"] == 2.0
    assert summary["wins"] == 1.0
    assert summary["losses"] == 1.0
    assert summary["net_profit"] == (20 - 2) + (-10 - 1)


def test_extract_statistics_parses_percentages_and_numbers():
    stats = {
        "Sharpe Ratio": "1.2",
        "Sortino Ratio": "0.8",
        "Drawdown": "12.5%",
        "Win Rate": "55%",
        "Loss Rate": "45%",
        "Total Trades": "120",
    }
    extracted = extract_statistics(stats)
    assert extracted["sharpe_ratio"] == 1.2
    assert extracted["sortino_ratio"] == 0.8
    assert extracted["drawdown"] == 0.125
    assert extracted["win_rate"] == 0.55
    assert extracted["loss_rate"] == 0.45
    assert extracted["total_trades"] == 120.0


def test_persist_metrics_report_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    report = {"symbol": "BTCUSD", "error_metrics": {"mae": 0.1}}

    def _mock_metrics_output_path(symbol: str) -> Path:
        return tmp_path / f"{symbol}.json"

    monkeypatch.setattr(performance, "metrics_output_path", _mock_metrics_output_path)
    output_path = persist_metrics_report("BTCUSD", report)

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["symbol"] == "BTCUSD"
