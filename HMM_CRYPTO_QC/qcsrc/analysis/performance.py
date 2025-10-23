"""Utilities for summarising QuantConnect backtest performance."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from ..io.file_locator import metrics_output_path
from ..util.logging_utils import create_logger, log_phase_checkpoint

LOGGER = create_logger(__name__)


@dataclass(frozen=True)
class TradeSnapshot:
    """Simplified view of a QuantConnect closed trade."""

    symbol: str
    profit_loss: float
    fees: float

    @property
    def net_profit(self) -> float:
        return self.profit_loss - self.fees


def _coerce_trade(trade: Any) -> TradeSnapshot:
    symbol = getattr(getattr(trade, "Symbol", None), "Value", None) or getattr(trade, "symbol", "UNKNOWN")
    profit_loss = float(getattr(trade, "ProfitLoss", getattr(trade, "profit_loss", 0.0)) or 0.0)
    fees = float(getattr(trade, "TotalFees", getattr(trade, "fees", 0.0)) or 0.0)
    return TradeSnapshot(symbol=symbol, profit_loss=profit_loss, fees=fees)


def group_trades_by_symbol(closed_trades: Iterable[Any]) -> Dict[str, List[TradeSnapshot]]:
    """Group closed trades by their symbol."""

    grouped: MutableMapping[str, List[TradeSnapshot]] = defaultdict(list)
    for trade in closed_trades or []:
        snapshot = _coerce_trade(trade)
        grouped[snapshot.symbol].append(snapshot)
    return dict(grouped)


def summarize_closed_trades(trades: Iterable[Any]) -> Dict[str, float]:
    """Compute win/loss statistics from closed trades."""

    snapshots = [_coerce_trade(trade) for trade in trades]
    total_trades = len(snapshots)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "net_profit": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
        }

    net_profits = [trade.net_profit for trade in snapshots]
    wins = [value for value in net_profits if value > 0]
    losses = [value for value in net_profits if value <= 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades
    loss_rate = loss_count / total_trades
    average_win = sum(wins) / win_count if win_count else 0.0
    average_loss = sum(losses) / loss_count if loss_count else 0.0

    return {
        "total_trades": float(total_trades),
        "wins": float(win_count),
        "losses": float(loss_count),
        "win_rate": float(win_rate),
        "loss_rate": float(loss_rate),
        "net_profit": float(sum(net_profits)),
        "average_win": float(average_win),
        "average_loss": float(average_loss),
    }


def _parse_percentage(value: str) -> Optional[float]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.endswith("%"):
        try:
            return float(stripped[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(stripped)
    except ValueError:
        return None


def extract_statistics(statistics: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Extract Sharpe, Sortino, drawdown, win/loss rates, and trades from QC statistics."""

    if statistics is None:
        statistics = {}

    sharpe = statistics.get("Sharpe Ratio")
    sortino = statistics.get("Sortino Ratio")
    drawdown = statistics.get("Drawdown")
    win_rate = statistics.get("Win Rate")
    loss_rate = statistics.get("Loss Rate")
    total_trades = statistics.get("Total Trades")

    return {
        "sharpe_ratio": float(sharpe) if _parse_percentage(str(sharpe)) is not None else None,
        "sortino_ratio": float(sortino) if _parse_percentage(str(sortino)) is not None else None,
        "drawdown": _parse_percentage(str(drawdown)),
        "win_rate": _parse_percentage(str(win_rate)),
        "loss_rate": _parse_percentage(str(loss_rate)),
        "total_trades": float(total_trades) if str(total_trades).strip() else None,
    }


def persist_metrics_report(symbol: str, report: Mapping[str, Any]) -> Path:
    """Persist a JSON metrics report for ``symbol`` to disk."""

    log_phase_checkpoint(LOGGER, f"Persisting metrics report for {symbol}")
    path = metrics_output_path(symbol)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return path


__all__ = [
    "TradeSnapshot",
    "extract_statistics",
    "group_trades_by_symbol",
    "persist_metrics_report",
    "summarize_closed_trades",
]
