"""Performance analytics helpers for backtest evaluation."""

from .performance import (
    extract_statistics,
    group_trades_by_symbol,
    persist_metrics_report,
    summarize_closed_trades,
)

__all__ = [
    "extract_statistics",
    "group_trades_by_symbol",
    "persist_metrics_report",
    "summarize_closed_trades",
]
