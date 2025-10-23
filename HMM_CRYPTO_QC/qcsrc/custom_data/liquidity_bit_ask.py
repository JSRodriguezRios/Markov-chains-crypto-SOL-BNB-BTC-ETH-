"""Custom data definition for liquidity snapshots."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ._base import is_header, next_hour, parse_timestamp
from ._compat import FileFormat, PythonData, SubscriptionDataSource, SubscriptionTransportMedium
from ..io.file_locator import liquidity_export_path


class LiquidityBitAsk(PythonData):
    """QuantConnect ``PythonData`` class for hourly liquidity features."""

    columns = ("time", "best_bid", "best_ask", "bid_ask_spread", "depth_pct_1", "depth_pct_5")

    def __init__(self) -> None:
        super().__init__()
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.bid_ask_spread: float = 0.0
        self.depth_pct_1: float = 0.0
        self.depth_pct_5: float = 0.0

    def GetSource(self, config, date: datetime, isLiveMode: bool) -> SubscriptionDataSource:  # noqa: N802
        symbol = _resolve_symbol(config)
        path = liquidity_export_path(symbol)
        return SubscriptionDataSource(
            str(path),
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv,
        )

    def Reader(self, config, line: str, date: datetime, isLiveMode: bool) -> Optional["LiquidityBitAsk"]:  # noqa: N802
        if not line:
            return None

        tokens = line.split(",")
        if is_header(tokens):
            return None

        if len(tokens) < len(self.columns):
            return None

        data = LiquidityBitAsk()
        data.Symbol = getattr(config, "Symbol", None)
        data.Time = parse_timestamp(tokens[0])
        data.EndTime = next_hour(data.Time)
        data.best_bid = float(tokens[1])
        data.best_ask = float(tokens[2])
        data.bid_ask_spread = float(tokens[3])
        data.depth_pct_1 = float(tokens[4])
        data.depth_pct_5 = float(tokens[5])
        data.Value = data.best_bid
        return data


def _resolve_symbol(config) -> str:
    symbol = getattr(getattr(config, "Symbol", None), "Value", None)
    if symbol is None:
        return "UNKNOWN"
    return str(symbol).upper()


__all__ = ["LiquidityBitAsk"]
