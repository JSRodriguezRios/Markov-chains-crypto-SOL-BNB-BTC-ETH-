"""Custom data definition for market sentiment signals."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ._base import is_header, next_hour, parse_timestamp
from ._compat import FileFormat, PythonData, SubscriptionDataSource, SubscriptionTransportMedium
from ..io.file_locator import sentiment_export_path


class MarketSentiment(PythonData):
    """QuantConnect ``PythonData`` class for CoinStats sentiment data."""

    columns = ("time", "fear_greed_score", "confidence", "z_score", "lag_1", "lag_3", "lag_6")

    def __init__(self) -> None:
        super().__init__()
        self.fear_greed_score: float = 0.0
        self.confidence: float = 0.0
        self.z_score: float = 0.0
        self.lag_1: float = 0.0
        self.lag_3: float = 0.0
        self.lag_6: float = 0.0

    def GetSource(self, config, date: datetime, isLiveMode: bool) -> SubscriptionDataSource:  # noqa: N802
        symbol = _resolve_symbol(config)
        path = sentiment_export_path(symbol)
        return SubscriptionDataSource(
            str(path),
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv,
        )

    def Reader(self, config, line: str, date: datetime, isLiveMode: bool) -> Optional["MarketSentiment"]:  # noqa: N802
        if not line:
            return None

        tokens = line.split(",")
        if is_header(tokens):
            return None

        if len(tokens) < 3:
            return None

        data = MarketSentiment()
        data.Symbol = getattr(config, "Symbol", None)
        data.Time = parse_timestamp(tokens[0])
        data.EndTime = next_hour(data.Time)
        data.fear_greed_score = float(tokens[1])
        data.confidence = float(tokens[2])
        data.z_score = float(tokens[3]) if len(tokens) > 3 else 0.0
        data.lag_1 = float(tokens[4]) if len(tokens) > 4 else 0.0
        data.lag_3 = float(tokens[5]) if len(tokens) > 5 else 0.0
        data.lag_6 = float(tokens[6]) if len(tokens) > 6 else 0.0
        data.Value = data.fear_greed_score
        return data


def _resolve_symbol(config) -> str:
    symbol = getattr(getattr(config, "Symbol", None), "Value", None)
    if symbol is None:
        return "UNKNOWN"
    return str(symbol).upper()


__all__ = ["MarketSentiment"]
