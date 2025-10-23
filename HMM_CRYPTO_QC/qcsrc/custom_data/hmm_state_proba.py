"""Custom data definition for HMM posterior probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ._base import is_header, next_hour, parse_timestamp
from ._compat import FileFormat, PythonData, SubscriptionDataSource, SubscriptionTransportMedium
from ..io.file_locator import hmm_probability_export_path


@dataclass(frozen=True)
class HMMThresholds:
    """Threshold defaults used for live signal extraction."""

    bullish: float = 0.7
    bearish: float = 0.7
    consolidation: float = 0.7


class HMMStateProba(PythonData):
    """QuantConnect ``PythonData`` class for HMM regime probabilities."""

    columns = ("time", "prob_bullish", "prob_bearish", "prob_consolidation")

    def __init__(self) -> None:
        super().__init__()
        self.prob_bullish: float = 0.0
        self.prob_bearish: float = 0.0
        self.prob_consolidation: float = 0.0

    def GetSource(self, config, date: datetime, isLiveMode: bool) -> SubscriptionDataSource:  # noqa: N802
        symbol = _resolve_symbol(config)
        path = hmm_probability_export_path(symbol)
        return SubscriptionDataSource(
            str(path),
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv,
        )

    def Reader(self, config, line: str, date: datetime, isLiveMode: bool) -> Optional["HMMStateProba"]:  # noqa: N802
        if not line:
            return None

        tokens = line.split(",")
        if is_header(tokens):
            return None

        if len(tokens) < len(self.columns):
            return None

        data = HMMStateProba()
        data.Symbol = getattr(config, "Symbol", None)
        data.Time = parse_timestamp(tokens[0])
        data.EndTime = next_hour(data.Time)
        data.prob_bullish = float(tokens[1])
        data.prob_bearish = float(tokens[2])
        data.prob_consolidation = float(tokens[3])
        data.Value = max(data.prob_bullish, data.prob_bearish, data.prob_consolidation)
        return data

    def GetSignal(self, threshold: float = 0.7) -> str:  # noqa: N802
        """Return the trading action associated with stored probabilities."""

        if self.prob_bullish >= threshold:
            return "bullish"
        if self.prob_bearish >= threshold:
            return "bearish"
        if self.prob_consolidation >= threshold:
            return "consolidation"
        return "neutral"


def _resolve_symbol(config) -> str:
    symbol = getattr(getattr(config, "Symbol", None), "Value", None)
    if symbol is None:
        return "UNKNOWN"
    return str(symbol).upper()


__all__ = ["HMMStateProba", "HMMThresholds"]
