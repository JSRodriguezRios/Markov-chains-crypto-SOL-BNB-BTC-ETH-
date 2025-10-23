"""Compatibility helpers for QuantConnect imports during local testing."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

try:  # pragma: no cover - executed in QuantConnect environment
    from AlgorithmImports import (  # type: ignore
        FileFormat,
        PythonData,
        SubscriptionDataSource,
        SubscriptionTransportMedium,
    )
except ImportError:  # pragma: no cover - fallback for local testing

    class SubscriptionTransportMedium(Enum):
        """Minimal transport medium enum matching QuantConnect expectations."""

        LocalFile = "LocalFile"

    class FileFormat(Enum):
        """Minimal file format enum."""

        Csv = "csv"

    class SubscriptionDataSource:
        """Lightweight stand-in for QuantConnect's SubscriptionDataSource."""

        def __init__(self, source: str, medium: SubscriptionTransportMedium, format: FileFormat) -> None:
            self.Source = source
            self.TransportMedium = medium
            self.DataFormat = format

    class PythonData:
        """Baseline PythonData substitute for tests without AlgorithmImports."""

        def __init__(self) -> None:
            self.Symbol: Any = None
            self.Value: float = 0.0
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            self.Time: datetime = epoch
            self.EndTime: datetime = epoch

__all__ = [
    "FileFormat",
    "PythonData",
    "SubscriptionDataSource",
    "SubscriptionTransportMedium",
]
