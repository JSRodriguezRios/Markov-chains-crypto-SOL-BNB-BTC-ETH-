"""Compatibility shims for QuantConnect imports during local testing."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Deque, Iterable, Optional

try:  # pragma: no cover - use real QC classes when available
    from AlgorithmImports import *  # type: ignore  # noqa: F401,F403
except ImportError:  # pragma: no cover - fallback minimal stubs

    class Resolution:
        Minute = "Minute"
        Hour = "Hour"

    @dataclass
    class Symbol:
        Value: str

    class TradeBar:
        def __init__(
            self,
            time: datetime,
            open_price: float,
            high: float,
            low: float,
            close: float,
            volume: float,
        ) -> None:
            self.Time = time
            self.EndTime = time + timedelta(hours=1)
            self.Open = open_price
            self.High = high
            self.Low = low
            self.Close = close
            self.Volume = volume

    class EventHook:
        def __init__(self) -> None:
            self._handlers: list[Callable[[Any, Any], None]] = []

        def __iadd__(self, handler: Callable[[Any, Any], None]):
            self._handlers.append(handler)
            return self

        def emit(self, sender: Any, data: Any) -> None:
            for handler in list(self._handlers):
                handler(sender, data)

        def __call__(self, sender: Any, data: Any) -> None:
            self.emit(sender, data)

    class TradeBarConsolidator:
        def __init__(self, period: timedelta) -> None:
            self.period = period
            self.DataConsolidated = EventHook()

        def Update(self, bar: TradeBar) -> None:  # noqa: N802 - QC naming
            self.DataConsolidated(self, bar)

    class PortfolioHolding:
        def __init__(self) -> None:
            self.Quantity = 0.0
            self.AveragePrice = 0.0

    class Portfolio:
        def __init__(self) -> None:
            self.Cash = 0.0
            self.total_holdings: dict[str, PortfolioHolding] = {}

        def __getitem__(self, symbol: Symbol) -> PortfolioHolding:
            if symbol.Value not in self.total_holdings:
                self.total_holdings[symbol.Value] = PortfolioHolding()
            return self.total_holdings[symbol.Value]

    class SubscriptionManager:
        def AddConsolidator(self, symbol: Symbol, consolidator: TradeBarConsolidator) -> None:  # noqa: N802
            return None

    class QCAlgorithm:
        def __init__(self) -> None:
            self.Portfolio = Portfolio()
            self.SubscriptionManager = SubscriptionManager()

        def SetStartDate(self, year: int, month: int, day: int) -> None:
            return None

        def SetEndDate(self, year: int, month: int, day: int) -> None:
            return None

        def SetCash(self, amount: float) -> None:
            self.Portfolio.Cash = amount

        def SetBrokerageModel(self, model: Any = None) -> None:
            return None

        def SetWarmUp(self, period: timedelta) -> None:
            return None

        def AddCrypto(self, ticker: str, resolution: str = Resolution.Minute):
            return SimpleNamespace(Symbol=Symbol(ticker))

        def AddData(self, data_type: Any, ticker: str, resolution: str = Resolution.Hour):
            return SimpleNamespace(Symbol=Symbol(ticker))

        def Debug(self, message: str) -> None:
            print(f"DEBUG: {message}")

        def Log(self, message: str) -> None:
            print(f"LOG: {message}")

        def SetHoldings(self, symbol: Symbol, weight: float) -> None:
            holding = self.Portfolio[symbol]
            holding.Quantity = weight

        def Liquidate(self, symbol: Optional[Symbol] = None) -> None:
            if symbol is None:
                return
            holding = self.Portfolio[symbol]
            holding.Quantity = 0.0

        class ScheduleManager:
            def __init__(self) -> None:
                self.events: list[Any] = []

            def On(self, *args: Any, **kwargs: Any) -> None:
                self.events.append((args, kwargs))

        @property
        def Schedule(self) -> "QCAlgorithm.ScheduleManager":
            return QCAlgorithm.ScheduleManager()

    class RollingWindow(Deque[Any]):
        def __init__(self, size: int) -> None:
            super().__init__(maxlen=size)

    class PortfolioTarget:
        def __init__(self, symbol: Symbol, quantity: float) -> None:
            self.Symbol = symbol
            self.Quantity = quantity

    class Slice:
        def __init__(self, data: Optional[dict[str, Any]] = None) -> None:
            self._data = data or {}

        def Get(self, data_type: Any):  # noqa: N802 - QC naming
            return self._data.get(data_type, {})


__all__ = [name for name in globals() if not name.startswith("_")]
