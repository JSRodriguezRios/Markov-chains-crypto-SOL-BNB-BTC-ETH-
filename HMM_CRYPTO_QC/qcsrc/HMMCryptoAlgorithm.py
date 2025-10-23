"""QuantConnect algorithm entry point for the HMM crypto strategy."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Dict, Iterable, List, Optional

from ._algorithm_compat import (  # type: ignore
    QCAlgorithm,
    Resolution,
    RollingWindow,
    Slice,
    Symbol,
    TradeBar,
    TradeBarConsolidator,
)
from .custom_data.hmm_state_proba import HMMThresholds
from .custom_data.liquidity_bit_ask import LiquidityBitAsk
from .custom_data.market_sentiment import MarketSentiment
from .io.file_locator import (
    hmm_model_output_path,
    scaler_output_path,
)
from .models.thresholds import should_flat, should_long, should_short
from .util.logging_utils import create_logger, log_phase_checkpoint

LOGGER = create_logger(__name__)
UTC = timezone.utc

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from .models.hmm_model import GaussianHMMWrapper


@dataclass
class SymbolContext:
    """Container for tracking per-symbol state within the algorithm."""

    symbol: Symbol
    lookback: int
    trade_bars: Deque[TradeBar] = field(default_factory=lambda: deque(maxlen=336))
    liquidity_points: Deque[LiquidityBitAsk] = field(default_factory=deque)
    sentiment_points: Deque[MarketSentiment] = field(default_factory=deque)
    model: Optional["GaussianHMMWrapper"] = None
    scaler: Optional[object] = None
    state_mapping: Dict[int, str] = field(default_factory=dict)
    last_signal: str = "neutral"
    probabilities: Dict[str, float] = field(
        default_factory=lambda: {"bullish": 0.0, "bearish": 0.0, "consolidation": 0.0}
    )
    predicted_returns: List[float] = field(default_factory=list)
    actual_returns: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.trade_bars = deque(maxlen=self.lookback)
        self.liquidity_points = deque(maxlen=self.lookback)
        self.sentiment_points = deque(maxlen=self.lookback)


def determine_signal(
    probabilities: Dict[str, float],
    threshold: float,
    previous_signal: str,
) -> str:
    """Return the semantic trading signal based on probability thresholds."""

    if should_long(probabilities, threshold):
        return "bullish"
    if should_short(probabilities, threshold):
        return "bearish"
    if should_flat(probabilities, threshold):
        return "consolidation"
    return previous_signal


def target_allocation(signal: str, max_allocation: float) -> float:
    """Translate a semantic signal into a portfolio weight."""

    if signal == "bullish":
        return max_allocation
    if signal == "bearish":
        return -max_allocation
    return 0.0


def compute_error_metrics(predicted: Iterable[float], actual: Iterable[float]) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE for the supplied sequences."""

    predicted_list = list(predicted)
    actual_list = list(actual)
    if not predicted_list or not actual_list:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
    if len(predicted_list) != len(actual_list):
        raise ValueError("Predicted and actual sequences must have the same length.")

    n = len(predicted_list)
    absolute_errors = [abs(p - a) for p, a in zip(predicted_list, actual_list)]
    mae = sum(absolute_errors) / n
    squared_errors = [(p - a) ** 2 for p, a in zip(predicted_list, actual_list)]
    rmse = sqrt(sum(squared_errors) / n)

    epsilon = 1e-8
    percentage_errors = [
        abs((a - p) / (abs(a) if abs(a) > epsilon else epsilon))
        for p, a in zip(predicted_list, actual_list)
    ]
    mape = sum(percentage_errors) / n
    return {"mae": mae, "rmse": rmse, "mape": mape}


class HMMCryptoAlgorithm(QCAlgorithm):
    """Main QuantConnect algorithm orchestrating the HMM crypto pipeline."""

    def Initialize(self) -> None:  # noqa: N802 - QC naming convention
        log_phase_checkpoint(LOGGER, "Initializing HMMCryptoAlgorithm")
        from .util.config_loader import (
            load_assets_config,
            load_features_config,
            load_yaml_config,
        )

        settings = load_yaml_config("config/settings.yaml")
        assets_config = load_assets_config()
        self.settings = settings
        self.threshold = float(settings.get("threshold", HMMThresholds().bullish))
        self.lookback_hours = int(settings.get("lookback_hours", 336))
        self.retrain_hours = int(settings.get("retrain_hours", 24))
        self.max_allocation = float(
            settings.get("risk_management", {}).get("max_allocation_per_asset", 0.25)
        )
        self.seed = int(settings.get("seed", 42))

        self.SetStartDate(2022, 1, 1)
        self.SetCash(100000)
        self.SetBrokerageModel()
        self.SetWarmUp(timedelta(hours=self.lookback_hours))

        self._features_config = load_features_config()
        self._contexts: Dict[str, SymbolContext] = {}
        self._last_retrain_time: Optional[datetime] = None

        for asset_key, meta in assets_config.items():
            ticker = meta.get("qc_ticker", asset_key)
            security = self.AddCrypto(ticker, Resolution.Minute)
            consolidator = TradeBarConsolidator(timedelta(hours=1))
            consolidator.DataConsolidated += self._wrap_hour_handler(asset_key)
            if hasattr(self, "SubscriptionManager"):
                try:
                    self.SubscriptionManager.AddConsolidator(security.Symbol, consolidator)
                except AttributeError:
                    LOGGER.debug("SubscriptionManager stub does not support AddConsolidator.")

            self.AddData(LiquidityBitAsk, security.Symbol.Value, Resolution.Hour)
            self.AddData(MarketSentiment, security.Symbol.Value, Resolution.Hour)

            context = SymbolContext(symbol=security.Symbol, lookback=self.lookback_hours)
            context.model = self._try_load_model(asset_key)
            context.scaler = self._try_load_scaler(asset_key)
            self._contexts[asset_key] = context

        log_phase_checkpoint(LOGGER, f"Loaded {len(self._contexts)} symbol contexts")

    # region Lifecycle hooks
    def OnData(self, slice: Slice) -> None:  # noqa: N802 - QC naming convention
        if slice is None:
            return

        liquidity_data = slice.Get(LiquidityBitAsk)
        for symbol, point in getattr(liquidity_data, "items", lambda: [])():
            context = self._get_context(symbol.Value)
            if context is None:
                continue
            context.liquidity_points.append(point)

        sentiment_data = slice.Get(MarketSentiment)
        for symbol, point in getattr(sentiment_data, "items", lambda: [])():
            context = self._get_context(symbol.Value)
            if context is None:
                continue
            context.sentiment_points.append(point)

    # endregion

    # region Internal helpers
    def _wrap_hour_handler(self, asset_key: str):
        def handler(sender, bar: TradeBar) -> None:
            self._on_hour_bar(asset_key, bar)

        return handler

    def _get_context(self, ticker: str) -> Optional[SymbolContext]:
        return self._contexts.get(ticker) or self._contexts.get(ticker.upper())

    def _on_hour_bar(self, asset_key: str, bar: TradeBar) -> None:
        context = self._contexts.get(asset_key)
        if context is None:
            return

        context.trade_bars.append(bar)
        if len(context.trade_bars) < self.lookback_hours:
            return

        if not context.model or not context.scaler:
            LOGGER.debug("Model or scaler missing for %s; skipping inference", asset_key)
            return

        try:
            interim_frame = self._build_interim_frame(context)
        except ValueError as exc:
            LOGGER.debug("Interim frame unavailable for %s: %s", asset_key, exc)
            return

        try:
            feature_frame = self._assemble_feature_frame(interim_frame)
        except ValueError as exc:
            LOGGER.debug("Feature frame unavailable for %s: %s", asset_key, exc)
            return

        latest_features = feature_frame.iloc[-1]
        scaled_row = context.scaler.transform([latest_features.values])  # type: ignore[union-attr]
        probabilities = context.model.predict_proba(scaled_row)[0]

        if not context.state_mapping:
            from .models.state_mapper import map_hidden_states

            context.state_mapping = map_hidden_states(
                context.model.model,
                list(feature_frame.columns),
                getattr(context.scaler, "mean_", [0.0] * len(feature_frame.columns)),
                getattr(context.scaler, "scale_", [1.0] * len(feature_frame.columns)),
            )

        from .models.state_mapper import remap_probabilities

        semantic_probs = remap_probabilities(probabilities, context.state_mapping)
        context.probabilities = semantic_probs

        signal = determine_signal(semantic_probs, self.threshold, context.last_signal)
        allocation = target_allocation(signal, self.max_allocation)
        context.last_signal = signal

        if signal == "consolidation":
            self.Liquidate(context.symbol)
        elif allocation != 0.0:
            self.SetHoldings(context.symbol, allocation)

        self._update_return_tracking(context, feature_frame, probabilities)

    def _build_interim_frame(self, context: SymbolContext):
        if not context.trade_bars:
            raise ValueError("No trade bars collected yet.")
        records: Dict[datetime, Dict[str, float]] = {}

        for bar in context.trade_bars:
            timestamp = getattr(bar, "EndTime", getattr(bar, "Time", None))
            if timestamp is None:
                continue
            timestamp = timestamp.astimezone(UTC)
            records[timestamp] = {
                "timestamp": timestamp,
                "open": float(getattr(bar, "Open", 0.0)),
                "high": float(getattr(bar, "High", 0.0)),
                "low": float(getattr(bar, "Low", 0.0)),
                "close": float(getattr(bar, "Close", 0.0)),
                "volume": float(getattr(bar, "Volume", 0.0)),
            }

        for point in context.liquidity_points:
            timestamp = getattr(point, "Time", None)
            if timestamp is None:
                continue
            timestamp = timestamp.astimezone(UTC)
            record = records.setdefault(timestamp, {"timestamp": timestamp})
            record.update(
                {
                    "best_bid": float(getattr(point, "best_bid", 0.0)),
                    "best_ask": float(getattr(point, "best_ask", 0.0)),
                    "bid_ask_spread": float(getattr(point, "bid_ask_spread", 0.0)),
                    "depth_pct_1": float(getattr(point, "depth_pct_1", 0.0)),
                    "depth_pct_5": float(getattr(point, "depth_pct_5", 0.0)),
                }
            )

        for point in context.sentiment_points:
            timestamp = getattr(point, "Time", None)
            if timestamp is None:
                continue
            timestamp = timestamp.astimezone(UTC)
            record = records.setdefault(timestamp, {"timestamp": timestamp})
            record.update(
                {
                    "fear_greed_score": float(getattr(point, "fear_greed_score", 0.0)),
                    "confidence": float(getattr(point, "confidence", 0.0)),
                    "sentiment_z": float(getattr(point, "z_score", 0.0)),
                }
            )

        if not records:
            raise ValueError("No aligned records available.")

        import pandas as pd

        frame = pd.DataFrame(sorted(records.values(), key=lambda item: item["timestamp"]))
        frame = frame.set_index("timestamp").sort_index()
        frame = frame.dropna(subset=["close", "volume", "fear_greed_score"], how="any")
        if frame.empty:
            raise ValueError("Aligned frame is empty after dropping incomplete rows.")
        frame.reset_index(inplace=True)
        return frame

    def _assemble_feature_frame(self, interim_frame):
        import pandas as pd
        from .features.feature_defs import (
            compute_liquidity_features,
            compute_log_returns,
            compute_rolling_volatility,
            compute_sentiment_features,
            compute_volume_zscore,
        )

        interim_frame = interim_frame.copy()
        interim_frame.set_index("timestamp", inplace=True)

        features: Dict[str, pd.Series] = {}
        returns_config = self._features_config.get("returns", {})
        return_features = compute_log_returns(
            interim_frame["close"], returns_config.get("windows", [1])
        )
        features.update(return_features)

        base_return_key = next(iter(return_features.keys()), None)
        if base_return_key is None:
            raise ValueError("No return features generated.")
        base_returns = return_features[base_return_key]

        volatility_config = self._features_config.get("volatility", {})
        features.update(
            compute_rolling_volatility(base_returns, volatility_config.get("windows", []))
        )

        liquidity_config = self._features_config.get("liquidity", {})
        features.update(
            compute_liquidity_features(
                interim_frame,
                include_spread=liquidity_config.get("include_spread", True),
                include_imbalance=liquidity_config.get("include_imbalance", True),
            )
        )

        sentiment_config = self._features_config.get("sentiment", {})
        features.update(
            compute_sentiment_features(
                interim_frame["fear_greed_score"],
                sentiment_config.get("lags", []),
                sentiment_config.get("zscore_window", 24),
            )
        )

        features["volume_zscore_24h"] = compute_volume_zscore(interim_frame["volume"], window=24)

        feature_frame = pd.DataFrame(features).replace([float("inf"), float("-inf")], None)
        feature_frame = feature_frame.dropna()
        if feature_frame.empty:
            raise ValueError("Feature frame is empty after dropping NaN values.")
        return feature_frame

    def _update_return_tracking(
        self,
        context: SymbolContext,
        feature_frame,
        probabilities: Iterable[float],
    ) -> None:
        if not context.model or not context.scaler:
            return

        feature_columns = list(feature_frame.columns)
        return_column = next(
            (name for name in feature_columns if name.startswith("return_log_")),
            feature_columns[0],
        )
        latest_return = float(feature_frame.iloc[-1][return_column])

        means = context.model.model.means_[:, feature_columns.index(return_column)]
        scale = getattr(context.scaler, "scale_", [1.0] * len(feature_columns))
        mean = getattr(context.scaler, "mean_", [0.0] * len(feature_columns))
        index = feature_columns.index(return_column)
        actual_means = means * scale[index] + mean[index]

        expected_return = float(sum(p * m for p, m in zip(probabilities, actual_means)))
        context.actual_returns.append(latest_return)
        context.predicted_returns.append(expected_return)

    # endregion

    def OnEndOfAlgorithm(self) -> None:  # noqa: N802 - QC naming convention
        for ticker, context in self._contexts.items():
            metrics = compute_error_metrics(context.predicted_returns, context.actual_returns)
            message = (
                f"{ticker} | MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, "
                f"MAPE: {metrics['mape']:.6f}"
            )
            self.Debug(message)

    # region Persistence helpers
    def _try_load_model(self, asset_key: str) -> Optional["GaussianHMMWrapper"]:
        from .models.hmm_model import GaussianHMMWrapper

        try:
            path = hmm_model_output_path(asset_key)
            if not Path(path).exists():
                return None
            return GaussianHMMWrapper.load(path)
        except (FileNotFoundError, ImportError) as exc:
            LOGGER.warning("Unable to load HMM model for %s: %s", asset_key, exc)
            return None

    def _try_load_scaler(self, asset_key: str):
        import pickle

        path = scaler_output_path(asset_key)
        if not Path(path).exists():
            return None
        try:
            with Path(path).open("rb") as handle:
                return pickle.load(handle)
        except (pickle.PickleError, ModuleNotFoundError) as exc:
            LOGGER.warning("Unable to load scaler for %s: %s", asset_key, exc)
            return None

    # endregion
