"""Reusable feature computation primitives."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from ..util.logging_utils import create_logger

LOGGER = create_logger(__name__)


def compute_log_returns(close: pd.Series, windows: Iterable[int]) -> Dict[str, pd.Series]:
    """Return a mapping of log-return features for the provided windows."""

    features: Dict[str, pd.Series] = {}
    safe_close = close.astype(float)
    for window in windows:
        if window <= 0:
            LOGGER.warning("Ignoring non-positive return window: %s", window)
            continue
        shifted = safe_close.shift(window)
        with np.errstate(divide="ignore"):
            feature = np.log(safe_close / shifted)
        features[f"return_log_{window}h"] = feature.replace([np.inf, -np.inf], np.nan)
    return features


def compute_rolling_volatility(returns: pd.Series, windows: Iterable[int]) -> Dict[str, pd.Series]:
    """Return a mapping of rolling standard deviation features."""

    vol_features: Dict[str, pd.Series] = {}
    safe_returns = returns.astype(float)
    for window in windows:
        if window <= 1:
            LOGGER.warning("Ignoring volatility window <= 1 hour: %s", window)
            continue
        vol_features[f"volatility_{window}h"] = safe_returns.rolling(window).std(ddof=0)
    return vol_features


def compute_liquidity_features(frame: pd.DataFrame, include_spread: bool, include_imbalance: bool) -> Dict[str, pd.Series]:
    """Derive liquidity-oriented predictors from interim order-book data."""

    features: Dict[str, pd.Series] = {}
    depth_mapping = {
        "depth_pct_1": "liquidity_depth_z_1pct",
        "depth_pct_5": "liquidity_depth_z_5pct",
    }
    for column, feature_name in depth_mapping.items():
        series = frame.get(column)
        if series is None:
            continue
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            features[feature_name] = pd.Series(0.0, index=series.index)
        else:
            features[feature_name] = (series - mean) / std

    if include_spread and "bid_ask_spread" in frame:
        features["bid_ask_spread"] = frame["bid_ask_spread"].astype(float)

    if include_imbalance:
        bid = frame.get("best_bid")
        ask = frame.get("best_ask")
        if bid is not None and ask is not None:
            imbalance = (bid - ask) / (bid + ask)
            features["order_book_imbalance"] = imbalance.replace([np.inf, -np.inf], np.nan)

    return features


def compute_sentiment_features(score: pd.Series, lags: Iterable[int], zscore_window: int) -> Dict[str, pd.Series]:
    """Construct sentiment z-score and lag features."""

    features: Dict[str, pd.Series] = {}
    rolling_mean = score.rolling(zscore_window, min_periods=1).mean()
    rolling_std = score.rolling(zscore_window, min_periods=1).std(ddof=0)
    zscore = (score - rolling_mean) / rolling_std.replace(0, np.nan)
    features["sentiment_zscore"] = zscore.replace([np.inf, -np.inf], np.nan)

    for lag in lags:
        if lag <= 0:
            LOGGER.warning("Ignoring non-positive sentiment lag: %s", lag)
            continue
        features[f"sentiment_zscore_lag_{lag}h"] = features["sentiment_zscore"].shift(lag)
    return features


def compute_volume_zscore(volume: pd.Series, window: int = 24) -> pd.Series:
    """Return a rolling z-score of traded volume as an optional feature."""

    rolling_mean = volume.rolling(window, min_periods=1).mean()
    rolling_std = volume.rolling(window, min_periods=1).std(ddof=0)
    zscore = (volume - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan)
