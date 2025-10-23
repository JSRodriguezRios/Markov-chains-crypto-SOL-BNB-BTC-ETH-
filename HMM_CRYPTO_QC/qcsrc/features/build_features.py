"""Feature matrix construction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..io import schemas
from ..io.file_locator import processed_output_path, scaler_output_path
from ..util.config_loader import load_features_config
from ..util.logging_utils import create_logger, log_phase_checkpoint
from ..util.timezones import ensure_utc
from .feature_defs import (
    compute_liquidity_features,
    compute_log_returns,
    compute_rolling_volatility,
    compute_sentiment_features,
    compute_volume_zscore,
)

LOGGER = create_logger(__name__)


@dataclass(frozen=True)
class FeatureBuildResult:
    """Container bundling the feature matrix and scaler path."""

    features: pd.DataFrame
    scaler: StandardScaler


def _ensure_dataframe(df_interim: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df_interim.columns:
        raise KeyError("Interim dataframe must include a timestamp column.")

    frame = df_interim.copy()
    frame["timestamp"] = ensure_utc(frame["timestamp"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.set_index("timestamp")
    return frame


def _persist_features(symbol: str, features: pd.DataFrame) -> None:
    output_path = processed_output_path(symbol)
    try:
        features.to_parquet(output_path, index=True)
    except (ImportError, ValueError) as exc:
        fallback = output_path.with_suffix(".csv")
        LOGGER.warning("%s while writing parquet; using CSV fallback at %s", exc, fallback)
        features.to_csv(fallback)


def _persist_scaler(symbol: str, scaler: StandardScaler) -> None:
    import pickle

    path = scaler_output_path(symbol)
    with path.open("wb") as handle:
        pickle.dump(scaler, handle)


def _assemble_feature_columns(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    config = load_features_config()

    feature_columns: Dict[str, pd.Series] = {}
    returns_config = config.get("returns", {})
    return_features = compute_log_returns(
        frame["close"], returns_config.get("windows", [1])
    )
    feature_columns.update(return_features)

    base_return_key = next(iter(return_features.keys()), None)
    base_returns = (
        return_features.get("return_log_1h")
        if "return_log_1h" in return_features
        else (return_features[base_return_key] if base_return_key else frame["close"].pct_change())
    )

    volatility_config = config.get("volatility", {})
    feature_columns.update(
        compute_rolling_volatility(base_returns, volatility_config.get("windows", []))
    )

    liquidity_config = config.get("liquidity", {})
    liquidity_features = compute_liquidity_features(
        frame,
        include_spread=liquidity_config.get("include_spread", True),
        include_imbalance=liquidity_config.get("include_imbalance", True),
    )
    feature_columns.update(liquidity_features)

    sentiment_config = config.get("sentiment", {})
    sentiment_features = compute_sentiment_features(
        frame["fear_greed_score"],
        lags=sentiment_config.get("lags", []),
        zscore_window=sentiment_config.get("zscore_window", 24),
    )
    feature_columns.update(sentiment_features)

    volume_feature = compute_volume_zscore(frame["volume"], window=24)
    feature_columns["volume_zscore_24h"] = volume_feature

    features_df = pd.DataFrame(feature_columns)
    features_df.index = frame.index
    return features_df


def build_feature_matrix(symbol: str, df_interim: pd.DataFrame) -> FeatureBuildResult:
    """Transform interim data into a normalized feature matrix."""

    interim_frame = _ensure_dataframe(df_interim)
    assembled = _assemble_feature_columns(symbol, interim_frame)

    assembled = assembled.replace([np.inf, -np.inf], np.nan).dropna()
    if assembled.empty:
        raise ValueError("No rows available after feature engineering; check input coverage.")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(assembled.values)
    scaled = pd.DataFrame(scaled_values, index=assembled.index, columns=assembled.columns)
    scaled.reset_index(inplace=True)
    scaled.rename(columns={"index": "timestamp"}, inplace=True)

    schemas.validate_features(scaled)
    _persist_features(symbol, scaled.set_index("timestamp"))
    _persist_scaler(symbol, scaler)
    log_phase_checkpoint(LOGGER, f"Built features for {symbol} spanning {len(scaled)} rows")

    return FeatureBuildResult(features=scaled, scaler=scaler)
