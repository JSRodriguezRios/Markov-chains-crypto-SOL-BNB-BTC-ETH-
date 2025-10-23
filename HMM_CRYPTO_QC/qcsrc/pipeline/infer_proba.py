"""Offline inference pipeline for HMM posterior probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..io.custom_export import export_probability_timeseries
from ..io.file_locator import hmm_model_output_path, processed_output_path, scaler_output_path
from ..models.hmm_model import GaussianHMMWrapper
from ..util.logging_utils import create_logger, log_phase_checkpoint
from ..util.timezones import ensure_utc

try:  # pragma: no cover - optional dependency handling
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

LOGGER = create_logger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """Result metadata for probability inference."""

    symbol: str
    probability_path: Path
    latest_timestamp: "pd.Timestamp"


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover - defensive guard
        raise ImportError("pandas is required for probability inference. Install pandas first.")


def _load_features(symbol: str, features: Optional["pd.DataFrame"] = None) -> "pd.DataFrame":
    if features is not None:
        return features.copy()

    _require_pandas()
    parquet_path = processed_output_path(symbol)
    csv_path = parquet_path.with_suffix(".csv")

    if parquet_path.exists():
        try:
            frame = pd.read_parquet(parquet_path)
        except (ImportError, ValueError):
            LOGGER.warning(
                "Unable to read parquet for %s; falling back to CSV if present.", parquet_path
            )
        else:
            frame = frame.reset_index().rename(columns={"index": "timestamp"})
            return frame

    if csv_path.exists():
        frame = pd.read_csv(csv_path, parse_dates=[0])
        frame.rename(columns={frame.columns[0]: "timestamp"}, inplace=True)
        return frame

    raise FileNotFoundError(f"No processed features located for {symbol} at {parquet_path}")


def _load_scaler(symbol: str):
    scaler_path = scaler_output_path(symbol)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file missing for {symbol}: {scaler_path}")
    with scaler_path.open("rb") as handle:
        import pickle

        return pickle.load(handle)


def _prepare_matrix(
    frame: "pd.DataFrame", scaler, feature_columns: Sequence[str]
) -> np.ndarray:
    values = frame.loc[:, feature_columns].to_numpy()
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        return values

    # Detect if the frame already appears scaled (mean ~0, std ~1).
    if np.allclose(values.mean(axis=0), 0, atol=1e-3) and np.allclose(
        values.std(axis=0), 1, atol=1e-3
    ):
        return values

    return (values - scaler.mean_) / scaler.scale_


def infer_probabilities(
    symbol: str,
    features: Optional["pd.DataFrame"] = None,
) -> InferenceResult:
    """Generate posterior probabilities for ``symbol`` and export them for QC."""

    _require_pandas()
    scaler = _load_scaler(symbol)
    model_path = hmm_model_output_path(symbol)
    model = GaussianHMMWrapper.load(model_path)

    feature_frame = _load_features(symbol, features)
    feature_frame["timestamp"] = ensure_utc(feature_frame["timestamp"])
    feature_frame = feature_frame.sort_values("timestamp").drop_duplicates("timestamp")

    feature_columns = [column for column in feature_frame.columns if column != "timestamp"]
    if not feature_columns:
        raise ValueError("Feature frame must include at least one predictor column.")

    matrix = _prepare_matrix(feature_frame, scaler, feature_columns)
    probabilities = model.predict_proba(matrix)

    probability_frame = pd.DataFrame(
        probabilities,
        columns=["prob_bullish", "prob_bearish", "prob_consolidation"],
    )
    probability_frame.insert(0, "time", feature_frame["timestamp"])

    export_path = export_probability_timeseries(symbol, probability_frame)
    latest_timestamp = probability_frame["time"].max()
    log_phase_checkpoint(
        LOGGER,
        f"Inferred probabilities for {symbol} and persisted to {export_path}",
    )
    return InferenceResult(
        symbol=symbol,
        probability_path=export_path,
        latest_timestamp=latest_timestamp,
    )


__all__ = ["infer_probabilities", "InferenceResult"]
