"""Training orchestration for Gaussian HMM models."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Tuple

import numpy as np

from ..io.file_locator import (
    diagnostics_output_path,
    hmm_model_output_path,
    metrics_output_path,
    processed_output_path,
    scaler_output_path,
)
from ..models.hmm_model import GaussianHMMWrapper
from ..models.state_mapper import map_hidden_states
from ..util.config_loader import load_assets_config, load_yaml_config
from ..util.logging_utils import create_logger, log_phase_checkpoint

try:  # pragma: no cover - optional dependencies
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.preprocessing import StandardScaler

LOGGER = create_logger(__name__)
RETURN_FEATURE_PREFERENCES: Tuple[str, ...] = ("return_log_1h", "return_log_4h")


@dataclass(frozen=True)
class TrainingResult:
    symbol: str
    diagnostics_path: Path
    metrics_path: Path
    model_path: Path


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover - defensive
        raise ImportError(
            "pandas is required for train_hmm pipeline. Install pandas to continue."
        )


def _load_features(symbol: str) -> "pd.DataFrame":
    _require_pandas()
    path = processed_output_path(symbol)
    csv_fallback = path.with_suffix(".csv")
    if path.exists():
        try:
            frame = pd.read_parquet(path)
        except (ImportError, ValueError):
            LOGGER.warning(
                "Parquet engine unavailable for %s; falling back to CSV export if present.",
                path,
            )
        else:
            frame = frame.reset_index().rename(columns={"index": "timestamp"})
            return frame
    if csv_fallback.exists():
        frame = pd.read_csv(csv_fallback, parse_dates=[0])
        frame.rename(columns={frame.columns[0]: "timestamp"}, inplace=True)
        return frame
    raise FileNotFoundError(f"Processed feature matrix for {symbol} not found.")


def _load_scaler(symbol: str) -> "StandardScaler":
    scaler_path = scaler_output_path(symbol)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file for {symbol} not found at {scaler_path}.")
    with scaler_path.open("rb") as handle:
        return pickle.load(handle)


def _split_train_validation(df: "pd.DataFrame", validation_fraction: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    sorted_df = df.sort_values("timestamp")
    features_df = sorted_df.drop(columns=["timestamp"])
    values = features_df.to_numpy()
    if len(values) < 2:
        raise ValueError("Not enough rows to perform train/validation split.")
    split_index = max(int(len(values) * (1 - validation_fraction)), 1)
    train = values[:split_index]
    validation = values[split_index:]
    if validation.size == 0:
        validation = train
    return train, validation


def _determine_return_column(columns: Iterable[str]) -> str:
    column_list = list(columns)
    for candidate in RETURN_FEATURE_PREFERENCES:
        if candidate in column_list:
            return candidate
    if not column_list:
        raise ValueError("No feature columns available to determine return feature.")
    LOGGER.warning(
        "Preferred return feature missing; defaulting to %s", column_list[0]
    )
    return column_list[0]


def _actual_returns(
    scaled_matrix: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    return_index: int,
) -> np.ndarray:
    return scaled_matrix[:, return_index] * scaler_scale[return_index] + scaler_mean[return_index]


def _expected_returns(
    probabilities: np.ndarray,
    state_means: np.ndarray,
) -> np.ndarray:
    return probabilities @ state_means


def _heuristic_labels(actual_returns: np.ndarray) -> np.ndarray:
    labels = np.full(actual_returns.shape[0], "consolidation", dtype=object)
    labels[actual_returns > 0] = "bullish"
    labels[actual_returns < 0] = "bearish"
    return labels


def _confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
    classes: Iterable[str] = ("bullish", "bearish", "consolidation"),
) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {
        actual_label: {pred_label: 0 for pred_label in classes} for actual_label in classes
    }
    for a_label, p_label in zip(actual, predicted):
        matrix.setdefault(a_label, {label: 0 for label in classes})
        matrix[a_label][p_label] = matrix[a_label].get(p_label, 0) + 1
    return matrix


def _persist_json(path: Path, payload: Dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)
    return path


def train_symbol(symbol: str) -> TrainingResult:
    settings = load_yaml_config("config/settings.yaml")
    scaler = _load_scaler(symbol)
    features = _load_features(symbol)
    train_matrix, validation_matrix = _split_train_validation(features)

    wrapper = GaussianHMMWrapper(
        n_components=settings.get("max_states", 3),
        random_state=settings.get("seed"),
    )
    summary = wrapper.fit(train_matrix)
    validation_score = wrapper.score(validation_matrix)

    feature_columns = [column for column in features.columns if column != "timestamp"]
    mapping = map_hidden_states(
        wrapper.model,
        feature_columns,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        preferred_returns=RETURN_FEATURE_PREFERENCES,
    )

    full_matrix = features.drop(columns=["timestamp"]).to_numpy()
    probabilities = wrapper.predict_proba(full_matrix)

    return_column = _determine_return_column(feature_columns)
    return_index = feature_columns.index(return_column)
    actual = _actual_returns(full_matrix, scaler.mean_, scaler.scale_, return_index)

    state_means_scaled = wrapper.state_means(return_index)
    state_means_actual = state_means_scaled * scaler.scale_[return_index] + scaler.mean_[return_index]
    expected = _expected_returns(probabilities, state_means_actual)

    mae = float(np.mean(np.abs(expected - actual)))
    rmse = float(np.sqrt(np.mean((expected - actual) ** 2)))
    denominator = np.where(actual == 0, 1e-8, np.abs(actual))
    mape = float(np.mean(np.abs((expected - actual) / denominator)) * 100)

    predicted_labels = np.array(
        [
            mapping.get(int(idx), "consolidation")
            for idx in np.argmax(probabilities, axis=1)
        ]
    )
    actual_labels = _heuristic_labels(actual)
    confusion = _confusion_matrix(actual_labels, predicted_labels)

    diagnostics_payload = {
        "symbol": symbol,
        "log_likelihood": summary.log_likelihood,
        "validation_log_likelihood": validation_score,
        "aic": summary.aic,
        "bic": summary.bic,
        "state_mapping": mapping,
        "state_means_actual": state_means_actual.tolist(),
        "posterior_head": probabilities[:5].tolist(),
        "confusion_matrix": confusion,
    }

    metrics_payload = {
        "symbol": symbol,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }

    diagnostics_path = _persist_json(diagnostics_output_path(symbol), diagnostics_payload)
    metrics_path = _persist_json(metrics_output_path(symbol), metrics_payload)
    model_path = wrapper.save(hmm_model_output_path(symbol))

    log_phase_checkpoint(LOGGER, f"Completed HMM training for {symbol}")
    return TrainingResult(
        symbol=symbol,
        diagnostics_path=diagnostics_path,
        metrics_path=metrics_path,
        model_path=model_path,
    )


def train_all_symbols() -> Dict[str, TrainingResult]:
    assets = load_assets_config()
    results: Dict[str, TrainingResult] = {}
    for symbol in assets.keys():
        try:
            results[symbol] = train_symbol(symbol)
        except Exception as exc:  # pragma: no cover - orchestration guard
            LOGGER.exception("Failed to train symbol %s: %s", symbol, exc)
    return results


if __name__ == "__main__":
    train_all_symbols()
