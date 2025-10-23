"""Utilities to interpret hidden states into semantic market regimes."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from ..util.logging_utils import create_logger

LOGGER = create_logger(__name__)


def _locate_return_column(columns: Sequence[str], preferred: Iterable[str]) -> str:
    for candidate in preferred:
        if candidate in columns:
            return candidate
    if not columns:
        raise ValueError("Feature matrix is empty; cannot determine return column.")
    LOGGER.warning("Preferred return feature not found; defaulting to first column: %s", columns[0])
    return columns[0]


def map_hidden_states(
    model: "GaussianHMM",
    feature_columns: Sequence[str],
    scaler_mean: Sequence[float],
    scaler_scale: Sequence[float],
    preferred_returns: Iterable[str] = ("return_log_1h", "return_log_4h"),
) -> Dict[int, str]:
    """Map HMM state indices to semantic labels using return expectations."""

    return_column = _locate_return_column(feature_columns, preferred_returns)
    return_index = feature_columns.index(return_column)

    scaled_means = model.means_[:, return_index]
    scale = scaler_scale[return_index]
    mean = scaler_mean[return_index]
    actual_means = scaled_means * scale + mean

    order = np.argsort(actual_means)
    mapping: Dict[int, str] = {}
    if len(order) < 3:
        LOGGER.warning(
            "Expected at least three states; received %s. Falling back to partial mapping.",
            len(order),
        )
    labels = ["bearish", "consolidation", "bullish"]
    for label, state_index in zip(labels, order):
        mapping[int(state_index)] = label
    return mapping


def remap_probabilities(
    probabilities: np.ndarray,
    mapping: Mapping[int, str],
) -> Dict[str, float]:
    """Convert state-indexed probabilities to a semantic dictionary."""

    result: Dict[str, float] = {"bullish": 0.0, "bearish": 0.0, "consolidation": 0.0}
    for idx, value in enumerate(probabilities):
        label = mapping.get(idx)
        if label is None:
            continue
        result[label] = float(value)
    return result
