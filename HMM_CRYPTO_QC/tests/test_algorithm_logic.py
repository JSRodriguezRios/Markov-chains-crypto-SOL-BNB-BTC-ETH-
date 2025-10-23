from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import math

import pytest

from qcsrc.HMMCryptoAlgorithm import compute_error_metrics, determine_signal, target_allocation


@pytest.mark.parametrize(
    "probabilities, previous, expected",
    [
        ({"bullish": 0.75, "bearish": 0.1, "consolidation": 0.15}, "neutral", "bullish"),
        ({"bullish": 0.1, "bearish": 0.8, "consolidation": 0.1}, "bullish", "bearish"),
        ({"bullish": 0.1, "bearish": 0.2, "consolidation": 0.75}, "bearish", "consolidation"),
        ({"bullish": 0.3, "bearish": 0.2, "consolidation": 0.4}, "bullish", "bullish"),
    ],
)
def test_determine_signal_respects_thresholds(probabilities, previous, expected):
    threshold = 0.7
    result = determine_signal(probabilities, threshold, previous)
    assert result == expected


@pytest.mark.parametrize(
    "signal, expected",
    [
        ("bullish", 0.25),
        ("bearish", -0.25),
        ("consolidation", 0.0),
        ("neutral", 0.0),
    ],
)
def test_target_allocation_maps_signal_to_weight(signal, expected):
    assert target_allocation(signal, 0.25) == pytest.approx(expected)


def test_compute_error_metrics_handles_sequences():
    predicted = [0.05, -0.02, 0.01]
    actual = [0.04, -0.01, 0.015]
    metrics = compute_error_metrics(predicted, actual)
    assert metrics["mae"] == pytest.approx(0.01 / 3 + 0.01 / 3 + 0.005 / 3, rel=1e-6)
    rmse_expected = math.sqrt(((0.01) ** 2 + (-0.01) ** 2 + (-0.005) ** 2) / 3)
    assert metrics["rmse"] == pytest.approx(rmse_expected, rel=1e-6)
    assert metrics["mape"] >= 0.0


def test_compute_error_metrics_returns_zero_for_empty_sequences():
    metrics = compute_error_metrics([], [])
    assert metrics == {"mae": 0.0, "rmse": 0.0, "mape": 0.0}


def test_compute_error_metrics_requires_equal_lengths():
    with pytest.raises(ValueError):
        compute_error_metrics([0.1, 0.2], [0.1])
