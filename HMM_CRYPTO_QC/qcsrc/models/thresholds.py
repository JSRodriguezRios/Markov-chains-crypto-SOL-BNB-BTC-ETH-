"""Signal threshold helpers for HMM state probabilities."""
from __future__ import annotations

from typing import Mapping

DEFAULT_THRESHOLD = 0.7


def _probability(probabilities: Mapping[str, float], key: str) -> float:
    value = probabilities.get(key, 0.0)
    return float(value)


def should_long(probabilities: Mapping[str, float], threshold: float = DEFAULT_THRESHOLD) -> bool:
    if not probabilities:
        return False
    return _probability(probabilities, "bullish") >= threshold


def should_short(probabilities: Mapping[str, float], threshold: float = DEFAULT_THRESHOLD) -> bool:
    if not probabilities:
        return False
    return _probability(probabilities, "bearish") >= threshold


def should_flat(probabilities: Mapping[str, float], threshold: float = DEFAULT_THRESHOLD) -> bool:
    if not probabilities:
        return True
    return _probability(probabilities, "consolidation") >= threshold
