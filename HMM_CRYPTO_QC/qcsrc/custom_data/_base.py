"""Shared helpers for QuantConnect custom data definitions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

UTC = timezone.utc


def parse_timestamp(value: str) -> datetime:
    """Parse ``value`` into a timezone-aware UTC datetime."""

    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Timestamp value cannot be empty")

    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1]

    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def next_hour(moment: datetime) -> datetime:
    """Return the exclusive end time one hour after ``moment``."""

    return moment + timedelta(hours=1)


def is_header(tokens: Iterable[str]) -> bool:
    """Determine if ``tokens`` represent a CSV header row."""

    normalized = {token.strip().lower() for token in tokens}
    if "time" not in normalized:
        return False
    return bool(
        {"prob_bullish", "bid_ask_spread", "fear_greed_score"}.intersection(normalized)
    )
