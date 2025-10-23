"""Utilities for timezone normalization."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .logging_utils import create_logger

LOGGER = create_logger(__name__)


UTC = timezone.utc


def ensure_utc(series: pd.Series) -> pd.Series:
    """Return a timezone-aware copy of ``series`` normalized to UTC."""

    if series.empty:
        return series

    converted = pd.to_datetime(series, utc=True, errors="coerce")
    if converted.isna().any():
        LOGGER.warning("Encountered NaT values while coercing timestamps to UTC.")
    return converted


def hourly_range(start: datetime, end: datetime) -> pd.DatetimeIndex:
    """Generate an hourly UTC index inclusive of ``start`` and exclusive of ``end``."""

    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    if end_utc <= start_utc:
        raise ValueError("end must be later than start for hourly_range")
    return pd.date_range(start=start_utc, end=end_utc, freq="1H", tz=UTC, inclusive="left")


def floor_to_hour(moment: datetime) -> datetime:
    """Floor ``moment`` to the previous hour in UTC."""

    as_utc = moment.astimezone(UTC)
    return as_utc.replace(minute=0, second=0, microsecond=0)


def ceil_to_hour(moment: datetime) -> datetime:
    """Ceil ``moment`` to the next hour boundary in UTC."""

    floored = floor_to_hour(moment)
    if moment.astimezone(UTC) == floored:
        return floored
    return floored + pd.Timedelta(hours=1)


def assert_utc_index(index: Iterable[datetime]) -> None:
    """Ensure all datetimes in ``index`` carry UTC tzinfo."""

    for stamp in index:
        if stamp.tzinfo != UTC:
            raise ValueError("Timestamp index contains non-UTC entries")
