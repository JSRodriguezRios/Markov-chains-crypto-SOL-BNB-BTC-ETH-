"""Timezone helpers with graceful pandas handling."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency handling
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

from .logging_utils import create_logger

LOGGER = create_logger(__name__)
UTC = timezone.utc


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover - defensive guard
        raise ImportError("pandas is required for timezone utilities.")


def ensure_utc(series: Sequence) -> "pd.Series":  # type: ignore[override]
    """Return a timezone-aware copy of ``series`` normalized to UTC."""

    _require_pandas()
    pandas_series = pd.Series(series)
    if pandas_series.empty:
        return pandas_series

    converted = pd.to_datetime(pandas_series, utc=True, errors="coerce")
    if converted.isna().any():
        LOGGER.warning("Encountered NaT values while coercing timestamps to UTC.")
    return converted


def hourly_range(start: datetime, end: datetime):  # type: ignore[override]
    """Generate an hourly UTC index inclusive of ``start`` and exclusive of ``end``."""

    _require_pandas()
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
    return floored + timedelta(hours=1)


def assert_utc_index(index: Iterable[datetime]) -> None:
    """Ensure all datetimes in ``index`` carry UTC tzinfo."""

    for stamp in index:
        if stamp.tzinfo != UTC:
            raise ValueError("Timestamp index contains non-UTC entries")
