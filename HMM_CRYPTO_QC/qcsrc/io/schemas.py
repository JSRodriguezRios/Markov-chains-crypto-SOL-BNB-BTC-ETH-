"""Schema validation helpers for pipeline outputs."""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Tuple, TYPE_CHECKING

try:  # pragma: no cover - exercised only when pandas is available
    import pandas as pd
    from pandas.api.types import is_datetime64_any_dtype
except ModuleNotFoundError:  # pragma: no cover - optional dependency in minimal envs
    pd = None  # type: ignore[assignment]
    is_datetime64_any_dtype = None

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    import pandas as pd


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover - executed only when dependency missing
        raise ModuleNotFoundError("pandas is required for qcsrc.io.schemas")


@dataclass(frozen=True)
class SchemaField:
    """Describe an expected dataframe column."""

    name: str
    dtype: Tuple[type, ...] | type
    allow_null: bool = False

    def acceptable_types(self) -> Tuple[type, ...]:
        if isinstance(self.dtype, tuple):
            return self.dtype
        return (self.dtype,)

    def is_datetime(self) -> bool:
        return any(member is datetime for member in self.acceptable_types())


class DataFrameValidationError(ValueError):
    """Raised when a dataframe does not match the expected schema."""


def _validate_columns(df: "pd.DataFrame", expected: Iterable[SchemaField]) -> None:
    _require_pandas()
    missing = [field.name for field in expected if field.name not in df.columns]
    if missing:
        raise DataFrameValidationError(f"Missing columns: {missing}")


def _validate_datetime(series: "pd.Series", field: SchemaField) -> None:
    _require_pandas()
    if series.dropna().empty:
        return
    if is_datetime64_any_dtype is None or not is_datetime64_any_dtype(series):
        raise DataFrameValidationError(
            f"Column '{field.name}' is not datetime typed."
        )


def _coerce_numeric(series: "pd.Series", field: SchemaField) -> None:
    _require_pandas()
    if series.dropna().empty:
        return
    last_error: Exception | None = None
    for dtype in field.acceptable_types():
        try:
            series.astype(dtype)
            return
        except (ValueError, TypeError) as exc:
            last_error = exc
    raise DataFrameValidationError(
        f"Column '{field.name}' cannot be coerced to {field.acceptable_types()}"
    ) from last_error


def _validate_types(df: "pd.DataFrame", expected: Iterable[SchemaField]) -> None:
    _require_pandas()
    for field in expected:
        series = df[field.name]
        if field.allow_null and series.dropna().empty:
            continue
        if field.is_datetime():
            _validate_datetime(series, field)
        else:
            _coerce_numeric(series, field)


def validate_cryptoquant(df: "pd.DataFrame") -> None:
    _require_pandas()
    expected = [
        SchemaField("timestamp", datetime),
        SchemaField("open", float),
        SchemaField("high", float),
        SchemaField("low", float),
        SchemaField("close", float),
        SchemaField("volume", float),
        SchemaField("quote_volume", float),
    ]
    _validate_columns(df, expected)
    _validate_types(df, expected)


def validate_binance_orderbook(df: "pd.DataFrame") -> None:
    _require_pandas()
    expected = [
        SchemaField("timestamp", datetime),
        SchemaField("best_bid", float),
        SchemaField("best_ask", float),
        SchemaField("bid_ask_spread", float),
        SchemaField("depth_pct_1", float),
        SchemaField("depth_pct_5", float),
    ]
    _validate_columns(df, expected)
    _validate_types(df, expected)


def validate_coinstats(df: "pd.DataFrame") -> None:
    _require_pandas()
    expected = [
        SchemaField("timestamp", datetime),
        SchemaField("fear_greed_score", float),
        SchemaField("confidence", float),
    ]
    _validate_columns(df, expected)
    _validate_types(df, expected)


def validate_interim(df: "pd.DataFrame") -> None:
    _require_pandas()
    expected = [
        SchemaField("timestamp", datetime),
        SchemaField("open", float),
        SchemaField("high", float),
        SchemaField("low", float),
        SchemaField("close", float),
        SchemaField("volume", float),
        SchemaField("quote_volume", float),
        SchemaField("best_bid", float),
        SchemaField("best_ask", float),
        SchemaField("bid_ask_spread", float),
        SchemaField("depth_pct_1", float),
        SchemaField("depth_pct_5", float),
        SchemaField("fear_greed_score", float),
        SchemaField("confidence", float),
    ]
    _validate_columns(df, expected)
    _validate_types(df, expected)


def validate_features(df: "pd.DataFrame") -> None:
    _require_pandas()
    """Ensure feature matrices include timestamp and numeric predictors."""

    numeric_columns = [column for column in df.columns if column != "timestamp"]
    if not numeric_columns:
        raise DataFrameValidationError("Feature matrix does not contain any features.")

    _validate_columns(df, [SchemaField("timestamp", datetime)])
    _validate_types(df[["timestamp"]], [SchemaField("timestamp", datetime)])
    for column in numeric_columns:
        _coerce_numeric(df[column], SchemaField(column, float))
