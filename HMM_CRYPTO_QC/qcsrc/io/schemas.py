"""Schema validation helpers for pipeline outputs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


@dataclass(frozen=True)
class SchemaField:
    name: str
    dtype: type


class DataFrameValidationError(ValueError):
    """Raised when a dataframe does not match the expected schema."""


def _validate_columns(df: pd.DataFrame, expected: Iterable[SchemaField]) -> None:
    missing = [field.name for field in expected if field.name not in df.columns]
    if missing:
        raise DataFrameValidationError(f"Missing columns: {missing}")


def _validate_types(df: pd.DataFrame, expected: Iterable[SchemaField]) -> None:
    for field in expected:
        series = df[field.name]
        if not series.dropna().empty:
            if field.dtype is datetime:
                if not is_datetime64_any_dtype(series):
                    raise DataFrameValidationError(
                        f"Column '{field.name}' is not datetime typed."
                    )
            else:
                try:
                    series.astype(field.dtype)
                except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
                    raise DataFrameValidationError(
                        f"Column '{field.name}' cannot be coerced to {field.dtype}."
                    ) from exc


def validate_cryptoquant(df: pd.DataFrame) -> None:
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


def validate_binance_orderbook(df: pd.DataFrame) -> None:
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


def validate_coinstats(df: pd.DataFrame) -> None:
    expected = [
        SchemaField("timestamp", datetime),
        SchemaField("fear_greed_score", float),
        SchemaField("confidence", float),
    ]
    _validate_columns(df, expected)
    _validate_types(df, expected)


def validate_interim(df: pd.DataFrame) -> None:
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
