"""IO helper exports with optional schema validation bindings."""

from .custom_export import (
    export_liquidity_timeseries,
    export_probability_timeseries,
    export_sentiment_timeseries,
)
from .file_locator import (
    diagnostics_output_path,
    hmm_model_output_path,
    hmm_probability_export_path,
    interim_output_path,
    liquidity_export_path,
    metrics_output_path,
    processed_output_path,
    project_root,
    scaler_output_path,
    sentiment_export_path,
)

try:  # pragma: no cover - exercised when pandas is installed
    from .schemas import (
        DataFrameValidationError,
        validate_binance_orderbook,
        validate_coinstats,
        validate_cryptoquant,
        validate_features,
        validate_interim,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight envs

    class DataFrameValidationError(RuntimeError):
        """Raised when dataframe validation is unavailable due to missing deps."""

    def _missing_dependency(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "pandas is required for dataframe validation helpers"
        )

    validate_binance_orderbook = _missing_dependency
    validate_coinstats = _missing_dependency
    validate_cryptoquant = _missing_dependency
    validate_features = _missing_dependency
    validate_interim = _missing_dependency


__all__ = [
    "export_liquidity_timeseries",
    "export_probability_timeseries",
    "export_sentiment_timeseries",
    "diagnostics_output_path",
    "hmm_model_output_path",
    "hmm_probability_export_path",
    "interim_output_path",
    "liquidity_export_path",
    "metrics_output_path",
    "processed_output_path",
    "project_root",
    "scaler_output_path",
    "sentiment_export_path",
    "DataFrameValidationError",
    "validate_binance_orderbook",
    "validate_coinstats",
    "validate_cryptoquant",
    "validate_features",
    "validate_interim",
]
