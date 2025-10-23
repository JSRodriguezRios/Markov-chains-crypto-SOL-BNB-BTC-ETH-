from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

try:  # pragma: no cover - guard minimal environments
    import pandas as pd  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip(
        "pandas and scikit-learn are required for feature engineering tests",
        allow_module_level=True,
    )

from qcsrc.features.build_features import build_feature_matrix
from qcsrc.io.file_locator import processed_output_path, scaler_output_path


def _sample_interim_frame(rows: int = 40) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(hours=idx) for idx in range(rows)]
    data = {
        "timestamp": timestamps,
        "open": np.linspace(100, 140, rows),
        "high": np.linspace(101, 145, rows),
        "low": np.linspace(99, 135, rows),
        "close": np.linspace(100, 150, rows),
        "volume": np.linspace(1000, 2000, rows),
        "quote_volume": np.linspace(500, 1000, rows),
        "best_bid": np.linspace(99, 149, rows),
        "best_ask": np.linspace(101, 151, rows),
        "bid_ask_spread": np.full(rows, 2.0),
        "depth_pct_1": np.linspace(10000, 20000, rows),
        "depth_pct_5": np.linspace(30000, 50000, rows),
        "fear_greed_score": np.linspace(40, 60, rows),
        "confidence": np.linspace(0.4, 0.8, rows),
    }
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def _cleanup(symbol: str = "BTCUSD"):
    yield
    for path_factory in (processed_output_path, scaler_output_path):
        path = path_factory(symbol)
        if path.exists():
            path.unlink()
        csv_fallback = path.with_suffix(".csv")
        if csv_fallback.exists():
            csv_fallback.unlink()


def test_build_feature_matrix_produces_scaled_features():
    frame = _sample_interim_frame()
    result = build_feature_matrix("BTCUSD", frame)

    assert not result.features.empty
    assert "timestamp" in result.features.columns
    numeric_columns = [col for col in result.features.columns if col != "timestamp"]
    assert numeric_columns, "Expected at least one engineered feature"

    scaler_path = scaler_output_path("BTCUSD")
    features_path = processed_output_path("BTCUSD")
    assert scaler_path.exists()
    assert features_path.exists() or features_path.with_suffix(".csv").exists()

    means = result.features[numeric_columns].mean().abs()
    assert (means < 1e-6).all(), "Scaled features should be centered around zero"


def test_build_feature_matrix_requires_timestamp():
    import pandas as pd

    frame = _sample_interim_frame().drop(columns=["timestamp"])
    with pytest.raises(KeyError):
        build_feature_matrix("BTCUSD", frame)
