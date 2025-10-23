from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

try:  # pragma: no cover - dependency guard
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    from hmmlearn.hmm import GaussianHMM  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip(
        "hmmlearn, numpy, and pandas are required for model tests",
        allow_module_level=True,
    )

from qcsrc.models.hmm_model import GaussianHMMWrapper
from qcsrc.models.state_mapper import map_hidden_states
from qcsrc.models.thresholds import should_flat, should_long, should_short


def _synthetic_dataset() -> tuple["np.ndarray", list[str], "np.ndarray"]:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=300, freq="H")
    returns = np.concatenate([
        rng.normal(0.005, 0.002, 100),
        rng.normal(0.0, 0.001, 100),
        rng.normal(-0.004, 0.002, 100),
    ])
    volatility = np.abs(rng.normal(0.02, 0.005, 300))
    sentiment = rng.normal(0.0, 1.0, 300)
    data = np.column_stack([returns, volatility, sentiment])
    df = pd.DataFrame(data, columns=["return_log_1h", "volatility_24h", "sentiment_zscore"])
    df["timestamp"] = timestamps
    return df.drop(columns=["timestamp"]).to_numpy(), df.columns[:-1].tolist(), returns


def test_hmm_wrapper_trains_and_serializes(tmp_path: Path) -> None:
    features, feature_names, _ = _synthetic_dataset()
    wrapper = GaussianHMMWrapper(n_components=3, random_state=7, n_iter=150)
    summary = wrapper.fit(features)

    assert summary.log_likelihood != 0.0
    assert wrapper.is_fitted()

    proba = wrapper.predict_proba(features[:5])
    assert proba.shape == (5, 3)

    model_path = tmp_path / "btc_hmm.pkl"
    wrapper.save(model_path)
    assert model_path.exists()

    restored = GaussianHMMWrapper.load(model_path)
    restored_proba = restored.predict_proba(features[:5])
    np.testing.assert_allclose(proba, restored_proba, atol=1e-6)


def test_state_mapping_orders_states_by_return() -> None:
    features, feature_names, raw_returns = _synthetic_dataset()
    wrapper = GaussianHMMWrapper(n_components=3, random_state=7, n_iter=200)
    wrapper.fit(features)

    class DummyScaler:
        mean_ = np.array([raw_returns.mean(), 0.0, 0.0])
        scale_ = np.array([raw_returns.std(ddof=0), 1.0, 1.0])

    mapping = map_hidden_states(
        wrapper.model,
        feature_names,
        scaler_mean=DummyScaler.mean_,
        scaler_scale=DummyScaler.scale_,
    )

    assert set(mapping.values()) == {"bullish", "bearish", "consolidation"}

    proba = wrapper.predict_proba(features)
    dominant_states = np.argmax(proba, axis=1)
    mapped_labels = [mapping[int(state)] for state in dominant_states]
    assert "bullish" in mapped_labels
    assert "bearish" in mapped_labels


def test_threshold_helpers_apply_expected_logic() -> None:
    probabilities = {"bullish": 0.75, "bearish": 0.1, "consolidation": 0.15}
    assert should_long(probabilities)
    assert not should_short(probabilities)
    assert not should_flat(probabilities)

    bearish_probs = {"bullish": 0.1, "bearish": 0.72, "consolidation": 0.18}
    assert should_short(bearish_probs)
    assert not should_long(bearish_probs)

    consolidation_probs = {"bullish": 0.2, "bearish": 0.1, "consolidation": 0.71}
    assert should_flat(consolidation_probs)
