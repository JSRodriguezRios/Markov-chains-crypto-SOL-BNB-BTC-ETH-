"""Tests for the probability inference pipeline."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from qcsrc.pipeline import infer_proba
class DummyWrapper:
    def __init__(self, probabilities: np.ndarray) -> None:
        self._probabilities = probabilities

    def predict_proba(self, _: np.ndarray) -> np.ndarray:
        return self._probabilities


def test_infer_probabilities_uses_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    features = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"], utc=True
            ),
            "feature_a": [2.0, 4.0],
            "feature_b": [1.0, 3.0],
        }
    )

    scaler = SimpleNamespace(mean_=np.array([1.0, 1.0]), scale_=np.array([1.0, 2.0]))

    probabilities = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    dummy_wrapper = DummyWrapper(probabilities)

    exported_frames: dict[str, pd.DataFrame] = {}

    def fake_export(symbol: str, frame: pd.DataFrame) -> Path:
        exported_frames[symbol] = frame.copy()
        return Path(f"/tmp/{symbol.lower()}.csv")

    monkeypatch.setattr(infer_proba, "_load_scaler", lambda symbol: scaler)
    monkeypatch.setattr(infer_proba, "_load_features", lambda symbol, features=None: features)
    monkeypatch.setattr(infer_proba.GaussianHMMWrapper, "load", classmethod(lambda cls, path: dummy_wrapper))
    monkeypatch.setattr(infer_proba, "export_probability_timeseries", fake_export)

    result = infer_proba.infer_probabilities("BTCUSD")

    assert result.symbol == "BTCUSD"
    assert result.probability_path == Path("/tmp/btcusd.csv")
    assert not exported_frames["BTCUSD"].empty
    np.testing.assert_array_almost_equal(
        exported_frames["BTCUSD"]["prob_bearish"].to_numpy(), probabilities[:, 1]
    )

