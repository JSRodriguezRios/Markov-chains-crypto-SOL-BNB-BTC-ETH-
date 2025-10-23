"""Gaussian HMM wrapper utilities."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..util.logging_utils import create_logger, log_phase_checkpoint

try:  # pragma: no cover - optional dependency handling
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    GaussianHMM = None  # type: ignore[misc]

LOGGER = create_logger(__name__)


@dataclass(frozen=True)
class HMMFitSummary:
    """Container describing training diagnostics for an HMM fit."""

    log_likelihood: float
    aic: float
    bic: float


def _ensure_model_available() -> None:
    if GaussianHMM is None:  # pragma: no cover - defensive check
        raise ImportError(
            "hmmlearn is required to use GaussianHMMWrapper. Install hmmlearn before "
            "running the training pipeline."
        )


def _ensure_two_dimensional(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Input array must be two dimensional for HMM operations.")
    return matrix


class GaussianHMMWrapper:
    """Thin convenience layer around ``hmmlearn``'s GaussianHMM."""

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        _ensure_model_available()
        self._model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )

    @property
    def model(self) -> "GaussianHMM":
        return self._model

    def fit(self, observations: np.ndarray) -> HMMFitSummary:
        data = _ensure_two_dimensional(observations)
        self._model.fit(data)
        log_likelihood = float(self._model.score(data))
        num_params = self._model._get_n_free_parameters()  # pylint: disable=protected-access
        num_samples = data.shape[0]
        aic = 2 * num_params - 2 * log_likelihood
        bic = num_params * np.log(num_samples) - 2 * log_likelihood
        log_phase_checkpoint(LOGGER, "Trained GaussianHMM model")
        return HMMFitSummary(log_likelihood=log_likelihood, aic=aic, bic=bic)

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        data = _ensure_two_dimensional(observations)
        return self._model.predict_proba(data)

    def score(self, observations: np.ndarray) -> float:
        data = _ensure_two_dimensional(observations)
        return float(self._model.score(data))

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self._model, handle)
        LOGGER.info("Persisted HMM model to %s", destination)
        return destination

    @classmethod
    def load(cls, path: Path | str) -> "GaussianHMMWrapper":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Model file {source} not found.")
        with source.open("rb") as handle:
            model = pickle.load(handle)
        wrapper = cls.__new__(cls)  # type: ignore
        setattr(wrapper, "_model", model)
        return wrapper

    def state_means(self, feature_index: int) -> np.ndarray:
        if feature_index < 0 or feature_index >= self._model.means_.shape[1]:
            raise IndexError("Feature index out of bounds for model means.")
        return self._model.means_[:, feature_index]

    def state_sequence(self, observations: np.ndarray) -> Sequence[int]:
        data = _ensure_two_dimensional(observations)
        return self._model.predict(data)

    def is_fitted(self) -> bool:
        return hasattr(self._model, "means_")
