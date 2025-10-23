"""Model utilities for the HMM crypto project."""

try:  # pragma: no cover - optional dependency guard for local testing
    from .hmm_model import GaussianHMMWrapper  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - allow importing without numpy/hmmlearn
    GaussianHMMWrapper = None  # type: ignore[misc]

try:  # pragma: no cover
    from .state_mapper import map_hidden_states  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    map_hidden_states = None  # type: ignore[misc]

from .thresholds import should_flat, should_long, should_short  # noqa: F401

__all__ = [
    "GaussianHMMWrapper",
    "map_hidden_states",
    "should_flat",
    "should_long",
    "should_short",
]
