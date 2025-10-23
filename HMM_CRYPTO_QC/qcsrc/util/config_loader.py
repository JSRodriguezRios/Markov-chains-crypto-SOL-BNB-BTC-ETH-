"""YAML configuration loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .logging_utils import create_logger

LOGGER = create_logger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml_config(relative_path: str) -> Dict[str, Any]:
    """Load a YAML config file located under the project root."""

    config_path = _project_root() / relative_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    with config_path.open("r", encoding="utf-8") as handle:
        LOGGER.debug("Loading configuration from %s", config_path)
        return yaml.safe_load(handle)


def load_config(relative_path: str, *, key: Optional[str] = None, default: Any | None = None) -> Any:
    """Load a YAML config file and optionally return a nested key."""

    payload = load_yaml_config(relative_path)
    if key is None:
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Config {relative_path} must be a mapping to select key '{key}'")
    return payload.get(key, default)


def load_settings_config() -> Dict[str, Any]:
    """Return the global strategy settings configuration."""

    return load_yaml_config("config/settings.yaml")


def load_assets_config() -> Dict[str, Any]:
    return load_config("config/assets.yaml", key="assets", default={})


def load_schedule_config() -> Dict[str, Any]:
    return load_yaml_config("config/schedule.yaml")


def load_features_config() -> Dict[str, Any]:
    """Return the feature engineering configuration dictionary."""

    return load_yaml_config("config/features.yaml")
