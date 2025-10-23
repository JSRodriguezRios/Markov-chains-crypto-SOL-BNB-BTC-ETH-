"""Secret management utilities."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from .logging_utils import create_logger

LOGGER = create_logger(__name__)

CREDENTIALS_ENV_MAP = {
    "cryptoquant": "CRYPTOQUANT_API_KEY",
    "coinstats": "COINSTATS_API_KEY",
    "binance": "BINANCE_API_KEY",
}


class SecretNotFoundError(RuntimeError):
    """Raised when a requested secret is missing."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _credentials_path() -> Path:
    return _project_root() / "secrets" / "credentials.json"


def _load_credentials_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        LOGGER.debug("Credentials file %s not found; falling back to environment.", path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_credentials() -> Dict[str, str]:
    """Return a provider -> api_key mapping."""
    credentials = {key: os.getenv(env_key, "") for key, env_key in CREDENTIALS_ENV_MAP.items()}
    if all(value for value in credentials.values()):
        return credentials

    file_credentials = _load_credentials_file(_credentials_path())
    for provider, env_key in CREDENTIALS_ENV_MAP.items():
        if credentials[provider]:
            continue
        credentials[provider] = file_credentials.get(provider, "")
    return credentials


def get_api_key(provider: str) -> str:
    """Return a specific provider API key or raise an error."""
    normalized = provider.lower()
    if normalized not in CREDENTIALS_ENV_MAP:
        raise ValueError(f"Unsupported provider '{provider}'.")

    credentials = load_credentials()
    api_key = credentials.get(normalized, "")
    if api_key:
        return api_key

    raise SecretNotFoundError(
        f"API key for provider '{provider}' not found in environment or secrets/credentials.json"
    )
