"""HTTP helpers with retry/backoff support."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from ..util.logging_utils import create_logger

LOGGER = create_logger(__name__)


def request_with_backoff(
    client: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    max_attempts: int = 3,
) -> requests.Response:
    """Perform a GET request with exponential backoff on rate limits or server errors."""
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        response = client.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code in {429, 503} and attempt < max_attempts:
            LOGGER.warning(
                "Request to %s hit status %s. Backing off for %.1f seconds (attempt %s/%s)",
                url,
                response.status_code,
                delay,
                attempt,
                max_attempts,
            )
            time.sleep(delay)
            delay *= 2
            continue
        return response
    return response
