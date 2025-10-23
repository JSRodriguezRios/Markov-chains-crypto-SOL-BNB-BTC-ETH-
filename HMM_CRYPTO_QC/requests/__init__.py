from __future__ import annotations

from .api import request
from .exceptions import HTTPError
from .sessions import Session

__all__ = ["Session", "request", "HTTPError"]


def get(url: str, **kwargs):
    return request("GET", url, **kwargs)
