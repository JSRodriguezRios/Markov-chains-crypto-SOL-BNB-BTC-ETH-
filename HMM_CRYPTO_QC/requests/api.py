from __future__ import annotations

from typing import Any

from .sessions import Session


def request(method: str, url: str, **kwargs: Any):
    return Session().request(method, url, **kwargs)
