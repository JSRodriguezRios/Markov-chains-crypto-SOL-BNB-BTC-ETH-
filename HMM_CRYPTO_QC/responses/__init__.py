from __future__ import annotations

import json as _json
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import requests

GET = "GET"


class _MockResponse:
    def __init__(self, *, status: int, payload: Optional[Dict[str, Any]] = None) -> None:
        self.status_code = status
        self._payload = payload or {}
        self.text = _json.dumps(self._payload)

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            raise requests.HTTPError(f"Status code: {self.status_code}")


class _ResponsesState:
    def __init__(self) -> None:
        self.queue: List[Dict[str, Any]] = []

    def add(self, method: str, url: str, *, json: Optional[Dict[str, Any]] = None, status: int = 200) -> None:
        self.queue.append({"method": method, "url": url, "json": json, "status": status})

    def request(self, method: str, url: str, **kwargs: Any) -> _MockResponse:
        if not self.queue:
            raise AssertionError("No mocked responses available")
        expected = self.queue.pop(0)
        if method.upper() != expected["method"] or url != expected["url"]:
            raise AssertionError(f"Unexpected request {method} {url}")
        return _MockResponse(status=expected["status"], payload=expected["json"])


_state_local = threading.local()


def _get_state() -> _ResponsesState:
    state = getattr(_state_local, "state", None)
    if state is None:
        raise RuntimeError("responses.activate must wrap the test function")
    return state


def add(method: str, url: str, json: Optional[Dict[str, Any]] = None, status: int = 200) -> None:
    _get_state().add(method, url, json=json, status=status)


def activate(func):
    def wrapper(*args: Any, **kwargs: Any):
        state = _ResponsesState()
        _state_local.state = state
        with patch("requests.sessions.Session.request", side_effect=state.request), patch(
            "requests.api.request", side_effect=state.request
        ):
            try:
                return func(*args, **kwargs)
            finally:
                _state_local.state = None
    return wrapper
