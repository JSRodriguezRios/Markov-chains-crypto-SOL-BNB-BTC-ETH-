from __future__ import annotations

from typing import Any


class Session:
    """Minimal session object that can be patched by tests."""

    def request(self, method: str, url: str, **kwargs: Any):  # pragma: no cover - overridden in tests
        raise NotImplementedError("Network requests are not supported in the stub.")

    def get(self, url: str, **kwargs: Any):
        return self.request("GET", url, **kwargs)


__all__ = ["Session"]
