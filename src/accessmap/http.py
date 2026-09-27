"""HTTP with timeouts, retries and exponential backoff, shared by all external calls."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

import requests

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "access-map/0.1 (accessibility research; https://github.com/)"

T = TypeVar("T")


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Exponential backoff with full jitter: attempt 0 -> up to 1 s, 1 -> 2 s, ..."""
    return random.uniform(0, min(cap, base * 2**attempt))


def with_retries(
    fn: Callable[[], T],
    *,
    retries: int = 4,
    is_retryable: Callable[[Exception], bool] = lambda e: True,
    sleep: Callable[[float], None] = time.sleep,
    what: str = "request",
) -> T:
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified below
            if attempt == retries or not is_retryable(e):
                raise
            delay = backoff_delay(attempt)
            log.warning("%s failed (%s); retry %d/%d in %.1fs",
                        what, e, attempt + 1, retries, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def _retryable_http(e: Exception) -> bool:
    if isinstance(e, HttpError):
        return e.status is None or e.status in RETRY_STATUS
    return isinstance(e, (requests.ConnectionError, requests.Timeout))


def request(
    method: str,
    url: str,
    *,
    timeout: float = 30,
    retries: int = 4,
    session: requests.Session | None = None,
    **kwargs,
) -> requests.Response:
    """Send a request; retry on connection errors, timeouts, 429 and 5xx."""
    s = session or requests
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}

    def once() -> requests.Response:
        r = s.request(method, url, timeout=timeout, headers=headers, **kwargs)
        if r.status_code >= 400:
            raise HttpError(f"HTTP {r.status_code}: {r.text[:300]}", r.status_code)
        return r

    return with_retries(once, retries=retries, is_retryable=_retryable_http, what=f"{method} {url}")


def get_json(url: str, **kwargs) -> dict:
    return request("GET", url, **kwargs).json()
