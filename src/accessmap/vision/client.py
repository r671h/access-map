"""Gemini client: cache, budget ledger, rate limit, retries, strict validation (SPEC §4)."""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from accessmap.http import with_retries
from accessmap.vision.prompt import render
from accessmap.vision.schema import response_model

log = logging.getLogger(__name__)

MAX_SIDE_PX = 1536
JPEG_QUALITY = 85
RETRY_CODES = {429, 500, 502, 503, 504}
THINKING_FALLBACK = {"MINIMAL": "LOW", "LOW": None}


class BudgetExceeded(RuntimeError):
    pass


class CallLedger:
    """Append-only log of every completed Gemini call; the budget is checked against it."""

    def __init__(self, path: Path, max_calls: int):
        self.path = path
        self.max_calls = max_calls
        self._lock = threading.Lock()
        self._count = self._read_count()

    def _read_count(self) -> int:
        if not self.path.is_file():
            return 0
        with open(self.path, encoding="utf-8") as f:
            return sum(json.loads(line).get("calls", 1) for line in f if line.strip())

    @property
    def used(self) -> int:
        return self._count

    def reserve(self) -> None:
        """Claim one call from the budget before sending it."""
        with self._lock:
            if self._count + 1 > self.max_calls:
                raise BudgetExceeded(
                    f"Gemini budget reached: {self._count}/{self.max_calls} calls used. "
                    "Ask the user before raising budget.max_gemini_calls.")
            self._count += 1

    def release(self) -> None:
        """Give back a reserved call that failed before producing a (billed) response."""
        with self._lock:
            self._count -= 1

    def record(self, entry: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": datetime.now(UTC).isoformat(timespec="seconds"),
                                    "calls": 1, **entry}) + "\n")


class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


def prepare_image(path: Path, max_side: int = MAX_SIDE_PX) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()


def _is_retryable(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code in RETRY_CODES:
        return True
    return isinstance(e, (ConnectionError, TimeoutError)) or "timed out" in str(e).lower()


class GeminiAnalyzer:
    def __init__(self, *, client, model: str, prompt_version: str, active_types: list[str],
                 cache_dir: Path, ledger: CallLedger, limiter: RateLimiter,
                 failed_log: Path, thinking_level: str | None = "MINIMAL"):
        self.client = client
        self.model = model
        self.prompt_version = prompt_version
        self.active_types = list(active_types)
        self.schema = response_model(tuple(active_types))
        self.cache_dir = cache_dir / model / prompt_version
        self.ledger = ledger
        self.limiter = limiter
        self.failed_log = failed_log
        self.thinking_level = thinking_level

    def cache_path(self, frame_id: str) -> Path:
        return self.cache_dir / f"{frame_id}.json"

    def _config(self):
        from google.genai import types

        kw = dict(temperature=0.1, response_mime_type="application/json",
                  response_schema=self.schema)
        if self.thinking_level:
            kw["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        return types.GenerateContentConfig(**kw)

    def _call(self, image: bytes, prompt: str, frame_id: str):
        from google.genai import types

        def once():
            self.ledger.reserve()
            self.limiter.wait()
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[types.Part.from_bytes(data=image, mime_type="image/jpeg"), prompt],
                    config=self._config(),
                )
            except Exception as e:
                if "thinking" in str(e).lower() and self.thinking_level:
                    # Some models don't support the lowest level; step up (MINIMAL -> LOW),
                    # and only drop the setting if even that is rejected.
                    nxt = THINKING_FALLBACK.get(self.thinking_level)
                    log.warning("%s rejects thinking_level=%s; using %s", self.model,
                                self.thinking_level, nxt)
                    self.thinking_level = nxt
                self.ledger.release()
                self.ledger.record({"model": self.model, "frame_id": frame_id, "ok": False,
                                    "calls": 0, "error": str(e)[:200]})
                raise
            u = resp.usage_metadata
            usage = {
                "prompt_tokens": getattr(u, "prompt_token_count", None),
                "output_tokens": getattr(u, "candidates_token_count", None),
                "thinking_tokens": getattr(u, "thoughts_token_count", None),
            }
            self.ledger.record({"model": self.model, "prompt_version": self.prompt_version,
                                "frame_id": frame_id, "ok": True, **usage})
            return resp, usage

        return with_retries(once, retries=6, is_retryable=lambda e: not isinstance(
            e, BudgetExceeded) and (_is_retryable(e) or "thinking" in str(e).lower()),
            what=f"gemini {self.model} {frame_id}")

    def analyze(self, frame: dict, root: Path) -> dict | None:
        """Analyse one frame (manifest row as dict). Returns the cache record or None."""
        cp = self.cache_path(frame["frame_id"])
        if cp.is_file():
            return json.loads(cp.read_text(encoding="utf-8"))

        captured = frame["captured_at"]
        prompt = render(self.prompt_version, active_types=self.active_types,
                        fov_deg=float(frame["fov"]),
                        captured_at=captured.strftime("%Y-%m-%d")
                        if hasattr(captured, "strftime") else str(captured)[:10])
        image = prepare_image(root / frame["path"])

        last_error, usages = None, []
        for _attempt in range(2):  # one retry on invalid output
            resp, usage = self._call(image, prompt, frame["frame_id"])
            usages.append(usage)
            try:
                parsed = self.schema.model_validate(json.loads(resp.text))
                break
            except (ValidationError, json.JSONDecodeError, TypeError) as e:
                last_error = str(e)[:500]
                log.warning("invalid response for %s: %s", frame["frame_id"], last_error[:120])
        else:
            with open(self.failed_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"frame_id": frame["frame_id"], "model": self.model,
                                    "prompt_version": self.prompt_version,
                                    "error": last_error}) + "\n")
            return None

        record = {
            "frame_id": frame["frame_id"],
            "model": self.model,
            "prompt_version": self.prompt_version,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "usage": usages,
            "response": parsed.model_dump(mode="json"),
        }
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".part")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cp)
        return record
