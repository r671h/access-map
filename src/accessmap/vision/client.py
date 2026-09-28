"""Gemini client: fingerprinted cache, budget ledger, rate limit, retries, strict validation
(SPEC §4). LocalAnalyzer runs the same prompt/schema/cache on a local Ollama model."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pydantic import ValidationError

from accessmap.http import with_retries
from accessmap.vision.prompt import render
from accessmap.vision.schema import response_model

log = logging.getLogger(__name__)

MAX_SIDE_PX = 1536
JPEG_QUALITY = 85
RETRY_CODES = {429, 500, 502, 503, 504}
TEMPERATURE = 0.1
THINKING_FALLBACK = {"MINIMAL": "LOW", "LOW": None}
LOCAL_PREFIX = "ollama:"


def is_local(model: str) -> bool:
    return model.startswith(LOCAL_PREFIX)


def model_dir(model: str) -> str:
    """Folder/file-safe model name ("ollama:qwen3-vl:8b" -> "ollama_qwen3-vl_8b")."""
    return model.replace(":", "_").replace("/", "_")


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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        self.cache_dir = cache_dir / model_dir(model) / prompt_version
        self.stale_dir = cache_dir / "_stale"
        self.schema_sha = _sha(json.dumps(self.schema.model_json_schema(),
                                          sort_keys=True).encode())
        self.requested_thinking = thinking_level
        self.cache_stats: Counter = Counter()
        self._stats_lock = threading.Lock()
        self.ledger = ledger
        self.limiter = limiter
        self.failed_log = failed_log
        self.thinking_level = thinking_level

    def cache_path(self, frame_id: str) -> Path:
        return self.cache_dir / f"{frame_id}.json"

    def _config(self):
        from google.genai import types

        kw = dict(temperature=TEMPERATURE, response_mime_type="application/json",
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

    def _inputs(self, frame: dict, root: Path) -> tuple[str, Path, dict]:
        """Rendered prompt, image path and the fingerprint of everything that shapes the answer.

        The image is hashed as the source file plus the resize/JPEG settings, not the
        re-encoded bytes, so a Pillow upgrade cannot invalidate the whole cache. The thinking
        level is the configured one, not the runtime fallback, so a rejected MINIMAL does not
        make every later run look stale.
        """
        captured = frame["captured_at"]
        prompt = render(self.prompt_version, active_types=self.active_types,
                        fov_deg=float(frame["fov"]),
                        captured_at=captured.strftime("%Y-%m-%d")
                        if hasattr(captured, "strftime") else str(captured)[:10])
        image_path = root / frame["path"]
        inputs = {
            "model": self.model,
            "prompt_sha": _sha(prompt.encode("utf-8")),
            "image_sha": _sha(image_path.read_bytes()),
            "image_prep": f"{MAX_SIDE_PX}px/q{JPEG_QUALITY}",
            "schema_sha": self.schema_sha,
            "thinking": self.requested_thinking,
            "temperature": TEMPERATURE,
        }
        return prompt, image_path, inputs

    def lookup(self, frame: dict, root: Path) -> tuple[str, dict | None, tuple]:
        """Cache state of a frame without calling the API.

        Returns (state, record, inputs): state is hit | failed | stale | miss. A record
        written before fingerprints existed is adopted (fingerprint backfilled).
        """
        prompt, image_path, inputs = self._inputs(frame, root)
        fp = _sha(json.dumps(inputs, sort_keys=True).encode())
        cp = self.cache_path(frame["frame_id"])
        if not cp.is_file():
            return "miss", None, (prompt, image_path, inputs, fp)
        rec = json.loads(cp.read_text(encoding="utf-8"))
        if "fingerprint" not in rec:
            rec |= {"fingerprint": fp, "inputs": inputs, "fingerprint_backfilled": True}
            self._write(cp, rec)
        if rec["fingerprint"] != fp:
            return "stale", rec, (prompt, image_path, inputs, fp)
        state = "failed" if rec.get("response") is None else "hit"
        return state, rec, (prompt, image_path, inputs, fp)

    def _write(self, cp: Path, record: dict) -> None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".part")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cp)

    def _retire(self, cp: Path) -> None:
        """Move a stale record aside instead of deleting it (it was paid for)."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        dest = (self.stale_dir / model_dir(self.model) / self.prompt_version
                / f"{cp.stem}.{stamp}.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cp.replace(dest)
        log.warning("cache for %s is stale (inputs changed); moved to %s", cp.stem, dest)

    def _count(self, state: str) -> None:
        with self._stats_lock:
            self.cache_stats[state] += 1

    def analyze(self, frame: dict, root: Path, retry_failed: bool = False) -> dict | None:
        """Analyse one frame (manifest row as dict). Returns the cache record, or None when
        the model gave no valid answer (that outcome is cached too, see retry_failed)."""
        state, rec, (prompt, image_path, inputs, fp) = self.lookup(frame, root)
        if state == "hit" or (state == "failed" and not retry_failed):
            self._count(state)
            return rec if state == "hit" else None
        cp = self.cache_path(frame["frame_id"])
        if state == "stale":
            self._retire(cp)
        self._count("stale" if state == "stale" else "retried" if state == "failed" else "miss")

        image = prepare_image(image_path)
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
            parsed = None
            with open(self.failed_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"frame_id": frame["frame_id"], "model": self.model,
                                    "prompt_version": self.prompt_version,
                                    "error": last_error}) + "\n")

        record = {
            "frame_id": frame["frame_id"],
            "model": self.model,
            "prompt_version": self.prompt_version,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fingerprint": fp,
            "inputs": inputs,
            "thinking_used": self.thinking_level,
            "usage": usages,
            "response": parsed.model_dump(mode="json") if parsed else None,
        }
        if parsed is None:
            record["error"] = last_error
        self._write(cp, record)
        return record if parsed else None


class LocalAnalyzer(GeminiAnalyzer):
    """Same prompt, schema, cache and validation as GeminiAnalyzer, but the call goes to a
    local Ollama server (/api/chat with the JSON schema as `format`). Local calls cost
    nothing, so the ledger here is a separate log, not the Gemini budget."""

    # Small models under a JSON grammar can loop (the same curb 26 times with the box
    # shifted a few px each time), so the local grammar caps the list and the output length.
    # num_ctx 4096 keeps an 8B Q4 model fully on an 8 GB GPU (8192 spilled 20% to the CPU).
    MAX_FEATURES = 8
    OPTIONS = {"temperature": TEMPERATURE, "num_ctx": 4096, "num_predict": 1500}

    def __init__(self, *, host: str, timeout_s: float = 600, **kw):
        super().__init__(client=None, thinking_level=None, **kw)
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.format = self.schema.model_json_schema()
        self.format["properties"]["features"]["maxItems"] = self.MAX_FEATURES

    def _inputs(self, frame: dict, root: Path) -> tuple[str, Path, dict]:
        prompt, image_path, inputs = super()._inputs(frame, root)
        inputs["local"] = _sha(json.dumps({"format": self.format, "options": self.OPTIONS},
                                          sort_keys=True).encode())
        return prompt, image_path, inputs

    def _call(self, image: bytes, prompt: str, frame_id: str):
        import requests

        body = {
            "model": self.model.removeprefix(LOCAL_PREFIX),
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(image).decode("ascii")]}],
            "format": self.format,
            "stream": False,
            "think": False,
            "options": self.OPTIONS,
        }

        def once():
            self.ledger.reserve()
            try:
                r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_s)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                self.ledger.release()
                self.ledger.record({"model": self.model, "frame_id": frame_id, "ok": False,
                                    "calls": 0, "error": str(e)[:200]})
                raise
            usage = {
                "prompt_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
                "thinking_tokens": None,
                "seconds": round((data.get("total_duration") or 0) / 1e9, 2),
            }
            self.ledger.record({"model": self.model, "prompt_version": self.prompt_version,
                                "frame_id": frame_id, "ok": True, **usage})
            return SimpleNamespace(text=data["message"]["content"]), usage

        def retryable(e: Exception) -> bool:
            if isinstance(e, BudgetExceeded):
                return False
            if isinstance(e, requests.HTTPError):
                return e.response is not None and e.response.status_code in RETRY_CODES
            return isinstance(e, (requests.ConnectionError, requests.Timeout))

        return with_retries(once, retries=3, is_retryable=retryable,
                            what=f"ollama {self.model} {frame_id}")
