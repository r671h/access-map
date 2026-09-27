"""`accessmap check`: verify keys and API access for Gemini, Mapillary and Overpass."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field

from accessmap.config import Settings

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    service: str
    ok: bool
    detail: str
    extra: dict = field(default_factory=dict)


def _tiny_jpeg() -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 192), (150, 150, 150))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 120, 256, 192], fill=(90, 90, 90))  # "road"
    d.rectangle([0, 110, 256, 120], fill=(200, 200, 200))  # "curb"
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _explain_gemini_error(e: Exception) -> str:
    msg = str(e)
    if "API_KEY_INVALID" in msg or "API key not valid" in msg:
        return "GEMINI_API_KEY is invalid. Create a new key at aistudio.google.com (SETUP.md §2)."
    if "PERMISSION_DENIED" in msg:
        return f"Permission denied for this key/model: {msg[:200]}"
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return f"Quota/rate limit hit (key works, but limits are tight): {msg[:200]}"
    return msg[:300]


def check_gemini(settings: Settings) -> CheckResult:
    key = settings.secrets.gemini_api_key
    if not key:
        return CheckResult("Gemini", False, "GEMINI_API_KEY is not set in .env (see SETUP.md §2).")
    try:
        from google import genai
        from google.genai import types

        from accessmap.vision.schema import response_model

        client = genai.Client(api_key=key)
        models = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions:
                models.append(m.name.removeprefix("models/"))
        models.sort()

        wanted = settings.project.gemini.pilot_candidates or [m for m in models if "flash" in m][:1]
        available = [m for m in wanted if m in models]
        missing = [m for m in wanted if m not in models]
        if not available:
            return CheckResult("Gemini", False,
                               f"Key works but none of {wanted} are available.",
                               {"models": models})

        schema = response_model(tuple(settings.project.detection.types))
        tested = {}
        for model in available:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=_tiny_jpeg(), mime_type="image/jpeg"),
                    "This is a synthetic test image, not a street photo. Return JSON with "
                    "image_usable=false and an empty features list.",
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            schema.model_validate(json.loads(resp.text))
            usage = resp.usage_metadata
            tested[model] = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "thinking_tokens": getattr(usage, "thoughts_token_count", None),
            }
        detail = f"{len(models)} models; JSON-schema call OK on {', '.join(available)}"
        if missing:
            detail += f"; NOT available: {', '.join(missing)}"
        return CheckResult("Gemini", True, detail,
                           {"models": models, "tested": tested, "calls": len(available)})
    except Exception as e:  # noqa: BLE001 - reported to the user
        return CheckResult("Gemini", False, _explain_gemini_error(e))


def check_mapillary(settings: Settings) -> CheckResult:
    token = settings.secrets.mapillary_token
    if not token:
        return CheckResult("Mapillary", False,
                           "MAPILLARY_TOKEN is not set in .env (see SETUP.md §3).")
    if not token.startswith("MLY|"):
        return CheckResult("Mapillary", False,
                           "MAPILLARY_TOKEN should be the Client Token starting with 'MLY|' "
                           "(SETUP.md §3).")
    from accessmap.http import HttpError
    from accessmap.imagery.mapillary import COUNT_FIELDS, search_tile, tiles

    try:
        first = next(tiles(settings.project.area.bbox))
        data = search_tile(token, first, fields=COUNT_FIELDS, limit=1)
        return CheckResult("Mapillary", True,
                           f"token OK; {'found an image' if data else 'no image'} in first tile")
    except HttpError as e:
        if e.status in (401, 403):
            return CheckResult("Mapillary", False,
                               "MAPILLARY_TOKEN rejected (401/403). Copy the Client Token from "
                               "the developer dashboard (SETUP.md §3).")
        return CheckResult("Mapillary", False, str(e)[:300])
    except Exception as e:  # noqa: BLE001
        return CheckResult("Mapillary", False, str(e)[:300])


def check_overpass(settings: Settings) -> CheckResult:
    from accessmap.osm.overpass import count

    try:
        n = count(settings.project.area.bbox, {"highway_ways": 'way["highway"]'})
        return CheckResult("Overpass", True, f"{n['highway_ways']} highway ways in bbox")
    except Exception as e:  # noqa: BLE001
        return CheckResult("Overpass", False, str(e)[:300])


def run_checks(settings: Settings) -> list[CheckResult]:
    return [check_gemini(settings), check_mapillary(settings), check_overpass(settings)]


def format_table(results: list[CheckResult]) -> str:
    w = max(len(r.service) for r in results)
    lines = [f"{'service'.ljust(w)}  status  detail", f"{'-' * w}  ------  ------"]
    for r in results:
        lines.append(f"{r.service.ljust(w)}  {'OK  ' if r.ok else 'FAIL'}    {r.detail}")
    return "\n".join(lines)
