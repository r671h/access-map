"""`accessmap analyze` (phase 3): run Gemini over frames, write detections, pilot QA."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from accessmap.config import Settings
from accessmap.vision.client import (
    BudgetExceeded,
    CallLedger,
    GeminiAnalyzer,
    RateLimiter,
)

log = logging.getLogger(__name__)

PILOT_SEED = 7
DEFAULT_PROMPT = "v1"

# USD per 1M tokens, paid tier (ai.google.dev/gemini-api/docs/pricing, read 2026-09-27;
# Flash prices are introductory until 2026-12-31). Thinking tokens bill as output.
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}


def ledger_for(settings: Settings) -> CallLedger:
    return CallLedger(settings.paths.cache / "gemini_calls.jsonl",
                      settings.project.budget.max_gemini_calls)


def load_frames(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(settings.paths.processed / "frames.parquet")


def pilot_frame_ids(frames: pd.DataFrame, n: int = 30, seed: int = PILOT_SEED) -> list[str]:
    """Deterministic sample, identical for every model, with some panorama crops in it."""
    rng = random.Random(seed)
    ids = sorted(frames.frame_id)
    pano = sorted(frames.loc[frames.is_pano, "frame_id"])
    n_pano = min(len(pano), max(1, n // 6)) if pano else 0
    chosen = rng.sample(pano, n_pano) if n_pano else []
    rest = [i for i in ids if i not in set(chosen)]
    chosen += rng.sample(rest, min(n - n_pano, len(rest)))
    return sorted(chosen)


def make_analyzer(settings: Settings, model: str, prompt_version: str) -> GeminiAnalyzer:
    from google import genai

    key = settings.secrets.gemini_api_key
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (SETUP.md §2)")
    g = settings.project.gemini
    return GeminiAnalyzer(
        client=genai.Client(api_key=key),
        model=model,
        prompt_version=prompt_version,
        active_types=settings.project.detection.types,
        cache_dir=settings.paths.cache / "gemini",
        ledger=ledger_for(settings),
        limiter=RateLimiter(g.rpm),
        failed_log=settings.paths.cache / "failed.jsonl",
        thinking_level=(g.thinking or "minimal").upper(),
    )


def detection_rows(record: dict, frame: dict) -> dict:
    r = record["response"]
    return {
        "frame_id": frame["frame_id"], "image_id": frame["image_id"],
        "model": record["model"], "prompt_version": record["prompt_version"],
        "lon": frame["lon"], "lat": frame["lat"], "heading": frame["heading"],
        "fov": frame["fov"], "captured_at": str(frame["captured_at"]),
        "is_pano": bool(frame["is_pano"]), "path": frame["path"],
        "image_usable": r["image_usable"], "sidewalk_visible": r["sidewalk_visible"],
        "sidewalk_surface": r["sidewalk_surface"], "features": r["features"],
    }


def run(settings: Settings, model: str, frame_ids: list[str] | None,
        prompt_version: str = DEFAULT_PROMPT, out: Path | None = None,
        workers: int = 4) -> dict:
    from tqdm import tqdm

    frames = load_frames(settings)
    if frame_ids is not None:
        frames = frames[frames.frame_id.isin(set(frame_ids))]
    analyzer = make_analyzer(settings, model, prompt_version)
    used_before = analyzer.ledger.used
    to_call = sum(not analyzer.cache_path(f).is_file() for f in frames.frame_id)
    remaining = analyzer.ledger.max_calls - used_before
    if to_call > remaining:
        raise BudgetExceeded(f"{to_call} uncached frames but only {remaining} calls left "
                             f"({used_before}/{analyzer.ledger.max_calls} used)")
    log.info("%s %s: %d frames, %d uncached, budget %d/%d used", model, prompt_version,
             len(frames), to_call, used_before, analyzer.ledger.max_calls)

    rows = frames.to_dict("records")
    results: dict[str, dict | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(analyzer.analyze, row, settings.root): row["frame_id"]
                   for row in rows}
        for f in tqdm(futures, desc=f"analyze {model}", unit="frame"):
            fid = futures[f]
            try:
                results[fid] = f.result()
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001 - logged; counted as invalid
                log.error("frame %s failed: %s", fid, e)
                results[fid] = None

    out = out or settings.paths.processed / "detections.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    by_id = {r["frame_id"]: r for r in rows}
    with open(out, "w", encoding="utf-8") as f:
        for fid in sorted(results):
            if results[fid] is not None:
                f.write(json.dumps(detection_rows(results[fid], by_id[fid]),
                                   ensure_ascii=False) + "\n")

    return summarize(results, model, prompt_version, analyzer.ledger.used - used_before,
                     analyzer.ledger.used, analyzer.ledger.max_calls, len(load_frames(settings)))


def summarize(results: dict, model: str, prompt_version: str, calls_this_run: int,
              calls_total: int, budget: int, n_all_frames: int) -> dict:
    valid = [r for r in results.values() if r is not None]
    usable = [r for r in valid if r["response"]["image_usable"]]
    feats = [f for r in valid for f in r["response"]["features"]]
    usages = [u for r in valid for u in r["usage"]]

    def mean(key):
        vals = [u.get(key) or 0 for u in usages]
        return sum(vals) / len(vals) if vals else 0.0

    tokens_in, tokens_out = mean("prompt_tokens"), mean("output_tokens") + mean("thinking_tokens")
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    per_call = (tokens_in * price_in + tokens_out * price_out) / 1e6
    return {
        "model": model,
        "prompt_version": prompt_version,
        "frames": len(results),
        "valid": len(valid),
        "valid_share": round(len(valid) / len(results), 3) if results else 0.0,
        "usable": len(usable),
        "features": len(feats),
        "features_by_type": dict(Counter(f["type"] for f in feats).most_common()),
        "mean_confidence": round(sum(f["confidence"] for f in feats) / len(feats), 2)
        if feats else None,
        "tokens_per_call": {"input": round(tokens_in), "output_incl_thinking": round(tokens_out)},
        "usd_per_call_paid": round(per_call, 5),
        "projected_usd_all_frames_paid": round(per_call * n_all_frames, 2),
        "calls_this_run": calls_this_run,
        "calls_used_total": calls_total,
        "budget": budget,
    }


def compare_sheets(settings: Settings, models: list[str], frame_ids: list[str],
                   prompt_version: str = DEFAULT_PROMPT, per_sheet: int = 6) -> list[Path]:
    """Side-by-side sheets: each row = one frame, one tile per model, boxes drawn."""
    from accessmap.eval.contact import Box, Tile, contact_sheet

    frames = load_frames(settings).set_index("frame_id")
    cache = settings.paths.cache / "gemini"
    tiles_per_frame = []
    for fid in frame_ids:
        row = []
        for m in models:
            cp = cache / m / prompt_version / f"{fid}.json"
            rec = json.loads(cp.read_text(encoding="utf-8")) if cp.is_file() else None
            resp = rec["response"] if rec else None
            if resp is None:
                caption = f"{m}: no valid response"
                boxes = []
            else:
                boxes = [Box(f["box_2d"], f"{f['type']} {f['confidence']:.2f}", f["type"])
                         for f in resp["features"]]
                usable = "" if resp["image_usable"] else " UNUSABLE"
                caption = f"{m.removeprefix('gemini-')} · {fid[-10:]} · {len(boxes)} feat{usable}"
            row.append(Tile(settings.root / frames.loc[fid, "path"], caption, boxes))
        tiles_per_frame.append(row)
    # Frames with the most detections first, so the sheets show the interesting cases.
    tiles_per_frame.sort(key=lambda r: -sum(len(t.boxes) for t in r))
    out = []
    cols = 2 * len(models) if len(models) <= 2 else len(models)
    for i in range(0, len(tiles_per_frame), per_sheet):
        chunk = [t for row in tiles_per_frame[i:i + per_sheet] for t in row]
        path = settings.paths.qa / f"pilot_compare_{i // per_sheet + 1}.jpg"
        contact_sheet(chunk, path, cols=cols,
                      title=f"Pilot {prompt_version}: " + " | ".join(models)
                            + f" (sheet {i // per_sheet + 1})")
        out.append(path)
    return out


def type_sheets(settings: Settings, per_type: int = 12, seed: int = PILOT_SEED) -> list[Path]:
    """One sheet per barrier type from detections.jsonl: a random sample of frames with that
    type (not the most confident ones, so the sheet shows typical quality). Only boxes of
    the sheet's type are drawn."""
    from accessmap.eval.contact import Box, Tile, contact_sheet

    path = settings.paths.processed / "detections.jsonl"
    with open(path, encoding="utf-8") as f:
        frames = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(seed)
    out = []
    for btype in settings.project.detection.types:
        hits = [fr for fr in frames if any(ft["type"] == btype for ft in fr["features"])]
        if not hits:
            continue
        sample = rng.sample(hits, min(per_type, len(hits)))
        tiles = []
        for fr in sample:
            boxes = [Box(ft["box_2d"], f"{ft['type']} {ft['confidence']:.2f}", ft["type"])
                     for ft in fr["features"] if ft["type"] == btype]
            d = next(ft for ft in fr["features"] if ft["type"] == btype)
            dist = d.get("estimated_distance_m")
            caption = f"{fr['frame_id'][-10:]} · {len(boxes)}× · " + (
                f"{dist:.0f} m · " if dist else "") + d["description_en"]
            tiles.append(Tile(settings.root / fr["path"], caption, boxes))
        p = settings.paths.qa / f"types_{btype}.jpg"
        contact_sheet(tiles, p, cols=4,
                      title=f"{btype}: {len(sample)} of {len(hits)} frames "
                            f"({fr['model']}, prompt {fr['prompt_version']})")
        out.append(p)
    return out
