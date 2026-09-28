"""FastAPI app (SPEC §9): barriers, network, frames/photos, stats, feedback.

Routing (`/api/route`) is added in phase 6. Run with `accessmap serve`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from accessmap.config import Settings

STATIC = Path(__file__).parent / "static"


class Feedback(BaseModel):
    barrier_id: str = Field(min_length=1, max_length=64)
    verdict: Literal["confirm", "reject"]
    comment: str = Field(default="", max_length=500)


class _FileCache:
    """Re-reads a JSON file only when its mtime changes (pipeline reruns show up live)."""

    def __init__(self, path: Path, empty):
        self.path, self.empty = path, empty
        self._mtime, self._data = None, empty
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if not self.path.is_file():
                return self.empty
            m = self.path.stat().st_mtime
            if m != self._mtime:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                self._mtime = m
            return self._data


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, barrier_id TEXT NOT NULL,
                verdict TEXT NOT NULL CHECK (verdict IN ('confirm', 'reject')),
                comment TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL)""")

    def _conn(self):
        return sqlite3.connect(self.path)

    def add(self, fb: Feedback) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO feedback (barrier_id, verdict, comment, created_at) "
                            "VALUES (?, ?, ?, ?)",
                            (fb.barrier_id, fb.verdict, fb.comment.strip(),
                             datetime.now(UTC).isoformat(timespec="seconds")))
            return cur.lastrowid

    def summary(self) -> dict[str, dict]:
        """barrier_id -> {confirm, reject, status}; rejected when rejects outnumber confirms."""
        with self._conn() as c:
            rows = c.execute("SELECT barrier_id, verdict, COUNT(*) FROM feedback "
                             "GROUP BY barrier_id, verdict").fetchall()
        out: dict[str, dict] = {}
        for bid, verdict, n in rows:
            out.setdefault(bid, {"confirm": 0, "reject": 0})[verdict] = n
        for s in out.values():
            s["status"] = ("rejected" if s["reject"] > s["confirm"] else
                           "confirmed" if s["confirm"] > s["reject"] else "disputed")
        return out


def create_app(settings: Settings) -> FastAPI:
    proc = settings.paths.processed
    empty_fc = {"type": "FeatureCollection", "features": []}
    snapped = _FileCache(proc / "barriers.geojson", empty_fc)
    unsnapped = _FileCache(proc / "barriers_unsnapped.geojson", empty_fc)
    network = _FileCache(settings.paths.raw / "osm" / "walk_edges.geojson", empty_fc)
    metrics = _FileCache(settings.paths.reports / "metrics.json", {})
    feedback = FeedbackStore(settings.paths.data / "feedback.sqlite")
    frames_cache: dict = {}

    def frames() -> dict[str, dict]:
        path = proc / "frames.parquet"
        if not path.is_file():
            return {}
        m = path.stat().st_mtime
        if frames_cache.get("mtime") != m:
            import pandas as pd

            df = pd.read_parquet(path)
            frames_cache.update(mtime=m, data={r["frame_id"]: r for r in df.to_dict("records")})
        return frames_cache["data"]

    def all_barriers() -> list[dict]:
        fb = feedback.summary()
        out = []
        for fc in (snapped.get(), unsnapped.get()):
            for f in fc["features"]:
                p = f["properties"]
                out.append({**f, "properties": {**p, "feedback": fb.get(p["id"])}})
        return out

    app = FastAPI(title="access-map", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config():
        p = settings.project
        return {"area": {"name": p.area.name, "bbox": p.area.bbox},
                "languages": p.ui.languages, "language": p.ui.language,
                "types": p.detection.types, "profiles": p.profiles,
                "features": p.app.features, "routing": False}

    @app.get("/api/barriers")
    def barriers(types: str | None = Query(None, description="comma-separated types"),
                 min_conf: float = Query(0.0, ge=0, le=1),
                 include_unsnapped: bool = True, include_rejected: bool = True):
        wanted = set(types.split(",")) if types else None
        feats = []
        for f in all_barriers():
            p = f["properties"]
            if wanted is not None and p["type"] not in wanted:
                continue
            if p["confidence"] < min_conf or (not include_unsnapped and not p["snapped"]):
                continue
            if not include_rejected and (p["feedback"] or {}).get("status") == "rejected":
                continue
            feats.append(f)
        return {"type": "FeatureCollection", "features": feats}

    @app.get("/api/network")
    def get_network():
        return network.get()

    @app.get("/api/frames/{frame_id}")
    def frame(frame_id: str):
        fr = frames().get(frame_id)
        if fr is None:
            raise HTTPException(404, "unknown frame")
        return {"frame_id": frame_id, "image_id": fr["image_id"],
                "lon": fr["lon"], "lat": fr["lat"], "heading": fr["heading"],
                "captured_at": str(fr["captured_at"]), "is_pano": bool(fr["is_pano"]),
                "mapillary_url": f"https://www.mapillary.com/app/?pKey={fr['image_id']}",
                "image_url": f"/api/frames/{frame_id}/image"}

    @app.get("/api/frames/{frame_id}/image")
    def frame_image(frame_id: str):
        fr = frames().get(frame_id)  # only manifest frames: no user-controlled paths
        path = settings.root / fr["path"] if fr else None
        if path is None or not path.is_file():
            raise HTTPException(404, "unknown frame")
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/stats")
    def stats():
        feats = all_barriers()
        by_type = Counter(f["properties"]["type"] for f in feats)
        unsn = Counter(f["properties"]["type"] for f in feats if not f["properties"]["snapped"])
        fb = feedback.summary()
        g = settings.project.gemini
        m = metrics.get().get(f"{g.model}/{g.prompt}", {})
        final = m.get("holdout") or m.get("tuning") or {}
        ledger = settings.paths.cache / "gemini_calls.jsonl"
        calls = 0
        if ledger.is_file():
            with open(ledger, encoding="utf-8") as fh:
                calls = sum(json.loads(line).get("calls", 1) for line in fh if line.strip())
        return {
            "barriers": len(feats),
            "snapped": sum(f["properties"]["snapped"] for f in feats),
            "multi_view": sum(f["properties"]["n_views"] > 1 for f in feats),
            "by_type": {t: {"total": n, "unsnapped": unsn.get(t, 0)}
                        for t, n in by_type.most_common()},
            "frames": len(frames()),
            "model": g.model, "prompt": g.prompt,
            "precision": final.get("labels", {}).get("micro", {}).get("precision"),
            "recall": final.get("labels", {}).get("micro", {}).get("recall"),
            "gemini_calls": {"used": calls, "budget": settings.project.budget.max_gemini_calls},
            "feedback": {"barriers": len(fb),
                         "confirmed": sum(s["status"] == "confirmed" for s in fb.values()),
                         "rejected": sum(s["status"] == "rejected" for s in fb.values())},
        }

    @app.post("/api/feedback", status_code=201)
    def post_feedback(fb: Feedback):
        if "feedback" not in settings.project.app.features:
            raise HTTPException(404, "feedback is disabled")
        if not any(f["properties"]["id"] == fb.barrier_id for f in all_barriers()):
            raise HTTPException(404, "unknown barrier")
        fid = feedback.add(fb)
        return JSONResponse({"id": fid, "barrier": feedback.summary()[fb.barrier_id]},
                            status_code=201)

    return app
