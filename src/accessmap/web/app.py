"""FastAPI app (SPEC §9): barriers, network, frames/photos, stats, feedback.

The data logic lives in `SiteData`, shared with the static export for Vercel
(`accessmap export-site`), so the local API and the public site serve the same JSON.
The browser routes itself with static/router.js on /api/graph (so the Vercel site can);
/api/route answers the same with the Python router. Run with `accessmap serve`.
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
EMPTY_FC = {"type": "FeatureCollection", "features": []}


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


def feedback_status(confirm: int, reject: int) -> str:
    """Rejected when rejects outnumber confirms (same rule in the Vercel function)."""
    return "rejected" if reject > confirm else "confirmed" if confirm > reject else "disputed"


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
        """barrier_id -> {confirm, reject, status}."""
        with self._conn() as c:
            rows = c.execute("SELECT barrier_id, verdict, COUNT(*) FROM feedback "
                             "GROUP BY barrier_id, verdict").fetchall()
        out: dict[str, dict] = {}
        for bid, verdict, n in rows:
            out.setdefault(bid, {"confirm": 0, "reject": 0})[verdict] = n
        for s in out.values():
            s["status"] = feedback_status(s["confirm"], s["reject"])
        return out


class SiteData:
    """Everything the frontend reads, from the pipeline outputs under the project root."""

    def __init__(self, settings: Settings):
        self.settings = settings
        proc = settings.paths.processed
        self._snapped = _FileCache(proc / "barriers.geojson", EMPTY_FC)
        self._unsnapped = _FileCache(proc / "barriers_unsnapped.geojson", EMPTY_FC)
        self._network = _FileCache(settings.paths.raw / "osm" / "walk_edges.geojson", EMPTY_FC)
        self._metrics = _FileCache(settings.paths.reports / "metrics.json", {})
        self._graph = _FileCache(proc / "routing_graph.json", None)
        self._frames: dict = {}

    def config(self) -> dict:
        p = self.settings.project
        return {"area": {"name": p.area.name, "bbox": p.area.bbox},
                "languages": p.ui.languages, "language": p.ui.language,
                "types": p.detection.types, "profiles": p.profiles,
                "features": p.app.features, "routing": self.graph() is not None,
                "max_detour": p.routing.max_detour}

    def graph(self) -> dict | None:
        """routing_graph.json (accessmap build-graph), or None before it exists."""
        return self._graph.get()

    def frames(self) -> dict[str, dict]:
        path = self.settings.paths.processed / "frames.parquet"
        if not path.is_file():
            return {}
        m = path.stat().st_mtime
        if self._frames.get("mtime") != m:
            import pandas as pd

            df = pd.read_parquet(path)
            self._frames = {"mtime": m,
                            "data": {r["frame_id"]: r for r in df.to_dict("records")}}
        return self._frames["data"]

    def barriers(self, feedback: dict[str, dict] | None = None) -> list[dict]:
        fb = feedback or {}
        return [{**f, "properties": {**f["properties"], "feedback": fb.get(f["properties"]["id"])}}
                for fc in (self._snapped.get(), self._unsnapped.get()) for f in fc["features"]]

    def network(self) -> dict:
        return self._network.get()

    def stats(self, feedback: dict[str, dict] | None = None) -> dict:
        s = self.settings
        feats = self.barriers(feedback)
        by_type = Counter(f["properties"]["type"] for f in feats)
        unsn = Counter(f["properties"]["type"] for f in feats if not f["properties"]["snapped"])
        g = s.project.gemini
        m = self._metrics.get().get(f"{g.model}/{g.prompt}", {})
        split = "holdout" if m.get("holdout") else "tuning" if m.get("tuning") else None
        final = m.get(split) or {}
        labels = final.get("labels", {})
        # Per-type rows need the bigger sample: all hand-labelled frames (hold-out + tuning).
        by_type_split = "all_frames" if m.get("all_frames") else split
        type_labels = (m.get(by_type_split) or {}).get("labels", {})
        dates = sorted(str(fr["captured_at"])[:10] for fr in self.frames().values())
        ledger = s.paths.cache / "gemini_calls.jsonl"
        calls = 0
        if ledger.is_file():
            with open(ledger, encoding="utf-8") as fh:
                calls = sum(json.loads(line).get("calls", 1) for line in fh if line.strip())
        fb = feedback or {}
        return {
            "barriers": len(feats),
            "snapped": sum(f["properties"]["snapped"] for f in feats),
            "multi_view": sum(f["properties"]["n_views"] > 1 for f in feats),
            "by_type": {t: {"total": n, "unsnapped": unsn.get(t, 0)}
                        for t, n in by_type.most_common()},
            "frames": len(self.frames()),
            "model": g.model, "prompt": g.prompt,
            "precision": labels.get("micro", {}).get("precision"),
            "recall": labels.get("micro", {}).get("recall"),
            # Frame-level scores against the hand labels: headline on `split` (hold-out once
            # scored), the per-type table on all labelled frames (the stats page).
            "accuracy": {"split": split, "frames": labels.get("frames", 0),
                         "micro": labels.get("micro"),
                         "by_type_split": by_type_split,
                         "by_type_frames": type_labels.get("frames", 0),
                         "by_type_micro": type_labels.get("micro"),
                         "per_type": {t: {k: v[k] for k in
                                          ("tp", "fp", "fn", "precision", "recall")}
                                      for t, v in type_labels.get("per_type", {}).items()}},
            "photo_dates": {"from": dates[0], "to": dates[-1]} if dates else None,
            "gemini_calls": {"used": calls, "budget": s.project.budget.max_gemini_calls},
            "feedback": {"barriers": len(fb),
                         "confirmed": sum(v["status"] == "confirmed" for v in fb.values()),
                         "rejected": sum(v["status"] == "rejected" for v in fb.values())},
        }


def create_app(settings: Settings) -> FastAPI:
    data = SiteData(settings)
    feedback = FeedbackStore(settings.paths.data / "feedback.sqlite")
    routers: dict = {}

    app = FastAPI(title="access-map", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/stats.html", include_in_schema=False)
    def stats_page():
        return FileResponse(STATIC / "stats.html")

    @app.get("/router.js", include_in_schema=False)
    def router_js():
        return FileResponse(STATIC / "router.js", media_type="text/javascript")

    @app.get("/api/graph")
    def graph():
        g = data.graph()
        if g is None:
            raise HTTPException(404, "no routing graph: run `accessmap build-graph`")
        return g

    @app.get("/api/route")
    def route(start: str = Query(..., alias="from", description="lat,lon"),
              end: str = Query(..., alias="to", description="lat,lon"),
              profile: str = Query("wheelchair")):
        from accessmap.routing.router import Router, RoutingError

        g = data.graph()
        if g is None:
            raise HTTPException(404, "no routing graph: run `accessmap build-graph`")
        try:
            (la, lo), (lb, lob) = ([float(x) for x in p.split(",")] for p in (start, end))
        except ValueError:
            raise HTTPException(422, "from/to must be 'lat,lon'") from None
        if routers.get("graph") is not g:
            routers.update(graph=g, router=Router(g))
        rejected = {bid for bid, v in feedback.summary().items() if v["status"] == "rejected"}
        try:
            return routers["router"].route([lo, la], [lob, lb], profile, rejected)
        except RoutingError as e:
            raise HTTPException(422, str(e)) from None

    @app.get("/api/config")
    def config():
        return data.config()

    @app.get("/api/barriers")
    def barriers(types: str | None = Query(None, description="comma-separated types"),
                 min_conf: float = Query(0.0, ge=0, le=1),
                 include_unsnapped: bool = True, include_rejected: bool = True):
        wanted = set(types.split(",")) if types else None
        feats = []
        for f in data.barriers(feedback.summary()):
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
        return data.network()

    @app.get("/api/frames/{frame_id}")
    def frame(frame_id: str):
        fr = data.frames().get(frame_id)
        if fr is None:
            raise HTTPException(404, "unknown frame")
        return {"frame_id": frame_id, "image_id": fr["image_id"],
                "lon": fr["lon"], "lat": fr["lat"], "heading": fr["heading"],
                "captured_at": str(fr["captured_at"]), "is_pano": bool(fr["is_pano"]),
                "mapillary_url": f"https://www.mapillary.com/app/?pKey={fr['image_id']}",
                "image_url": f"/api/frames/{frame_id}/image"}

    @app.get("/api/frames/{frame_id}/image")
    def frame_image(frame_id: str):
        fr = data.frames().get(frame_id)  # only manifest frames: no user-controlled paths
        path = settings.root / fr["path"] if fr else None
        if path is None or not path.is_file():
            raise HTTPException(404, "unknown frame")
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/stats")
    def stats():
        return data.stats(feedback.summary())

    @app.get("/api/feedback")
    def get_feedback():
        """barrier_id -> {confirm, reject, status}; the Vercel function answers the same."""
        return feedback.summary()

    @app.post("/api/feedback", status_code=201)
    def post_feedback(fb: Feedback):
        if "feedback" not in settings.project.app.features:
            raise HTTPException(404, "feedback is disabled")
        if not any(f["properties"]["id"] == fb.barrier_id for f in data.barriers()):
            raise HTTPException(404, "unknown barrier")
        fid = feedback.add(fb)
        return JSONResponse({"id": fid, "barrier": feedback.summary()[fb.barrier_id]},
                            status_code=201)

    return app
