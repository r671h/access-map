"""`accessmap geolocate` (phase 4): detections.jsonl → barriers.geojson (SPEC §6)."""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path

from accessmap.config import Settings
from accessmap.geo.cluster import cluster
from accessmap.geo.project import place
from accessmap.geo.snap import NetworkIndex

log = logging.getLogger(__name__)

TYPE_COLORS = {
    "curb_ramp": "#1a9850", "raised_curb": "#d73027", "step": "#f46d43", "stairs": "#a50026",
    "narrow_passage": "#7b3294", "steep_slope": "#2166ac", "rough_surface": "#8c510a",
    "no_sidewalk": "#000000",
}


@lru_cache(maxsize=4096)
def image_size(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def load_detections(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def place_all(frames: list[dict], root: Path) -> list[dict]:
    """One placed point per permanent feature of every usable frame."""
    out = []
    for fr in frames:
        if not fr["image_usable"]:
            continue
        w, h = image_size(str(root / fr["path"]))
        for j, feat in enumerate(fr["features"]):
            if feat.get("permanence", "permanent") != "permanent":
                continue
            p = place(feat, fr["lon"], fr["lat"], fr["heading"], fr["fov"], w, h)
            out.append({
                "type": feat["type"], "lon": p.lon, "lat": p.lat, "confidence": p.confidence,
                "raw_confidence": feat["confidence"], "distance_m": p.distance_m,
                "distance_source": p.distance_source, "frame_id": fr["frame_id"],
                "image_id": fr["image_id"], "feature_index": j,
                "captured_at": fr["captured_at"], "cam_lon": fr["lon"], "cam_lat": fr["lat"],
                "box_2d": feat["box_2d"],
                "estimated_height_cm": feat.get("estimated_height_cm"),
                "estimated_width_m": feat.get("estimated_width_m"),
                "description_en": feat["description_en"],
                "description_de": feat["description_de"],
            })
    return out


def barrier_id(c: dict) -> str:
    """Stable across reruns as long as the same detections form the cluster."""
    key = c["type"] + "|" + "|".join(sorted(f"{m['frame_id']}#{m['feature_index']}"
                                            for m in c["members"]))
    return c["type"] + "-" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _median(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def to_feature(c: dict, snap) -> dict:
    best = max(c["members"], key=lambda m: m["raw_confidence"])
    lon, lat = (snap.lon, snap.lat) if snap else (c["lon"], c["lat"])
    props = {
        "id": barrier_id(c),
        "type": c["type"],
        "confidence": c["confidence"],
        "n_views": c["n_views"],
        "n_detections": c["n_detections"],
        "snapped": snap is not None,
        "snap_target": snap.target if snap else None,
        "snap_distance_m": round(snap.distance_m, 1) if snap else None,
        "edge_u": snap.edge[0] if snap and snap.edge else None,
        "edge_v": snap.edge[1] if snap and snap.edge else None,
        "edge_key": snap.edge[2] if snap and snap.edge else None,
        "node": snap.node if snap else None,
        "raw_lon": round(c["lon"], 7),
        "raw_lat": round(c["lat"], 7),
        "estimated_height_cm": _median(m["estimated_height_cm"] for m in c["members"]),
        "estimated_width_m": _median(m["estimated_width_m"] for m in c["members"]),
        "description_en": best["description_en"],
        "description_de": best["description_de"],
        "best_frame_id": best["frame_id"],
        "frame_ids": sorted({m["frame_id"] for m in c["members"]}),
        "image_ids": sorted({m["image_id"] for m in c["members"]}),
        "last_seen": max(m["captured_at"] for m in c["members"]),
        "distance_sources": dict(Counter(m["distance_source"] for m in c["members"])),
    }
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]}}


def load_index(settings: Settings) -> NetworkIndex:
    import osmnx as ox

    osm = settings.paths.raw / "osm"
    G = ox.load_graphml(osm / "walk.graphml")
    crossings = json.loads((osm / "crossings.geojson").read_text(encoding="utf-8"))
    pts = [tuple(f["geometry"]["coordinates"]) for f in crossings["features"]
           if f["geometry"]["type"] == "Point"]
    return NetworkIndex(G, pts)


def write_fc(path: Path, features: list[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features},
                               ensure_ascii=False), encoding="utf-8")


def run(settings: Settings) -> dict:
    proc = settings.paths.processed
    frames = load_detections(proc / "detections.jsonl")
    placed = place_all(frames, settings.root)
    clusters = cluster(placed)
    index = load_index(settings)
    g = settings.project.geo

    snapped, unsnapped = [], []
    for c in clusters:
        s = index.snap(c["lon"], c["lat"], c["type"], g.snap_curb_m, g.snap_other_m)
        (snapped if s else unsnapped).append(to_feature(c, s))
    write_fc(proc / "barriers.geojson", snapped)
    write_fc(proc / "barriers_unsnapped.geojson", unsnapped)

    all_feats = snapped + unsnapped
    by_type = Counter(f["properties"]["type"] for f in all_feats)
    unsnapped_by_type = Counter(f["properties"]["type"] for f in unsnapped)
    stats = {
        "frames": len(frames),
        "usable_frames": sum(f["image_usable"] for f in frames),
        "detections_placed": len(placed),
        "distance_sources": dict(Counter(p["distance_source"] for p in placed)),
        "clusters_kept": len(all_feats),
        "dropped_single_low_conf": len(placed) - sum(c["n_detections"] for c in clusters),
        "snapped": len(snapped),
        "unsnapped": len(unsnapped),
        "unsnapped_share": round(len(unsnapped) / len(all_feats), 3) if all_feats else 0.0,
        "multi_view": sum(f["properties"]["n_views"] > 1 for f in all_feats),
        "by_type": {t: {"total": n, "unsnapped": unsnapped_by_type.get(t, 0)}
                    for t, n in by_type.most_common()},
        "snap_curb_m": g.snap_curb_m,
        "snap_other_m": g.snap_other_m,
    }
    (settings.paths.reports / "geolocate.json").write_text(json.dumps(stats, indent=2),
                                                           encoding="utf-8", newline="\n")
    maps = barriers_map(settings, frames, snapped, unsnapped, index)
    stats["maps"] = [str(p) for p in maps]
    return stats


def barriers_map(settings: Settings, frames: list[dict], snapped: list[dict],
                 unsnapped: list[dict], index: NetworkIndex) -> list[Path]:
    """Network + camera positions + barriers; the full area and a zoom on the densest part."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from accessmap.geo.snap import to_metric

    def xy(feats, raw=False):
        pts = [((f["properties"]["raw_lon"], f["properties"]["raw_lat"]) if raw
                else tuple(f["geometry"]["coordinates"])) for f in feats]
        return [to_metric(lon, lat) for lon, lat in pts]

    cams = [to_metric(f["lon"], f["lat"]) for f in frames]
    all_pts = xy(snapped) + xy(unsnapped)
    out = []
    views = [("barriers_map.png", None)]
    if all_pts:
        # Zoom: 250 m window around the point with the most barriers within 125 m.
        import numpy as np

        arr = np.array(all_pts)
        counts = [(np.abs(arr - p).max(axis=1) < 125).sum() for p in arr]
        cx, cy = arr[int(np.argmax(counts))]
        views.append(("barriers_map_zoom.png", (cx - 125, cx + 125, cy - 125, cy + 125)))

    for name, window in views:
        fig, ax = plt.subplots(figsize=(12, 13) if window is None else (12, 12), dpi=110)
        for line in index.lines:
            x, y = line.xy
            ax.plot(x, y, color="#bbbbbb", lw=0.8 if window is None else 1.6, zorder=1)
        if cams:
            ax.scatter(*zip(*cams, strict=True), s=4 if window is None else 14,
                       color="#4a90d9", alpha=0.5, zorder=2, linewidths=0)
        for t, color in TYPE_COLORS.items():
            s_pts = xy([f for f in snapped if f["properties"]["type"] == t])
            u_pts = xy([f for f in unsnapped if f["properties"]["type"] == t])
            size = 18 if window is None else 70
            if s_pts:
                ax.scatter(*zip(*s_pts, strict=True), s=size, color=color, zorder=4,
                           edgecolors="white", linewidths=0.6)
            if u_pts:
                ax.scatter(*zip(*u_pts, strict=True), s=size, color=color, marker="x",
                           zorder=4, linewidths=1.5)
        if window is not None:
            # Snap moves: raw cluster position → snapped position.
            for f in snapped:
                (x0, y0), = xy([f], raw=True)
                (x1, y1), = xy([f])
                ax.plot([x0, x1], [y0, y1], color="#555555", lw=0.6, zorder=3)
            ax.set_xlim(window[0], window[1])
            ax.set_ylim(window[2], window[3])
        handles = [Line2D([], [], marker="o", ls="", color=c, label=t)
                   for t, c in TYPE_COLORS.items()]
        handles += [Line2D([], [], marker="x", ls="", color="k", label="unsnapped"),
                    Line2D([], [], marker="o", ls="", color="#4a90d9", alpha=0.5,
                           label="camera")]
        ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        title = "Barriers (snapped ●, unsnapped ×), walk network, camera positions"
        ax.set_title(title + (" — zoom 250 m, grey ticks = snap moves" if window else ""))
        path = settings.paths.qa / name
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        out.append(path)
    return out
