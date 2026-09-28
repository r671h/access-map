"""Merge repeated sightings of the same barrier (SPEC §6 step 4)."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

EARTH_R_M = 6_371_008.8
EPS_M = 6.0
CONF_CAP = 0.97
MIN_SINGLE_CONF = 0.5


def combined_confidence(detections: list[dict]) -> float:
    """1 − Π(1 − c), capped. Several boxes from the same frame count once (their max),
    so one image cannot vouch for itself."""
    per_frame: dict[str, float] = defaultdict(float)
    for d in detections:
        per_frame[d["frame_id"]] = max(per_frame[d["frame_id"]], d["confidence"])
    miss = math.prod(1 - c for c in per_frame.values())
    return min(1 - miss, CONF_CAP)


def cluster(detections: list[dict], eps_m: float = EPS_M) -> list[dict]:
    """DBSCAN (haversine) per type. Each detection needs type, lon, lat, confidence,
    frame_id, image_id. Returns one dict per kept cluster."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for d in detections:
        by_type[d["type"]].append(d)

    out = []
    for btype, dets in sorted(by_type.items()):
        coords = np.radians([[d["lat"], d["lon"]] for d in dets])
        labels = DBSCAN(eps=eps_m / EARTH_R_M, min_samples=1, metric="haversine",
                        algorithm="ball_tree").fit_predict(coords)
        groups: dict[int, list[dict]] = defaultdict(list)
        for label, d in zip(labels, dets, strict=True):
            groups[int(label)].append(d)
        for members in groups.values():
            conf = combined_confidence(members)
            if len(members) == 1 and conf < MIN_SINGLE_CONF:
                continue
            w = np.array([m["confidence"] for m in members])
            out.append({
                "type": btype,
                "lon": float(np.average([m["lon"] for m in members], weights=w)),
                "lat": float(np.average([m["lat"] for m in members], weights=w)),
                "confidence": round(conf, 3),
                "n_detections": len(members),
                "n_views": len({m["image_id"] for m in members}),
                "members": members,
            })
    return out
