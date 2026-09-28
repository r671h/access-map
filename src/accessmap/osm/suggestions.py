"""`accessmap export-osm` (SPEC §10): suggested OSM tags for detected barriers.

Never edits OpenStreetMap. Writes data/processed/osm_suggestions.geojson: one Point feature
per barrier that is permanent, confident (>= MIN_CONF), seen in >= MIN_VIEWS photos, has a
clear OSM tag, and is not mapped yet (no matching OSM object within NEAR_M for its kind).
It is a plain GeoJSON FeatureCollection, which MapRoulette accepts as a challenge file (one task per
feature); each feature carries the suggested tags, an instruction and a Mapillary link so a
mapper can verify the photo before editing by hand.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from pyproj import Geod
from shapely import LineString, Point

from accessmap.config import Settings

log = logging.getLogger(__name__)

GEOD = Geod(ellps="WGS84")
MIN_CONF = 0.8
MIN_VIEWS = 2
# "Already mapped" radius per OSM object: a crossing's two kerbs are ~8-15 m apart, so a
# lowered curb 15 m from a mapped kerb can be the unmapped other side; stairs are long and
# placed from photos with ~10-15 m error, so a detection 13 m from mapped steps is them.
NEAR_M = {"kerb": 10.0, "steps": 20.0}

# type -> (suggested tags, OSM geometry that already maps it, instruction)
TAGS = {
    "curb_ramp": ({"kerb": "lowered"}, "kerb",
                  "Street-level photos show a lowered curb here. If the photo confirms it, add a "
                  "node on the footway/crossing where it meets the road with kerb=lowered."),
    "raised_curb": ({"kerb": "raised"}, "kerb",
                    "Street-level photos show a raised (not lowered) curb at a crossing point. If "
                    "the photo confirms it, add a node with kerb=raised where the crossing meets "
                    "the road."),
    "stairs": ({"highway": "steps"}, "steps",
               "Street-level photos show outdoor stairs on a pedestrian route. If the photo "
               "confirms it and the stairs are missing, map them as a way with highway=steps "
               "(add step_count, handrail, ramp if visible)."),
}


def _distance_m(p: Point, geom) -> float:
    """Metres from p to the nearest point of geom (a Point or LineString in lon/lat)."""
    from shapely.ops import nearest_points

    q = nearest_points(p, geom)[1]
    return GEOD.inv(p.x, p.y, q.x, q.y)[2]


def load_osm(settings: Settings) -> dict[str, list]:
    osm = settings.paths.raw / "osm"
    kerbs = json.loads((osm / "kerbs.geojson").read_text(encoding="utf-8"))["features"]
    steps = json.loads((osm / "steps.geojson").read_text(encoding="utf-8"))["features"]
    return {
        # any kerb=* value counts as "already mapped": a wrong value is a job for a mapper
        # looking at the spot, not for an automatic suggestion
        "kerb": [Point(f["geometry"]["coordinates"]) for f in kerbs
                 if f["properties"].get("kerb")],
        "steps": [LineString(f["geometry"]["coordinates"]) for f in steps
                  if f["geometry"]["type"] == "LineString"],
    }


def suggestions(barriers: list[dict], osm: dict[str, list], min_conf: float = MIN_CONF,
                min_views: int = MIN_VIEWS, near_m: dict[str, float] = NEAR_M
                ) -> tuple[list[dict], dict]:
    """(features, counts of why barriers were skipped)."""
    out, skipped = [], {"no_tag": 0, "temporary": 0, "low_confidence": 0, "few_views": 0,
                        "already_in_osm": 0}
    for f in barriers:
        p = f["properties"]
        if p["type"] not in TAGS:
            skipped["no_tag"] += 1
            continue
        if p.get("permanence", "permanent") != "permanent":
            skipped["temporary"] += 1
            continue
        if p["confidence"] < min_conf:
            skipped["low_confidence"] += 1
            continue
        if p["n_views"] < min_views:
            skipped["few_views"] += 1
            continue
        tags, kind, instruction = TAGS[p["type"]]
        pt = Point(f["geometry"]["coordinates"])
        near = [g for g in osm[kind] if _distance_m(pt, g) <= near_m[kind]]
        if near:
            skipped["already_in_osm"] += 1
            continue
        image = p["best_frame_id"].split("_")[0]
        out.append({"type": "Feature", "geometry": f["geometry"], "properties": {
            "id": p["id"],
            "barrier_type": p["type"],
            "suggested_tags": tags,
            "confidence": p["confidence"],
            "n_views": p["n_views"],
            "estimated_height_cm": p.get("estimated_height_cm"),
            "description": p.get("description_en"),
            "last_seen": str(p.get("last_seen", ""))[:10],
            "mapillary": f"https://www.mapillary.com/app/?pKey={image}",
            "instruction": instruction + " Automatic detection: check the photo first.",
            "source": "access-map (Gemini on Mapillary street-level imagery)",
        }})
    return out, skipped


def run(settings: Settings) -> dict:
    from accessmap.web.app import SiteData

    barriers = SiteData(settings).barriers()
    feats, skipped = suggestions(barriers, load_osm(settings))
    path = settings.paths.processed / "osm_suggestions.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                               ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    by_type = Counter(f["properties"]["barrier_type"] for f in feats)
    return {"path": str(path), "barriers": len(barriers), "suggestions": len(feats),
            "by_type": dict(by_type), "skipped": skipped,
            "rules": {"min_confidence": MIN_CONF, "min_views": MIN_VIEWS,
                      "not_in_osm_within_m": NEAR_M}}
