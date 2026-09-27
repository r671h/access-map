"""Mapillary API v4 image search.

bbox searches have no cursor paging and return at most `limit` (2000) images, so the
area is split into ~0.005° tiles and any tile that comes back full is split again.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from accessmap.http import get_json

log = logging.getLogger(__name__)

API = "https://graph.mapillary.com"
MAX_LIMIT = 2000
TILE_DEG = 0.005
MIN_TILE_DEG = 0.0003

IMAGE_FIELDS = (
    "id,captured_at,geometry,computed_geometry,compass_angle,computed_compass_angle,"
    "is_pano,camera_type,camera_parameters,width,height,sequence,thumb_2048_url,"
    "on_foot,quality_score"
)
COUNT_FIELDS = "id,captured_at,geometry,computed_geometry,is_pano"

Bbox = tuple[float, float, float, float]


def _auth(token: str) -> dict[str, str]:
    # Header instead of ?access_token= so the token never shows up in logged URLs.
    return {"Authorization": f"OAuth {token}"}


def tiles(bbox: Bbox, size: float = TILE_DEG) -> Iterator[Bbox]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        nlon = min(lon + size, max_lon)
        while lat < max_lat:
            nlat = min(lat + size, max_lat)
            yield (lon, lat, nlon, nlat)
            lat = nlat
        lon = nlon


def split(bbox: Bbox) -> list[Bbox]:
    min_lon, min_lat, max_lon, max_lat = bbox
    mlon, mlat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
    return [
        (min_lon, min_lat, mlon, mlat),
        (mlon, min_lat, max_lon, mlat),
        (min_lon, mlat, mlon, max_lat),
        (mlon, mlat, max_lon, max_lat),
    ]


def search_tile(token: str, bbox: Bbox, fields: str = IMAGE_FIELDS, limit: int = MAX_LIMIT,
                **filters) -> list[dict]:
    params = {"bbox": ",".join(f"{x:.6f}" for x in bbox), "fields": fields, "limit": limit,
              **filters}
    return get_json(f"{API}/images", params=params, headers=_auth(token), timeout=60)["data"]


def search_bbox(token: str, bbox: Bbox, fields: str = IMAGE_FIELDS, **filters) -> list[dict]:
    """All images in bbox, de-duplicated by id."""
    seen: dict[str, dict] = {}
    stack = list(tiles(bbox))
    while stack:
        t = stack.pop()
        data = search_tile(token, t, fields=fields, **filters)
        if len(data) >= MAX_LIMIT and (t[2] - t[0]) > MIN_TILE_DEG:
            log.debug("tile %s full (%d), splitting", t, len(data))
            stack.extend(split(t))
            continue
        for img in data:
            seen[img["id"]] = img
    return list(seen.values())


def position(img: dict) -> tuple[float, float]:
    """(lon, lat), preferring the computed (SfM-corrected) geometry."""
    geom = img.get("computed_geometry") or img["geometry"]
    lon, lat = geom["coordinates"][:2]
    return float(lon), float(lat)
