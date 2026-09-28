"""Detection → map coordinate (SPEC §6 steps 1–3).

Deviation from SPEC §6.1: the bearing offset uses the pinhole model
atan((x − 0.5) · 2 · tan(FOV/2)) instead of the linear (x − 0.5) · FOV. Both agree at the
centre and at the image edges; in between the linear form is off by up to ~4° for a 90° lens
(≈ 0.7 m at 10 m). Panorama crops are rectilinear, so the pinhole model holds for them too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pyproj import Geod

GEOD = Geod(ellps="WGS84")

MIN_DIST_M = 2.0
MAX_DIST_M = 25.0
FALLBACK_DIST_M = 8.0
FALLBACK_CONF_FACTOR = 0.7
CAMERA_HEIGHT_M = 2.0


@dataclass(frozen=True)
class Placement:
    lon: float
    lat: float
    bearing: float
    distance_m: float
    distance_source: str        # gemini | box_bottom | fallback
    confidence: float


def bearing_offset(x_norm: float, fov_deg: float) -> float:
    """Horizontal angle (deg, + = right) of a normalized x (0–1) in a pinhole image."""
    half = math.radians(fov_deg) / 2
    return math.degrees(math.atan((x_norm - 0.5) * 2 * math.tan(half)))


def vertical_fov(hfov_deg: float, width: int, height: int) -> float:
    half = math.radians(hfov_deg) / 2
    return math.degrees(2 * math.atan(math.tan(half) * height / width))


def distance_from_box_bottom(y_bottom_norm: float, vfov_deg: float,
                             camera_height_m: float = CAMERA_HEIGHT_M) -> float | None:
    """Ground distance to the box's bottom edge, assuming a level camera and flat ground.

    Returns None when the bottom edge is at or above the horizon.
    """
    half = math.radians(vfov_deg) / 2
    below = math.atan((y_bottom_norm - 0.5) * 2 * math.tan(half))
    if below <= math.radians(0.5):
        return None
    return camera_height_m / math.tan(below)


def destination(lon: float, lat: float, bearing_deg: float, distance_m: float
                ) -> tuple[float, float]:
    dlon, dlat, _ = GEOD.fwd(lon, lat, bearing_deg, distance_m)
    return dlon, dlat


def place(feature: dict, cam_lon: float, cam_lat: float, heading: float, fov: float,
          width: int, height: int) -> Placement:
    """Map position of one Gemini feature seen from a camera (box_2d is 0–1000)."""
    ymin, xmin, ymax, xmax = feature["box_2d"]
    bearing = (heading + bearing_offset((xmin + xmax) / 2000, fov)) % 360
    conf = float(feature["confidence"])

    dist = feature.get("estimated_distance_m")
    source = "gemini"
    if dist is None or not math.isfinite(dist) or dist <= 0:
        dist = distance_from_box_bottom(ymax / 1000, vertical_fov(fov, width, height))
        source = "box_bottom"
    if dist is None:
        dist, source = FALLBACK_DIST_M, "fallback"
        conf *= FALLBACK_CONF_FACTOR
    dist = min(max(dist, MIN_DIST_M), MAX_DIST_M)

    lon, lat = destination(cam_lon, cam_lat, bearing, dist)
    return Placement(lon, lat, bearing, dist, source, conf)
