"""Frame selection (SPEC §3.3): age, distance to the walk network, thinning, spatial spread, cap."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

THIN_CELL_M = 8.0
HEADING_SECTORS = 8  # 45° each
SPREAD_BLOCK_M = 25.0
NEAR_EDGE_M = 15.0
SKIP_CAMERA_TYPES = {"fisheye"}
MAJOR_ROAD_M = 15.0          # frames this close to a motorway/trunk are shot from the road
MAJOR_ROAD_SEQ_SHARE = 0.5   # drop whole sequences that mostly run along such roads
MAX_POSITION_SHIFT_M = 15.0  # raw GPS vs SfM-corrected position disagreement
MAX_PANO_HEADING_DIFF = 60.0  # panorama heading vs direction of travel


@dataclass
class Candidate:
    image_id: str
    x: float  # projected metres
    y: float
    heading: float | None
    captured_at: int  # ms since epoch
    is_pano: bool
    on_foot: bool = False
    quality: float = 0.0

    @property
    def priority(self) -> tuple:
        # Newest first; ties broken by on-foot capture and Mapillary's quality score.
        return (self.captured_at, self.on_foot, self.quality)


def hfov_deg(camera_parameters: list[float] | None, width: int | None = None,
             default: float = 70.0) -> tuple[float, bool]:
    """Horizontal FOV from Mapillary camera_parameters [focal, k1, k2] where focal is a
    fraction of the image width. Returns (fov, was_defaulted)."""
    if camera_parameters and camera_parameters[0] and camera_parameters[0] > 0:
        f = float(camera_parameters[0])
        fov = math.degrees(2 * math.atan(0.5 / f))
        if 20 <= fov <= 150:
            return fov, False
    return default, True


def sector(heading: float | None) -> int:
    if heading is None:
        return -1
    return int(((heading % 360) + 22.5) // 45) % HEADING_SECTORS


def thin(cands: list[Candidate], cell_m: float = THIN_CELL_M) -> list[Candidate]:
    """At most one image per (8 m cell, 45° heading sector), keeping the highest priority.
    Panoramas see all directions, so they share one sector per cell."""
    best: dict[tuple, Candidate] = {}
    for c in cands:
        key = (math.floor(c.x / cell_m), math.floor(c.y / cell_m),
               "pano" if c.is_pano else sector(c.heading))
        if key not in best or c.priority > best[key].priority:
            best[key] = c
    return list(best.values())


def spread(cands: list[Candidate], cap: int, block_m: float = SPREAD_BLOCK_M
           ) -> list[Candidate]:
    """Pick up to `cap` candidates round-robin over block_m blocks (best first in each block),
    so a limited budget covers as many streets as possible instead of the densest ones."""
    blocks: dict[tuple, list[Candidate]] = defaultdict(list)
    for c in cands:
        blocks[(math.floor(c.x / block_m), math.floor(c.y / block_m))].append(c)
    queues = [sorted(v, key=lambda c: c.priority, reverse=True) for v in blocks.values()]
    # Deterministic block order: blocks with the best head candidate first.
    queues.sort(key=lambda q: q[0].priority, reverse=True)
    out: list[Candidate] = []
    depth = 0
    while len(out) < cap:
        added = False
        for q in queues:
            if depth < len(q):
                out.append(q[depth])
                added = True
                if len(out) >= cap:
                    break
        if not added:
            break
        depth += 1
    return out


def distance_to_network(xy: np.ndarray, lines) -> np.ndarray:
    """Distance (m) from each point to the nearest line; inputs in a metric CRS."""
    import shapely
    from shapely import STRtree

    tree = STRtree(lines)
    pts = shapely.points(xy)
    idx = tree.nearest(pts)
    return shapely.distance(pts, np.asarray(lines)[idx])


def major_road_sequences(seq_ids: list, dist_to_major: np.ndarray,
                         near_m: float = MAJOR_ROAD_M,
                         share: float = MAJOR_ROAD_SEQ_SHARE) -> set:
    """Sequences where at least `share` of the images are within near_m of a major road."""
    near: dict = defaultdict(list)
    for sid, d in zip(seq_ids, dist_to_major, strict=True):
        near[sid].append(d <= near_m)
    return {sid for sid, v in near.items() if sid is not None and np.mean(v) >= share}
