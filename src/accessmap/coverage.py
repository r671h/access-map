"""Coverage report for the area (SPEC §2): Mapillary imagery vs the walking network, OSM kerbs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from accessmap.config import Settings

log = logging.getLogger(__name__)

NEAR_M = 15.0
SAMPLE_STEP_M = 5.0

OSM_SELECTORS = {
    "kerb_nodes": 'node["kerb"]',
    "kerb_lowered_flush": 'node["kerb"~"^(lowered|flush)$"]',
    "kerb_raised": 'node["kerb"="raised"]',
    "crossing_nodes": 'node["highway"="crossing"]',
    "steps_ways": 'way["highway"="steps"]',
    "sidewalk_ways": 'way["footway"="sidewalk"]',
    "barrier_nodes": 'node["barrier"]',
}


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def image_age_years(captured_at_ms: int, now: datetime | None = None) -> float:
    now = now or datetime.now(UTC)
    return (now - datetime.fromtimestamp(captured_at_ms / 1000, UTC)).days / 365.25


def in_bbox(lonlat: tuple[float, float], bbox) -> bool:
    return bbox[0] <= lonlat[0] <= bbox[2] and bbox[1] <= lonlat[1] <= bbox[3]


def sample_lines(lines_xy: list[np.ndarray], step: float = SAMPLE_STEP_M
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step` m along each polyline (projected coords). Returns (points, weights)
    where the weights are the line length each point stands for."""
    pts, wts = [], []
    for xy in lines_xy:
        seg = np.diff(xy, axis=0)
        seglen = np.hypot(seg[:, 0], seg[:, 1])
        total = seglen.sum()
        if total == 0:
            continue
        n = max(1, int(np.ceil(total / step)))
        d = (np.arange(n) + 0.5) * total / n
        cum = np.concatenate([[0], np.cumsum(seglen)])
        x = np.interp(d, cum, xy[:, 0])
        y = np.interp(d, cum, xy[:, 1])
        pts.append(np.column_stack([x, y]))
        wts.append(np.full(n, total / n))
    if not pts:
        return np.empty((0, 2)), np.empty(0)
    return np.vstack(pts), np.concatenate(wts)


def covered_share(points: np.ndarray, weights: np.ndarray, images_xy: np.ndarray,
                  near_m: float = NEAR_M) -> float:
    """Length-weighted share of network sample points within near_m of any image."""
    if len(points) == 0:
        return 0.0
    if len(images_xy) == 0:
        return 0.0
    from sklearn.neighbors import KDTree

    dist, _ = KDTree(images_xy).query(points, k=1)
    near = dist[:, 0] <= near_m
    return float(weights[near].sum() / weights.sum())


def fetch_images(settings: Settings, refresh: bool = False) -> list[dict]:
    """Image index for the bbox (id, time, position, pano flag), cached on disk."""
    from accessmap.imagery.mapillary import COUNT_FIELDS, search_bbox

    out = settings.paths.raw / "mapillary" / "coverage_index.json"
    bbox = list(settings.project.area.bbox)
    if out.is_file() and not refresh:
        cached = json.loads(out.read_text(encoding="utf-8"))
        if cached.get("bbox") == bbox:
            return cached["images"]
    token = settings.secrets.mapillary_token
    if not token:
        raise RuntimeError("MAPILLARY_TOKEN is not set (SETUP.md §3)")
    images = search_bbox(token, tuple(bbox), fields=COUNT_FIELDS + ",on_foot")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bbox": bbox, "fetched_at": datetime.now(UTC).isoformat(),
                               "images": images}), encoding="utf-8")
    return images


def build_report(settings: Settings, refresh: bool = False) -> dict:
    import osmnx as ox
    from pyproj import Transformer

    from accessmap.imagery.mapillary import position
    from accessmap.osm.network import download_walk_graph
    from accessmap.osm.overpass import count

    p = settings.project
    bbox = p.area.bbox
    settings.paths.ensure()

    log.info("Counting OSM kerb/steps/barrier tags")
    osm_counts = count(bbox, OSM_SELECTORS)

    log.info("Downloading walking network")
    G = download_walk_graph(settings)
    edges = ox.graph_to_gdfs(G, nodes=False)
    epsg = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    edges_m = edges.to_crs(epsg)
    # Undirected length: each street appears once per direction in the walk graph.
    edges_m = edges_m[~edges_m.geometry.normalize().duplicated()]
    network_km = float(edges_m.length.sum() / 1000)

    log.info("Fetching Mapillary image index")
    # Mapillary filters on the raw position; drop images whose corrected position is outside.
    images = [i for i in fetch_images(settings, refresh=refresh) if in_bbox(position(i), bbox)]
    ages = np.array([image_age_years(i["captured_at"]) for i in images]) if images else np.empty(0)
    lonlat = np.array([position(i) for i in images]) if images else np.empty((0, 2))
    to_m = Transformer.from_crs(4326, epsg, always_xy=True)
    xy = np.column_stack(to_m.transform(lonlat[:, 0], lonlat[:, 1])) if images else lonlat

    pts, wts = sample_lines([np.asarray(g.coords) for g in edges_m.geometry])
    max_age = p.imagery.max_age_years
    fresh = ages <= max_age
    share_all = covered_share(pts, wts, xy)
    share_fresh = covered_share(pts, wts, xy[fresh])

    (x0, x1), (y0, y1) = to_m.transform([bbox[0], bbox[2]], [bbox[1], bbox[3]])
    area_km2 = abs((x1 - x0) * (y1 - y0)) / 1e6
    years = {}
    for i in images:
        y = datetime.fromtimestamp(i["captured_at"] / 1000, UTC).year
        years[y] = years.get(y, 0) + 1

    report = {
        "area": p.area.name,
        "bbox": list(bbox),
        "area_km2": round(area_km2, 2),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "osm": osm_counts | {"walk_network_km": round(network_km, 1),
                             "walk_edges_undirected": int(len(edges_m))},
        "mapillary": {
            "images_total": len(images),
            "panoramas": int(sum(bool(i.get("is_pano")) for i in images)),
            "on_foot": int(sum(bool(i.get("on_foot")) for i in images)),
            f"images_le_{max_age:g}y": int(fresh.sum()),
            "images_le_3y": int((ages <= 3).sum()),
            "by_year": dict(sorted(years.items())),
            "network_share_within_15m_all": round(share_all, 3),
            f"network_share_within_15m_le_{max_age:g}y": round(share_fresh, 3),
        },
    }
    (settings.paths.reports / "coverage.json").write_text(json.dumps(report, indent=2),
                                                          encoding="utf-8")
    (settings.paths.reports / "coverage.md").write_text(to_markdown(report), encoding="utf-8")
    plot_map(settings.paths.qa / "coverage_map.png", edges_m, xy, ages, max_age, epsg, bbox)
    return report


def to_markdown(r: dict) -> str:
    o, m = r["osm"], r["mapillary"]
    kerbs = (f"{o['kerb_nodes']} (lowered/flush {o['kerb_lowered_flush']}, "
             f"raised {o['kerb_raised']})")
    fresh_key = next(k for k in m if k.startswith("images_le_") and k != "images_le_3y")
    share_fresh_key = next(k for k in m if k.startswith("network_share_within_15m_le_"))
    kerb_note = ("" if o["kerb_nodes"] >= 30 else
                 "\n> **Warning:** fewer than 30 `kerb=*` nodes — automatic evaluation against "
                 "OSM will be weak (SPEC §7).\n")
    return f"""# Coverage report — {r['area']}

bbox `{','.join(str(x) for x in r['bbox'])}` · {r['area_km2']} km² · generated {r['generated_at']}

## Mapillary
| metric | value |
|---|---|
| images in bbox | {m['images_total']} |
| panoramas | {m['panoramas']} |
| captured on foot | {m['on_foot']} |
| images ≤ {fresh_key.removeprefix('images_le_')} old | {m[fresh_key]} |
| images ≤ 3y old | {m['images_le_3y']} |
| walk network within 15 m of any image | {m['network_share_within_15m_all']:.0%} |
| walk network within 15 m of an image ≤ max age | {m[share_fresh_key]:.0%} |

Images by year: {', '.join(f'{y}: {n}' for y, n in m['by_year'].items()) or '—'}

## OpenStreetMap
| metric | value |
|---|---|
| walking network (undirected) | {o['walk_network_km']} km, {o['walk_edges_undirected']} edges |
| `kerb=*` nodes | {kerbs} |
| crossing nodes | {o['crossing_nodes']} |
| `highway=steps` ways | {o['steps_ways']} |
| `footway=sidewalk` ways | {o['sidewalk_ways']} |
| `barrier=*` nodes | {o['barrier_nodes']} |
{kerb_note}
Map: `reports/qa/coverage_map.png`
"""


def plot_map(path: Path, edges_m, xy: np.ndarray, ages: np.ndarray, max_age: float, epsg: int,
             bbox) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    edges_m.plot(ax=ax, color="#555", linewidth=0.6)
    if len(xy):
        old = ages > max_age
        ax.scatter(xy[old, 0], xy[old, 1], s=3, c="#e67e22", alpha=0.5,
                   label=f"image > {max_age:g}y ({int(old.sum())})")
        ax.scatter(xy[~old, 0], xy[~old, 1], s=3, c="#1f77b4", alpha=0.6,
                   label=f"image ≤ {max_age:g}y ({int((~old).sum())})")
        ax.legend(loc="upper right", markerscale=4)
    ax.set_title(f"Walking network (grey) and Mapillary images — EPSG:{epsg}")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
