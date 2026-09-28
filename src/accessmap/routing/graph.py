"""`accessmap build-graph` (phase 6): walking graph + barriers → routing_graph.json.

One compact JSON is the single input of both routers: the Python one (API, demo, tests)
and the browser one (web/static/router.js), so the local app and the static site route
the same way. Costs are not precomputed: they depend on the profile and on which
barriers users have rejected, so each router evaluates them per request.

Barrier sources on an edge:
- detected: snapped barriers from barriers.geojson (on the edge recorded when snapping);
- osm: `highway=steps` → stairs on that edge; kerb nodes (`kerb=raised` → raised_curb,
  `lowered|flush` → curb_ramp) on the nearest edge within KERB_SNAP_M.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

from accessmap.config import Settings
from accessmap.osm.fetch import values
from accessmap.routing.profiles import SURFACE_ALIASES, load_profiles

log = logging.getLogger(__name__)

KERB_SNAP_M = 5.0
OSM_KERB_TYPES = {"raised": "raised_curb", "lowered": "curb_ramp", "flush": "curb_ramp"}
COORD_DIGITS = 6
FORMAT = 2  # 2: barriers carry "t", their position along the edge


def _edge_coords(G: nx.MultiDiGraph, u, v, d) -> list[list[float]]:
    """Coordinates oriented u → v (osmnx geometries may run either way)."""
    if "geometry" in d:
        coords = [list(c) for c in d["geometry"].coords]
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        first, last = coords[0], coords[-1]
        if (first[0] - ux) ** 2 + (first[1] - uy) ** 2 > (last[0] - ux) ** 2 + (last[1] - uy) ** 2:
            coords.reverse()
    else:
        coords = [[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]]
    return [[round(x, COORD_DIGITS), round(y, COORD_DIGITS)] for x, y in coords]


def _midpoint(coords: list[list[float]]) -> list[float]:
    from shapely import LineString

    p = LineString(coords).interpolate(0.5, normalized=True)
    return [round(p.x, COORD_DIGITS), round(p.y, COORD_DIGITS)]


def build(G: nx.MultiDiGraph, detected: list[dict], kerbs: list[dict], settings: Settings
          ) -> dict:
    """G: walking MultiDiGraph (both directions). detected: snapped barrier features.
    kerbs: OSM kerb point features."""
    from accessmap.geo.snap import NetworkIndex, to_metric

    node_ids = list(G.nodes)
    node_idx = {n: i for i, n in enumerate(node_ids)}
    edges, edge_idx = [], {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if (v, u, k) in edge_idx:
            edge_idx[(u, v, k)] = edge_idx[(v, u, k)]
            continue
        edge_idx[(u, v, k)] = len(edges)
        highway = [str(h) for h in values(d.get("highway"))]
        edges.append({
            "u": node_idx[u], "v": node_idx[v], "len": round(float(d["length"]), 2),
            "surface": sorted({str(s) for s in values(d.get("surface"))}),
            "wheelchair_no": "no" in [str(w) for w in values(d.get("wheelchair"))],
            "steps": "steps" in highway,
            "name": ", ".join(sorted({str(n) for n in values(d.get("name"))})) or None,
            "coords": _edge_coords(G, u, v, d),
            "barriers": [],
        })

    barriers = []

    def add(edge: int, b: dict) -> None:
        """b["t"]: position along the edge, 0 at node u, 1 at node v."""
        from shapely import LineString, Point

        line = LineString([to_metric(x, y) for x, y in edges[edge]["coords"]])
        b["t"] = round(line.project(Point(*to_metric(*b["lonlat"])), normalized=True), 4)
        b["edge"] = edge
        edges[edge]["barriers"].append(len(barriers))
        barriers.append(b)

    unmatched = 0
    for f in detected:
        p = f["properties"]
        key = (p.get("edge_u"), p.get("edge_v"), p.get("edge_key"))
        e = edge_idx.get(key)
        if e is None:
            unmatched += 1
            continue
        add(e, {"id": p["id"], "type": p["type"], "conf": p["confidence"], "source": "detected",
                "lonlat": [round(c, COORD_DIGITS) for c in f["geometry"]["coordinates"]]})
    if unmatched:
        log.warning("%d detected barriers reference edges not in the graph", unmatched)

    for i, e in enumerate(edges):
        if e["steps"]:
            add(i, {"id": f"osm-steps-{i}", "type": "stairs", "conf": 1.0, "source": "osm",
                    "lonlat": _midpoint(e["coords"])})

    index = NetworkIndex(G)
    kerbs_used = 0
    for k in kerbs:
        t = OSM_KERB_TYPES.get(k["properties"].get("kerb"))
        if t is None:
            continue
        from shapely import Point

        hit = index.nearest_edge(Point(*to_metric(*k["geometry"]["coordinates"])), KERB_SNAP_M)
        if hit is None:
            continue
        u, v, key = index.edge_keys[hit[0]]
        add(edge_idx[(u, v, key)], {
            "id": f"osm-kerb-{k['properties']['osm_id']}", "type": t, "conf": 1.0,
            "source": "osm", "lonlat": [round(c, COORD_DIGITS)
                                        for c in k["geometry"]["coordinates"]]})
        kerbs_used += 1

    pf = load_profiles(settings.root / "config" / "profiles.yaml", settings.project.profiles)
    return {
        "format": FORMAT,
        "nodes": [[round(G.nodes[n]["x"], COORD_DIGITS), round(G.nodes[n]["y"], COORD_DIGITS)]
                  for n in node_ids],
        "edges": edges,
        "barriers": barriers,
        "profiles": {k: v.model_dump() for k, v in pf.profiles.items()},
        "defaults": pf.defaults.model_dump(),
        "surface_aliases": SURFACE_ALIASES,
        "max_detour": settings.project.routing.max_detour,
        "stats": {"nodes": len(node_ids), "edges": len(edges), "barriers": len(barriers),
                  "detected_on_graph": len(detected) - unmatched,
                  "osm_steps_edges": sum(e["steps"] for e in edges),
                  "osm_kerbs": kerbs_used,
                  "wheelchair_no_edges": sum(e["wheelchair_no"] for e in edges)},
    }


def graph_path(settings: Settings) -> Path:
    return settings.paths.processed / "routing_graph.json"


def run(settings: Settings) -> dict:
    import osmnx as ox

    osm = settings.paths.raw / "osm"
    G = ox.load_graphml(osm / "walk.graphml")
    detected = json.loads((settings.paths.processed / "barriers.geojson"
                           ).read_text(encoding="utf-8"))["features"]
    kerbs = json.loads((osm / "kerbs.geojson").read_text(encoding="utf-8"))["features"]
    g = build(G, detected, kerbs, settings)
    graph_path(settings).write_text(json.dumps(g, ensure_ascii=False, separators=(",", ":")),
                                    encoding="utf-8", newline="\n")
    return g["stats"] | {"path": str(graph_path(settings))}
