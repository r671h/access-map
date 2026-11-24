"""`accessmap build-graph` (phase 6): walking graph + barriers → routing_graph.json.

One compact JSON is the single input of both routers: the Python one (API, demo, tests)
and the browser one (web/static/router.js), so the local app and the static site route
the same way. Costs are not precomputed: they depend on the profile and on which
barriers users have rejected, so each router evaluates them per request.

Barrier sources on an edge:
- detected: snapped barriers from barriers.geojson (on the edge recorded when snapping);
- osm: `highway=steps` → stairs on that edge; kerb nodes (`kerb=raised` → raised_curb,
  `lowered|flush` → curb_ramp) on the nearest edge within KERB_SNAP_M.

Elevators (`highway=elevator` nodes): osmnx keeps an elevator only where it lies on a
walkable way, and then usually on one level, so the levels it serves are often joined by
stairs alone. For each elevator, the nearest graph node on each of its levels (from its
`level` tag, else every level found nearby) within ELEVATOR_LINK_M is linked to a new hub
node at the elevator by "elevator" edges. Edge levels come from OSM `level`; untagged
edges count as level 0.
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
ELEVATOR_LINK_M = 20.0
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


def parse_levels(value) -> set[float]:
    """OSM `level` values ("0", "-1;0", lists from merged ways) → set of levels.
    Ranges ("0-3") and other non-numeric parts are skipped."""
    out = set()
    for v in values(value):
        for part in str(v).split(";"):
            try:
                out.add(float(part.strip()))
            except ValueError:
                continue
    return out


def node_levels(G: nx.MultiDiGraph) -> dict:
    """node → levels of the edges that meet there (untagged edges: level 0)."""
    out: dict = {}
    for u, v, d in G.edges(data=True):
        lv = parse_levels(d.get("level")) or {0.0}
        out.setdefault(u, set()).update(lv)
        out.setdefault(v, set()).update(lv)
    return out


def elevator_links(G: nx.MultiDiGraph, elevators: list[dict],
                   radius_m: float = ELEVATOR_LINK_M) -> dict[str, list[tuple]]:
    """elevator osm_id → [(level, node, distance m)] ordered by level, for elevators that
    join at least two distinct graph nodes."""
    from accessmap.geo.snap import to_metric

    levels = node_levels(G)
    pts = [(n, *to_metric(d["x"], d["y"])) for n, d in G.nodes(data=True)]
    out = {}
    for f in elevators:
        ex, ey = to_metric(*f["geometry"]["coordinates"])
        wanted = parse_levels(f["properties"].get("level"))
        nearest: dict[float, tuple[float, str, object]] = {}
        for n, x, y in pts:
            dist = ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5
            if dist > radius_m:
                continue
            for lv in levels.get(n, ()):
                # Fewer than two tagged levels says nothing about where the lift goes.
                if len(wanted) >= 2 and lv not in wanted:
                    continue
                cand = (dist, str(n), n)
                if lv not in nearest or cand[:2] < nearest[lv][:2]:
                    nearest[lv] = cand
        chosen, seen = [], set()
        for lv in sorted(nearest):
            dist, _, n = nearest[lv]
            if n not in seen:
                seen.add(n)
                chosen.append((lv, n, dist))
        if len(chosen) >= 2:
            out[str(f["properties"]["osm_id"])] = chosen
    return out


def build(G: nx.MultiDiGraph, detected: list[dict], kerbs: list[dict], settings: Settings,
          elevators: list[dict] = ()) -> dict:
    """G: walking MultiDiGraph (both directions). detected: snapped barrier features.
    kerbs, elevators: OSM point features."""
    from accessmap.geo.snap import NetworkIndex, to_metric

    node_ids = list(G.nodes)
    node_idx = {n: i for i, n in enumerate(node_ids)}
    nodes = [[round(G.nodes[n]["x"], COORD_DIGITS), round(G.nodes[n]["y"], COORD_DIGITS)]
             for n in node_ids]
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

    # Elevators: a hub node per linked elevator, one edge from it to each level's node.
    # Edges get no surface (the profile's `unknown` factor) and are never snapped onto.
    links = elevator_links(G, list(elevators))
    elevator_list = []
    for f in elevators:
        p = f["properties"]
        eid = f"osm-elevator-{p['osm_id']}"
        hub = [round(c, COORD_DIGITS) for c in f["geometry"]["coordinates"]]
        link = links.get(str(p["osm_id"]))
        elevator_list.append({"id": eid, "lonlat": hub, "name": p.get("name"),
                              "level": p.get("level"), "wheelchair": p.get("wheelchair"),
                              "linked": link is not None})
        if link is None:
            continue
        nodes.append(hub)
        for _lv, n, dist in link:
            edges.append({
                "u": node_idx[n], "v": len(nodes) - 1, "len": round(max(dist, 1.0), 2),
                "surface": [], "wheelchair_no": p.get("wheelchair") == "no", "steps": False,
                "name": p.get("name"), "coords": [nodes[node_idx[n]], hub], "barriers": [],
                "elevator": eid,
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
        "nodes": nodes,
        "edges": edges,
        "barriers": barriers,
        "elevators": elevator_list,
        "profiles": {k: v.model_dump() for k, v in pf.profiles.items()},
        "defaults": pf.defaults.model_dump(),
        "surface_aliases": SURFACE_ALIASES,
        "max_detour": settings.project.routing.max_detour,
        "stats": {"nodes": len(nodes), "edges": len(edges), "barriers": len(barriers),
                  "detected_on_graph": len(detected) - unmatched,
                  "osm_steps_edges": sum(e["steps"] for e in edges),
                  "osm_kerbs": kerbs_used,
                  "wheelchair_no_edges": sum(e["wheelchair_no"] for e in edges),
                  "osm_elevators": len(elevator_list),
                  "elevators_linked": len(links)},
    }


def load_elevators(osm_dir: Path) -> list[dict]:
    path = osm_dir / "elevators.geojson"
    if not path.is_file():
        log.warning("%s is missing: run `accessmap fetch-osm` to route through elevators",
                    path)
        return []
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def graph_path(settings: Settings) -> Path:
    return settings.paths.processed / "routing_graph.json"


def run(settings: Settings) -> dict:
    import osmnx as ox

    osm = settings.paths.raw / "osm"
    G = ox.load_graphml(osm / "walk.graphml")
    detected = json.loads((settings.paths.processed / "barriers.geojson"
                           ).read_text(encoding="utf-8"))["features"]
    kerbs = json.loads((osm / "kerbs.geojson").read_text(encoding="utf-8"))["features"]
    g = build(G, detected, kerbs, settings, load_elevators(osm))
    graph_path(settings).write_text(json.dumps(g, ensure_ascii=False, separators=(",", ":")),
                                    encoding="utf-8", newline="\n")
    return g["stats"] | {"path": str(graph_path(settings))}
