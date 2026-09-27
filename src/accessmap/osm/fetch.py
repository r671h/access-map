"""`accessmap fetch-osm`: walking graph + kerb/steps/barrier layers + tag summary (phase 1)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import networkx as nx

from accessmap.config import Settings
from accessmap.osm import overpass

log = logging.getLogger(__name__)

# Tags whose share of edges is reported in the summary.
SUMMARY_TAGS = ["surface", "smoothness", "width", "incline", "wheelchair", "sidewalk", "footway"]

LAYER_QUERIES = {
    "kerbs": 'node["kerb"]',
    "crossings": 'node["highway"="crossing"]',
    "barriers": 'node["barrier"]',
    "steps": 'way["highway"="steps"]',
}



def _tag(el: dict, key: str):
    return el.get("tags", {}).get(key)


# Local predicates matching LAYER_QUERIES; one element may belong to several layers.
LAYER_PREDICATES = {
    "kerbs": lambda el: el["type"] == "node" and _tag(el, "kerb") is not None,
    "crossings": lambda el: el["type"] == "node" and _tag(el, "highway") == "crossing",
    "barriers": lambda el: el["type"] == "node" and _tag(el, "barrier") is not None,
    "steps": lambda el: el["type"] == "way" and _tag(el, "highway") == "steps",
}


def first(v):
    """osmnx stores merged ways' tags as lists; use the first value."""
    if isinstance(v, list):
        return v[0] if v else None
    return v


def values(v) -> list:
    """All values of a possibly merged (list-valued) osmnx attribute."""
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def elements_to_geojson(elements: list[dict]) -> dict:
    """Overpass JSON (`out geom`) nodes and ways -> GeoJSON FeatureCollection."""
    features = []
    for el in elements:
        if el["type"] == "node":
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif el["type"] == "way" and el.get("geometry"):
            geom = {"type": "LineString",
                    "coordinates": [[p["lon"], p["lat"]] for p in el["geometry"]]}
        else:
            continue
        features.append({
            "type": "Feature",
            "id": f"{el['type']}/{el['id']}",
            "geometry": geom,
            "properties": {"osm_type": el["type"], "osm_id": el["id"], **el.get("tags", {})},
        })
    return {"type": "FeatureCollection", "features": features}


def split_layers(elements: list[dict]) -> dict[str, dict]:
    return {name: elements_to_geojson([el for el in elements if pred(el)])
            for name, pred in LAYER_PREDICATES.items()}


def fetch_layers(bbox) -> dict[str, dict]:
    """All layers in one Overpass request (public servers are slow; one round trip)."""
    b = overpass.overpass_bbox(bbox)
    union = "".join(f"{sel}({b});" for sel in LAYER_QUERIES.values())
    data = overpass.query(f"[out:json][timeout:170];({union});out geom;")
    layers = split_layers(data.get("elements", []))
    for name, fc in layers.items():
        log.info("OSM layer %s: %d features", name, len(fc["features"]))
    return layers


def undirected_edges(G: nx.MultiDiGraph) -> list[dict]:
    """One attribute dict per physical edge (the walk graph stores both directions)."""
    seen, out = set(), []
    for u, v, d in G.edges(data=True):
        osmid = d.get("osmid")
        # Reverse edges of merged ways may list the osmids in a different order.
        osmid = tuple(sorted(osmid)) if isinstance(osmid, list) else osmid
        key = (min(u, v), max(u, v), osmid, round(d.get("length", 0), 1))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def summarize(G_full: nx.MultiDiGraph, G: nx.MultiDiGraph, layers: dict[str, dict]) -> dict:
    """Numbers for reports/osm_summary.md. G is the graph used for routing (largest component)."""
    from accessmap.osm.network import component_sizes

    edges = undirected_edges(G)
    total_len = sum(d.get("length", 0) for d in edges)
    comps = component_sizes(G_full)
    full_len = sum(c[1] for c in comps)

    def share(tag: str) -> dict:
        tagged = [d for d in edges if first(d.get(tag)) is not None]
        return {
            "edges": round(len(tagged) / len(edges), 3) if edges else 0.0,
            "length": round(sum(d.get("length", 0) for d in tagged) / total_len, 3)
            if total_len else 0.0,
        }

    by_highway = Counter(first(d.get("highway")) for d in edges)
    surfaces = Counter(first(d.get("surface")) or "untagged" for d in edges)
    kerb_values = Counter(f["properties"].get("kerb") for f in layers["kerbs"]["features"])
    graph_kerb_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("kerb") is not None)
    steps = layers["steps"]["features"]

    return {
        "components": {
            "count": len(comps),
            "largest_share_of_length": round(comps[0][1] / full_len, 3) if full_len else 0.0,
            "top5": [{"nodes": n, "length_m": round(length)} for n, length in comps[:5]],
        },
        "graph": {
            "nodes": G.number_of_nodes(),
            "edges_directed": G.number_of_edges(),
            "edges_undirected": len(edges),
            "length_km": round(total_len / 1000, 2),
            "by_highway": dict(by_highway.most_common()),
            "edges_containing_steps": sum(1 for d in edges if "steps" in values(d.get("highway"))),
            "graph_nodes_with_kerb": graph_kerb_nodes,
            "crossing_nodes_in_graph": sum(1 for _, d in G.nodes(data=True)
                                           if d.get("highway") == "crossing"),
        },
        "tag_share": {t: share(t) for t in SUMMARY_TAGS},
        "surface_values": dict(surfaces.most_common()),
        "layers": {
            "kerb_nodes": len(layers["kerbs"]["features"]),
            "kerb_values": dict(kerb_values.most_common()),
            "crossing_nodes": len(layers["crossings"]["features"]),
            "barrier_nodes": len(layers["barriers"]["features"]),
            "barrier_values": dict(Counter(f["properties"].get("barrier")
                                           for f in layers["barriers"]["features"]).most_common()),
            "steps_ways": len(steps),
            "steps_with_step_count": sum(1 for f in steps if "step_count" in f["properties"]),
            "steps_with_ramp": sum(1 for f in steps
                                   if any(k.startswith("ramp") and v == "yes"
                                          for k, v in f["properties"].items())),
        },
    }


def summary_markdown(s: dict, area_name: str) -> str:
    g, c, lay = s["graph"], s["components"], s["layers"]
    tag_rows = "\n".join(f"| `{t}` | {v['edges']:.0%} | {v['length']:.0%} |"
                         for t, v in s["tag_share"].items())
    fmt = lambda d: ", ".join(f"{k}: {v}" for k, v in d.items()) or "—"  # noqa: E731
    comps = f"{c['count']} (largest = {c['largest_share_of_length']:.0%} of length)"
    steps = (f"with `step_count`: {lay['steps_with_step_count']}, "
             f"with a ramp: {lay['steps_with_ramp']}")
    return f"""# OSM summary — {area_name}

## Walking graph (largest connected component, used for routing)
| metric | value |
|---|---|
| nodes | {g['nodes']} |
| edges (undirected) | {g['edges_undirected']} |
| network length | {g['length_km']} km |
| components in the full download | {comps} |
| edges containing `highway=steps` | {g['edges_containing_steps']} |
| graph nodes with `kerb` | {g['graph_nodes_with_kerb']} |
| crossing nodes in graph | {g['crossing_nodes_in_graph']} |

Edges by `highway` (first value of merged edges): {fmt(g['by_highway'])}

## Tag completeness (share of edges / of length)
| tag | edges | length |
|---|---|---|
{tag_rows}

`surface` values: {fmt(s['surface_values'])}

## Point and way layers (whole bbox)
| layer | count | details |
|---|---|---|
| `kerb=*` nodes | {lay['kerb_nodes']} | {fmt(lay['kerb_values'])} |
| crossing nodes | {lay['crossing_nodes']} | |
| `barrier=*` nodes | {lay['barrier_nodes']} | {fmt(lay['barrier_values'])} |
| `highway=steps` ways | {lay['steps_ways']} | {steps} |
"""


def graph_to_geojson(G: nx.MultiDiGraph, path: Path) -> None:
    import osmnx as ox

    edges = ox.graph_to_gdfs(G, nodes=False).reset_index()
    for col in edges.columns:
        if col != "geometry" and edges[col].dtype == object:
            edges[col] = edges[col].map(lambda v: json.dumps(v) if isinstance(v, list) else v)
    edges.to_file(path, driver="GeoJSON")


def run(settings: Settings, refresh: bool = False) -> dict:
    import osmnx as ox

    from accessmap.osm.network import configure_osmnx, download_walk_graph, largest_component

    out = settings.paths.raw / "osm"
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / "walk_full.graphml"
    layer_paths = {name: out / f"{name}.geojson" for name in LAYER_QUERIES}
    configure_osmnx(settings)

    if graph_path.is_file() and not refresh:
        log.info("Using cached %s", graph_path)
        G_full = ox.load_graphml(graph_path)
    else:
        G_full = download_walk_graph(settings)
        ox.save_graphml(G_full, graph_path)

    if all(p.is_file() for p in layer_paths.values()) and not refresh:
        layers = {n: json.loads(p.read_text(encoding="utf-8")) for n, p in layer_paths.items()}
    else:
        layers = fetch_layers(settings.project.area.bbox)
        for name, fc in layers.items():
            layer_paths[name].write_text(json.dumps(fc), encoding="utf-8")

    G = largest_component(G_full)
    ox.save_graphml(G, out / "walk.graphml")
    graph_to_geojson(G, out / "walk_edges.geojson")

    s = summarize(G_full, G, layers)
    settings.paths.reports.mkdir(parents=True, exist_ok=True)
    (settings.paths.reports / "osm_summary.json").write_text(json.dumps(s, indent=2),
                                                             encoding="utf-8")
    (settings.paths.reports / "osm_summary.md").write_text(
        summary_markdown(s, settings.project.area.name), encoding="utf-8")
    return s
