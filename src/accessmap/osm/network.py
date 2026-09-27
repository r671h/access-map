"""Walking network from OSM via osmnx, keeping the accessibility tags (SPEC §3.1)."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import networkx as nx

from accessmap.config import Settings
from accessmap.osm.overpass import ENDPOINTS

log = logging.getLogger(__name__)

EDGE_TAGS = [
    "highway", "footway", "sidewalk", "crossing", "kerb", "surface", "smoothness", "width",
    "incline", "wheelchair", "step_count", "name", "service", "access", "foot", "tunnel",
    "bridge", "level", "indoor", "area",
]
NODE_TAGS = ["highway", "crossing", "kerb", "barrier", "wheelchair", "tactile_paving", "ref"]


def configure_osmnx(settings: Settings) -> None:
    import osmnx as ox

    ox.settings.cache_folder = str(settings.paths.cache / "osmnx")
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180
    ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | set(EDGE_TAGS))
    ox.settings.useful_tags_node = sorted(set(ox.settings.useful_tags_node) | set(NODE_TAGS))


def download_walk_graph(settings: Settings, endpoints: Iterable[str] = ENDPOINTS
                        ) -> nx.MultiDiGraph:
    """Walking graph for the project bbox; tries each Overpass mirror in turn."""
    import osmnx as ox

    configure_osmnx(settings)
    bbox = settings.project.area.bbox
    last = None
    for url in endpoints:
        ox.settings.overpass_url = url.removesuffix("/interpreter")
        try:
            return ox.graph_from_bbox(bbox, network_type="walk", simplify=True, retain_all=True)
        except Exception as e:  # noqa: BLE001 - try the next mirror
            log.warning("osmnx via %s failed: %s", url, e)
            last = e
    raise RuntimeError(f"could not download walking network: {last}")


def component_sizes(G: nx.MultiDiGraph) -> list[tuple[int, float]]:
    """(node count, total edge length m) per weakly connected component, largest first."""
    out = []
    for nodes in nx.weakly_connected_components(G):
        length = sum(d.get("length", 0) for _, _, d in G.subgraph(nodes).edges(data=True))
        out.append((len(nodes), length))
    return sorted(out, key=lambda c: c[1], reverse=True)


def largest_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """The weakly connected component with the most edge length."""
    best, best_len = None, -1.0
    for nodes in nx.weakly_connected_components(G):
        length = sum(d.get("length", 0) for _, _, d in G.subgraph(nodes).edges(data=True))
        if length > best_len:
            best, best_len = nodes, length
    return G.subgraph(best).copy()
