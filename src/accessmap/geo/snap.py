"""Attach clustered barriers to the pedestrian network (SPEC §6 step 5).

Everything is done in a metric CRS (ETRS89 / UTM 32N covers Bielefeld).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely import LineString, Point, STRtree

METRIC_CRS = "EPSG:25832"
CURB_TYPES = {"curb_ramp", "raised_curb"}

_to_m = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
_to_deg = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)


def to_metric(lon, lat):
    return _to_m.transform(lon, lat)


def to_wgs84(x, y):
    return _to_deg.transform(x, y)


@dataclass
class Snap:
    lon: float
    lat: float
    distance_m: float
    target: str                  # crossing | node | edge
    edge: tuple | None           # (u, v, key) of the nearest network edge
    node: object | None = None   # graph node id when snapped to a node


class NetworkIndex:
    """Spatial index over the undirected edges and nodes of a walking graph, plus optional
    extra crossing points (OSM crossings that osmnx simplified away)."""

    def __init__(self, G: nx.MultiDiGraph, crossings: list[tuple[float, float]] = ()):
        self.edge_keys: list[tuple] = []
        lines = []
        seen = set()
        for u, v, k, d in G.edges(keys=True, data=True):
            if (v, u, k) in seen:
                continue
            seen.add((u, v, k))
            if "geometry" in d:
                xs, ys = d["geometry"].xy
            else:
                xs = [G.nodes[u]["x"], G.nodes[v]["x"]]
                ys = [G.nodes[u]["y"], G.nodes[v]["y"]]
            mx, my = to_metric(np.asarray(xs), np.asarray(ys))
            lines.append(LineString(np.column_stack([mx, my])))
            self.edge_keys.append((u, v, k))
        self.edges = STRtree(lines)
        self.lines = lines

        self.node_ids = list(G.nodes)
        nx_, ny_ = to_metric(np.array([G.nodes[n]["x"] for n in self.node_ids]),
                             np.array([G.nodes[n]["y"] for n in self.node_ids]))
        self.node_is_crossing = [G.nodes[n].get("highway") == "crossing" for n in self.node_ids]
        pts = [Point(x, y) for x, y in zip(nx_, ny_, strict=True)]
        cross = [Point(*to_metric(lon, lat)) for lon, lat in crossings]
        self.curb_points = pts + cross
        self.curb_kind = ["crossing" if c else "node" for c in self.node_is_crossing] + [
            "crossing"] * len(cross)
        self.curb_tree = STRtree(self.curb_points)

    def nearest_edge(self, p: Point, max_m: float) -> tuple[int, float] | None:
        idx = self.edges.query_nearest(p, max_distance=max_m, all_matches=False)
        if len(idx) == 0:
            return None
        i = int(idx[0])
        return i, float(self.lines[i].distance(p))

    def snap(self, lon: float, lat: float, btype: str, curb_m: float, other_m: float
             ) -> Snap | None:
        p = Point(*to_metric(lon, lat))
        if btype in CURB_TYPES:
            idx = self.curb_tree.query_nearest(p, max_distance=curb_m, all_matches=False)
            if len(idx) == 0:
                return None
            i = int(idx[0])
            q = self.curb_points[i]
            edge = self.nearest_edge(q, curb_m)
            node = self.node_ids[i] if i < len(self.node_ids) else None
            return Snap(*to_wgs84(q.x, q.y), float(q.distance(p)), self.curb_kind[i],
                        self.edge_keys[edge[0]] if edge else None, node)
        edge = self.nearest_edge(p, other_m)
        if edge is None:
            return None
        i, dist = edge
        line = self.lines[i]
        q = line.interpolate(line.project(p))
        return Snap(*to_wgs84(q.x, q.y), dist, "edge", self.edge_keys[i])
