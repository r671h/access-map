"""Baseline vs accessible routes on routing_graph.json (SPEC §8).

web/static/router.js is a line-by-line port of this module for the static site; keep the
two in step (tests/test_routing.py runs both on the same graph and compares).

Edge cost for a profile, over the part [a, b] of an edge that is walked (0 = node u):
    length × (b − a) × max(surface multiplier over the edge's surface values)
    + Σ over barrier types with a barrier inside [a, b] (highest confidence per type,
      rejected ones skipped):
        forbidden type: conf ≥ forbid_threshold → blocked, else + fallback_penalty_m
        other types:    + penalty_m[type] × conf   (negative = small bonus, e.g. curb_ramp)
    wheelchair=no on the edge blocks it when the profile's osm_rules say "forbid".
    Clamped to ≥ MIN_COST_FACTOR × walked length so bonuses never make costs negative.

Start and end snap to the nearest edge point from which the profile can leave: the walk
to one of the edge's ends is not blocked and that end is not a dead end (so a click next
to a stair-only stub does not start on the stairs). Such a point wins over the plain
nearest one only if it is at most SNAP_PREFER_USABLE_M further away; otherwise a real
"no way around" would be hidden by snapping far away. Both routes share these points.
If no path avoids blocked edges, they are allowed at VIOLATION_M each (fewest violations
first) and the answer carries the warning `no_barrier_free_path`.
"""

from __future__ import annotations

import heapq
import json
import math
import time
from pathlib import Path

MAX_SNAP_M = 250.0
SNAP_PREFER_USABLE_M = 30.0
MIN_COST_FACTOR = 0.1
VIOLATION_M = 100_000.0
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


class RoutingError(ValueError):
    """Bad request (point far from the network, unknown profile, ...)."""


class Router:
    def __init__(self, graph: dict):
        self.g = graph
        self.nodes = graph["nodes"]
        self.edges = graph["edges"]
        self.barriers = graph["barriers"]
        self.profiles = graph["profiles"]
        self.defaults = graph["defaults"]
        self.aliases = graph["surface_aliases"]
        self.max_detour = graph.get("max_detour")
        lat0 = sum(n[1] for n in self.nodes) / len(self.nodes)
        self.kx = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
        self.adj: list[list[tuple[int, int]]] = [[] for _ in self.nodes]
        for i, e in enumerate(self.edges):
            self.adj[e["u"]].append((i, e["v"]))
            self.adj[e["v"]].append((i, e["u"]))

    @classmethod
    def from_file(cls, path: Path) -> Router:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- geometry -----------------------------------------------------------------------
    def _xy(self, p: list[float]) -> tuple[float, float]:
        return p[0] * self.kx, p[1] * M_PER_DEG_LAT

    def project(self, edge: dict, lonlat: list[float]) -> tuple[float, float, list[float]]:
        """(distance m, t along the edge 0..1, closest point [lon, lat])."""
        px, py = self._xy(lonlat)
        pts = [self._xy(c) for c in edge["coords"]]
        seg_len = [math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(pts, pts[1:], strict=False)]
        total = sum(seg_len) or 1.0
        best = (math.inf, 0.0, edge["coords"][0])
        run = 0.0
        for i, (a, b) in enumerate(zip(pts, pts[1:], strict=False)):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L2 = dx * dx + dy * dy
            f = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / L2))
            qx, qy = a[0] + f * dx, a[1] + f * dy
            d = math.hypot(px - qx, py - qy)
            if d < best[0]:
                best = (d, (run + f * seg_len[i]) / total,
                        [qx / self.kx, qy / M_PER_DEG_LAT])
            run += seg_len[i]
        return best

    def snap(self, lonlat: list[float], can_leave) -> dict:
        """can_leave(edge_idx, t) → bool. See the module docstring."""
        cands = []
        for i, e in enumerate(self.edges):
            d, t, q = self.project(e, lonlat)
            cands.append({"edge": i, "t": t, "lonlat": q, "snap_m": d})
        cands.sort(key=lambda c: (c["snap_m"], c["edge"]))
        if not cands or cands[0]["snap_m"] > MAX_SNAP_M:
            raise RoutingError(f"point is more than {MAX_SNAP_M:.0f} m from the walking network")
        limit = min(cands[0]["snap_m"] + SNAP_PREFER_USABLE_M, MAX_SNAP_M)
        for c in cands:
            if c["snap_m"] > limit:
                break
            if can_leave(c["edge"], c["t"]):
                return c
        return cands[0]

    # -- costs --------------------------------------------------------------------------
    def surface_factor(self, edge: dict, profile: dict) -> float:
        table = profile["surface_multiplier"]
        unknown = table.get("unknown", 1.0)
        vals = [table.get(self.aliases.get(s, s), unknown) for s in edge["surface"]]
        return max(vals) if vals else unknown

    def edge_barriers(self, edge: dict, rejected: set[str], a: float = 0.0, b: float = 1.0
                      ) -> dict[str, dict]:
        """Highest-confidence active barrier per type within [a, b] of the edge."""
        lo, hi = min(a, b), max(a, b)
        out: dict[str, dict] = {}
        for bi in edge["barriers"]:
            br = self.barriers[bi]
            if br["id"] in rejected or not lo <= br["t"] <= hi:
                continue
            if br["type"] not in out or br["conf"] > out[br["type"]]["conf"]:
                out[br["type"]] = br
        return out

    def edge_cost(self, edge: dict, profile: dict, rejected: set[str], a: float = 0.0,
                  b: float = 1.0) -> tuple[float, bool]:
        """(cost, blocked) for walking the part [a, b] of the edge."""
        walked = edge["len"] * abs(b - a)
        cost = walked * self.surface_factor(edge, profile)
        blocked = False
        thr = self.defaults["forbid_threshold"]
        for t, br in self.edge_barriers(edge, rejected, a, b).items():
            if t in profile["forbidden"]:
                if br["conf"] >= thr:
                    blocked = True
                else:
                    cost += self.defaults["fallback_penalty_m"]
            else:
                cost += profile["penalty_m"].get(t, 0.0) * br["conf"]
        if edge["wheelchair_no"] and profile["osm_rules"].get("wheelchair_no") == "forbid":
            blocked = True
        return max(cost, MIN_COST_FACTOR * walked), blocked

    # -- search -------------------------------------------------------------------------
    def _search(self, s: dict, e: dict, cost_fn) -> dict | None:
        """Cheapest path between two snapped points. cost_fn(edge_idx, a, b) → float or
        None (unusable). Returns {"cost", "nodes", "edges"}; nodes/edges are the full edges
        between the partial start and end edges."""
        best: dict | None = None
        if s["edge"] == e["edge"]:  # walk along the shared edge only
            c = cost_fn(s["edge"], s["t"], e["t"])
            if c is not None:
                best = {"cost": c, "nodes": [], "edges": []}
        se, ee = self.edges[s["edge"]], self.edges[e["edge"]]
        dist: dict[int, float] = {}
        prev: dict[int, tuple[int, int] | None] = {}
        heap: list[tuple[float, int]] = []
        for node, a, b in ((se["u"], s["t"], 0.0), (se["v"], s["t"], 1.0)):
            c = cost_fn(s["edge"], a, b)
            if c is not None and c < dist.get(node, math.inf):
                dist[node] = c
                prev[node] = None
                heapq.heappush(heap, (c, node))
        ends = {}
        for node, a, b in ((ee["u"], 0.0, e["t"]), (ee["v"], 1.0, e["t"])):
            c = cost_fn(e["edge"], a, b)
            if c is not None:
                ends[node] = min(c, ends.get(node, math.inf))
        done = set()
        while heap:
            d, n = heapq.heappop(heap)
            if n in done:
                continue
            if best is not None and d >= best["cost"]:
                break
            done.add(n)
            if n in ends and (best is None or d + ends[n] < best["cost"]):
                best = {"cost": d + ends[n], "end_node": n}
            for ei, m in self.adj[n]:
                w = cost_fn(ei, 0.0, 1.0)
                if w is None:
                    continue
                nd = d + w
                if nd < dist.get(m, math.inf):
                    dist[m] = nd
                    prev[m] = (n, ei)
                    heapq.heappush(heap, (nd, m))
        if best is None:
            return None
        if "end_node" not in best:
            return best
        nodes, edges = [best["end_node"]], []
        while prev[nodes[-1]] is not None:
            n, ei = prev[nodes[-1]]
            edges.append(ei)
            nodes.append(n)
        return {"cost": best["cost"], "nodes": nodes[::-1], "edges": edges[::-1]}

    # -- describe a path ----------------------------------------------------------------
    def _pieces(self, s: dict, e: dict, path: dict) -> list[tuple[int, float, float]]:
        """The walked parts as (edge, a, b) in walking order."""
        if not path["nodes"]:
            return [(s["edge"], s["t"], e["t"])]
        first, last = path["nodes"][0], path["nodes"][-1]
        se, ee = self.edges[s["edge"]], self.edges[e["edge"]]
        pieces = [(s["edge"], s["t"], 0.0 if se["u"] == first else 1.0)]
        for n, ei in zip(path["nodes"], path["edges"], strict=False):
            ed = self.edges[ei]
            pieces.append((ei, 0.0, 1.0) if ed["u"] == n else (ei, 1.0, 0.0))
        pieces.append((e["edge"], 0.0 if ee["u"] == last else 1.0, e["t"]))
        return pieces

    def _sub_coords(self, edge: dict, a: float, b: float) -> list[list[float]]:
        """Polyline of the edge between fractions a and b (in that direction)."""
        pts = [self._xy(c) for c in edge["coords"]]
        cum = [0.0]
        for p, q in zip(pts, pts[1:], strict=False):
            cum.append(cum[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))
        total = cum[-1] or 1.0

        def at(f: float) -> list[float]:
            x = f * total
            for i in range(1, len(cum)):
                if x <= cum[i] or i == len(cum) - 1:
                    seg = cum[i] - cum[i - 1] or 1.0
                    r = max(0.0, min(1.0, (x - cum[i - 1]) / seg))
                    p, q = pts[i - 1], pts[i]
                    return [(p[0] + r * (q[0] - p[0])) / self.kx,
                            (p[1] + r * (q[1] - p[1])) / M_PER_DEG_LAT]
            return list(edge["coords"][-1])

        lo, hi = min(a, b), max(a, b)
        inner = [list(edge["coords"][i]) for i in range(1, len(cum) - 1)
                 if lo * total < cum[i] < hi * total]
        out = [at(lo)] + inner + [at(hi)]
        return out if a <= b else out[::-1]

    def _describe(self, pieces: list[tuple[int, float, float]], profile: dict,
                  rejected: set[str]) -> dict:
        coords: list[list[float]] = []
        length = 0.0
        found: list[dict] = []
        thr = self.defaults["forbid_threshold"]
        wheelchair_rule = profile["osm_rules"].get("wheelchair_no") == "forbid"
        for ei, a, b in pieces:
            ed = self.edges[ei]
            if a == b:
                continue
            seg = self._sub_coords(ed, a, b)
            coords.extend(seg if not coords else seg[1:])
            length += ed["len"] * abs(b - a)
            for t, br in self.edge_barriers(ed, rejected, a, b).items():
                forbidden = t in profile["forbidden"]
                found.append({"id": br["id"], "type": t, "conf": br["conf"],
                              "source": br["source"], "lonlat": br["lonlat"],
                              "problem": forbidden or profile["penalty_m"].get(t, 0.0) > 0,
                              "violation": forbidden and br["conf"] >= thr})
            if ed["wheelchair_no"] and wheelchair_rule:
                found.append({"id": f"osm-wheelchair-no-{ei}", "type": "wheelchair_no",
                              "conf": 1.0, "source": "osm", "lonlat": seg[0],
                              "problem": True, "violation": True})
        counts: dict[str, int] = {}
        for br in found:
            if br["problem"]:
                counts[br["type"]] = counts.get(br["type"], 0) + 1
        return {"length_m": round(length, 1), "coords": coords, "barriers": found,
                "problems": counts, "problem_count": sum(counts.values()),
                "violations": sum(br["violation"] for br in found),
                "pieces": [[ei, round(a, 4), round(b, 4)] for ei, a, b in pieces if a != b]}

    # -- public -------------------------------------------------------------------------
    def route(self, start: list[float], end: list[float], profile_name: str,
              rejected: set[str] | frozenset = frozenset()) -> dict:
        """start/end are [lon, lat]."""
        t0 = time.perf_counter()
        if profile_name not in self.profiles:
            raise RoutingError(f"unknown profile {profile_name!r}")
        profile = self.profiles[profile_name]
        rejected = set(rejected)
        full = [self.edge_cost(e, profile, rejected) for e in self.edges]

        def can_leave(ei, t):
            ed = self.edges[ei]
            return any(len(self.adj[node]) > 1
                       and not self.edge_cost(ed, profile, rejected, t, end)[1]
                       for node, end in ((ed["u"], 0.0), (ed["v"], 1.0)))

        s = self.snap(start, can_leave)
        e = self.snap(end, can_leave)
        if s["edge"] == e["edge"] and abs(s["t"] - e["t"]) * self.edges[s["edge"]]["len"] < 1:
            raise RoutingError("start and end are the same point of the network")
        warnings: list[dict] = []

        def length_cost(ei, a, b):
            return self.edges[ei]["len"] * abs(b - a)

        def profile_cost(relaxed):
            def fn(ei, a, b):
                if a == 0.0 and b == 1.0 or a == 1.0 and b == 0.0:
                    c, blocked = full[ei]
                else:
                    c, blocked = self.edge_cost(self.edges[ei], profile, rejected, a, b)
                if blocked:
                    return c + VIOLATION_M if relaxed else None
                return c
            return fn

        base = self._search(s, e, length_cost)
        if base is None:
            raise RoutingError("no path between these points in the walking network")
        acc = self._search(s, e, profile_cost(False))
        if acc is None:
            acc = self._search(s, e, profile_cost(True))
            warnings.append({"code": "no_barrier_free_path"})

        baseline = self._describe(self._pieces(s, e, base), profile, rejected)
        accessible = self._describe(self._pieces(s, e, acc), profile, rejected)
        ratio = (accessible["length_m"] / baseline["length_m"] - 1
                 if baseline["length_m"] else 0.0)
        if self.max_detour is not None and ratio > self.max_detour + 1e-9:
            warnings.append({"code": "detour_exceeds_limit", "detour": round(ratio, 3),
                             "limit": self.max_detour})
        acc_ids = {br["id"] for br in accessible["barriers"]}
        avoided: dict[str, int] = {}
        for br in baseline["barriers"]:
            if br["problem"] and br["id"] not in acc_ids:
                avoided[br["type"]] = avoided.get(br["type"], 0) + 1
        return {
            "profile": profile_name,
            "from": {"lonlat": s["lonlat"], "snap_m": round(s["snap_m"], 1)},
            "to": {"lonlat": e["lonlat"], "snap_m": round(e["snap_m"], 1)},
            "baseline": baseline, "accessible": accessible,
            "same_route": baseline["pieces"] == accessible["pieces"],
            "extra_m": round(accessible["length_m"] - baseline["length_m"], 1),
            "detour": round(ratio, 3),
            "avoided": avoided,
            "warnings": warnings,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
