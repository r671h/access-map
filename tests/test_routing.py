"""Routing on a synthetic graph (no network, no data files).

Layout (east = x metres, north = y metres from an origin; every node listed):

    A(0,0) --- B(100,0) --- C(200,0)          direct way A→C: 200 m, stairs on A–B
      |                       |
    D(0,-60) ------------- E(200,-60)          detour A→D→E→C: 320 m, barrier-free
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from accessmap.routing.router import Router, RoutingError

ORIGIN = (8.53, 52.02)
KX = 111_320.0 * math.cos(math.radians(52.02))
KY = 110_574.0
PROFILES = {
    "wheelchair": {"label": "Wheelchair", "forbidden": ["stairs", "raised_curb"],
                   "penalty_m": {"rough_surface": 150, "curb_ramp": -5},
                   "surface_multiplier": {"asphalt": 1.0, "sett": 1.8, "unknown": 1.15},
                   "osm_rules": {"wheelchair_no": "forbid"}},
    "suitcase": {"label": "Suitcase", "forbidden": [], "penalty_m": {"stairs": 250},
                 "surface_multiplier": {"asphalt": 1.0, "unknown": 1.1},
                 "osm_rules": {"wheelchair_no": "ignore"}},
}


def ll(x, y):
    return [ORIGIN[0] + x / KX, ORIGIN[1] + y / KY]


def make_graph(detour=True, stairs_conf=1.0, max_detour=0.5, extra_barriers=()):
    pos = {"A": (0, 0), "B": (100, 0), "C": (200, 0), "D": (0, -60), "E": (200, -60)}
    names = list(pos)
    links = [("A", "B"), ("B", "C")] + ([("A", "D"), ("D", "E"), ("E", "C")] if detour else [])
    edges = []
    for u, v in links:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        edges.append({"u": names.index(u), "v": names.index(v),
                      "len": math.hypot(x1 - x0, y1 - y0), "surface": ["asphalt"],
                      "wheelchair_no": False, "steps": False, "name": f"{u}{v}",
                      "coords": [ll(x0, y0), ll(x1, y1)], "barriers": []})
    barriers = [{"id": "stairs-1", "type": "stairs", "conf": stairs_conf,
                 "source": "detected", "lonlat": ll(50, 0), "t": 0.5, "edge": 0}]
    barriers += list(extra_barriers)
    for i, b in enumerate(barriers):
        edges[b["edge"]]["barriers"].append(i)
    return {"format": 2, "nodes": [ll(*pos[n]) for n in names], "edges": edges,
            "barriers": barriers, "profiles": PROFILES,
            "defaults": {"forbid_threshold": 0.6, "fallback_penalty_m": 400,
                         "temporary_factor": 0.5, "temporary_max_age_days": 365},
            "surface_aliases": {"fine_gravel": "gravel"}, "max_detour": max_detour}


A, C = ll(-5, 3), ll(205, 3)   # clicks a few metres off nodes A and C


def test_wheelchair_avoids_stairs():
    r = Router(make_graph()).route(A, C, "wheelchair")
    assert r["baseline"]["length_m"] == pytest.approx(200, abs=1)
    assert r["baseline"]["problems"] == {"stairs": 1}
    assert r["accessible"]["length_m"] == pytest.approx(320, abs=1)
    assert r["accessible"]["problems"] == {} and r["accessible"]["violations"] == 0
    assert r["avoided"] == {"stairs": 1} and not r["same_route"]
    assert r["warnings"] == [{"code": "detour_exceeds_limit", "detour": 0.6, "limit": 0.5}]


def test_detour_within_limit_has_no_warning():
    r = Router(make_graph(max_detour=1.0)).route(A, C, "wheelchair")
    assert r["warnings"] == []


def test_no_way_around_returns_route_with_warning():
    r = Router(make_graph(detour=False)).route(A, C, "wheelchair")
    assert r["accessible"]["violations"] == 1
    assert {"code": "no_barrier_free_path"} in r["warnings"]


def test_rejected_barrier_has_no_effect():
    r = Router(make_graph()).route(A, C, "wheelchair", rejected={"stairs-1"})
    assert r["same_route"] and r["accessible"]["length_m"] == pytest.approx(200, abs=1)
    assert r["baseline"]["problems"] == {} and r["warnings"] == []


def test_low_confidence_forbidden_barrier_is_a_penalty_not_a_block():
    # conf 0.5 < 0.6: +400 m instead of blocking; the 120 m detour is cheaper.
    r = Router(make_graph(stairs_conf=0.5, max_detour=1.0)).route(A, C, "wheelchair")
    assert r["accessible"]["length_m"] == pytest.approx(320, abs=1)
    g = make_graph(stairs_conf=0.5, detour=False)
    r = Router(g).route(A, C, "wheelchair")
    assert r["accessible"]["violations"] == 0 and r["warnings"] == []


def test_suitcase_penalty_trades_against_detour():
    # stairs penalty 250 > 120 m detour → detour; conf 0.4 → 100 < 120 → stairs.
    assert Router(make_graph()).route(A, C, "suitcase")["accessible"]["length_m"] == \
        pytest.approx(320, abs=1)
    assert Router(make_graph(stairs_conf=0.4)).route(A, C, "suitcase")["same_route"]


def test_barrier_outside_walked_part_of_edge_is_ignored():
    # Start and end between A and B, both west of the stairs at x=50.
    r = Router(make_graph()).route(ll(10, 2), ll(40, 2), "wheelchair")
    assert r["same_route"] and r["accessible"]["problems"] == {}
    assert r["accessible"]["length_m"] == pytest.approx(30, abs=1)


def test_snaps_to_usable_edge_not_to_stair_stub():
    # A stair-only stub F reaches up from B; a click right next to F must not start on it.
    g = make_graph()
    g["nodes"].append(ll(100, 30))
    g["edges"].append({"u": 1, "v": 5, "len": 30.0, "surface": ["asphalt"],
                       "wheelchair_no": False, "steps": True, "name": "BF",
                       "coords": [ll(100, 0), ll(100, 30)], "barriers": [len(g["barriers"])]})
    g["barriers"].append({"id": "osm-steps-5", "type": "stairs", "conf": 1.0,
                          "source": "osm", "lonlat": ll(100, 15), "t": 0.5, "edge": 5})
    r = Router(g).route(ll(103, 28), C, "wheelchair")
    assert r["accessible"]["violations"] == 0
    assert r["from"]["snap_m"] > 20   # snapped to B–C, not onto the stairs


def test_wheelchair_no_osm_rule():
    g = make_graph(max_detour=1.0)
    g["barriers"] = []
    for e in g["edges"]:
        e["barriers"] = []
    g["edges"][1]["wheelchair_no"] = True
    assert Router(g).route(A, C, "wheelchair")["accessible"]["length_m"] == \
        pytest.approx(320, abs=1)
    assert Router(g).route(A, C, "suitcase")["same_route"]


def test_surface_multiplier_uses_worst_value_and_aliases():
    router = Router(make_graph())
    edge = {"surface": ["asphalt", "sett"]}
    assert router.surface_factor(edge, PROFILES["wheelchair"]) == 1.8
    assert router.surface_factor({"surface": []}, PROFILES["wheelchair"]) == 1.15
    assert router.surface_factor({"surface": ["fine_gravel"]}, PROFILES["wheelchair"]) == 1.15


def test_errors():
    router = Router(make_graph())
    with pytest.raises(RoutingError):
        router.route(A, C, "bicycle")
    with pytest.raises(RoutingError):
        router.route(ll(0, 5000), C, "wheelchair")
    with pytest.raises(RoutingError):
        router.route(A, ll(-5, 3), "wheelchair")


def test_route_is_fast():
    router = Router(make_graph())
    assert router.route(A, C, "wheelchair")["ms"] < 1000


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_router_matches_python(tmp_path):
    """web/static/router.js must give the same answers as the Python router."""
    cases = []
    graphs = {"detour": make_graph(), "nodetour": make_graph(detour=False),
              "lowconf": make_graph(stairs_conf=0.5, max_detour=1.0)}
    for gname in graphs:
        for prof in PROFILES:
            for rejected in ([], ["stairs-1"]):
                cases.append({"graph": gname, "profile": prof, "rejected": rejected,
                              "start": A, "end": C})
    cases.append({"graph": "detour", "profile": "wheelchair", "rejected": [],
                  "start": ll(10, 2), "end": ll(40, 2)})
    (tmp_path / "graphs.json").write_text(json.dumps(graphs), encoding="utf-8")
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    shutil.copy(Path(__file__).parents[1] / "src/accessmap/web/static/router.js", tmp_path)
    (tmp_path / "run.cjs").write_text("""
const fs = require("fs");
require("./router.js");
const graphs = JSON.parse(fs.readFileSync("graphs.json"));
const cases = JSON.parse(fs.readFileSync("cases.json"));
const out = cases.map(c => new globalThis.AccessRouter(graphs[c.graph])
  .route(c.start, c.end, c.profile, new Set(c.rejected)));
fs.writeFileSync("out.json", JSON.stringify(out));
""", encoding="utf-8")
    p = subprocess.run(["node", "run.cjs"], cwd=tmp_path, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == 0, p.stderr
    js = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    for case, jr in zip(cases, js, strict=True):
        pr = Router(graphs[case["graph"]]).route(case["start"], case["end"], case["profile"],
                                                 set(case["rejected"]))
        for key in ("same_route", "extra_m", "detour", "avoided", "warnings"):
            assert jr[key] == pr[key], (case, key, jr[key], pr[key])
        for side in ("baseline", "accessible"):
            for key in ("length_m", "problems", "violations", "pieces"):
                assert jr[side][key] == pr[side][key], (case, side, key)
