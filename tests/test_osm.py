import json
from pathlib import Path

import networkx as nx
import pytest

from accessmap.osm import fetch
from accessmap.osm.network import component_sizes, largest_component

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_graph() -> nx.MultiDiGraph:
    """Main component a-b-c (both directions, like osmnx walk graphs) + a stray edge x-y."""
    G = nx.MultiDiGraph(crs="epsg:4326")
    for n, (x, y) in {"a": (8.530, 52.020), "b": (8.531, 52.020), "c": (8.532, 52.020),
                      "x": (8.540, 52.030), "y": (8.5401, 52.030)}.items():
        G.add_node(n, x=x, y=y)
    G.nodes["b"]["kerb"] = "lowered"
    G.nodes["b"]["highway"] = "crossing"

    def both(u, v, **d):
        G.add_edge(u, v, **d)
        G.add_edge(v, u, **d)

    both("a", "b", osmid=1, length=70.0, highway="footway", footway="sidewalk", surface="asphalt")
    both("b", "c", osmid=2, length=70.0, highway="residential", surface=["sett", "asphalt"])
    both("x", "y", osmid=3, length=7.0, highway="footway")
    # A merged edge whose reverse lists the osmids in the other order.
    G.add_edge("c", "d", osmid=[4, 5], length=20.0, highway=["footway", "steps"])
    G.add_edge("d", "c", osmid=[5, 4], length=20.0, highway=["steps", "footway"])
    G.add_node("d", x=8.5322, y=52.020)
    return G


def test_largest_component_by_length():
    G = fixture_graph()
    H = largest_component(G)
    assert set(H.nodes) == {"a", "b", "c", "d"}
    sizes = component_sizes(G)
    assert [n for n, _ in sizes] == [4, 2]


def test_undirected_edges_dedupes_directions():
    assert len(fetch.undirected_edges(fixture_graph())) == 4


def test_values_handles_osmnx_lists():
    assert fetch.values(["footway", "steps"]) == ["footway", "steps"]
    assert fetch.values("steps") == ["steps"]
    assert fetch.values(None) == []


def test_first_handles_osmnx_lists():
    assert fetch.first(["sett", "asphalt"]) == "sett"
    assert fetch.first("asphalt") == "asphalt"
    assert fetch.first([]) is None


def test_elements_to_geojson_from_fixture():
    data = json.loads((FIXTURES / "overpass_sample.json").read_text())
    fc = fetch.elements_to_geojson(data["elements"])
    kinds = [f["geometry"]["type"] for f in fc["features"]]
    assert kinds == ["Point", "Point", "LineString"]  # way without geometry is skipped
    assert fc["features"][0]["properties"]["kerb"] == "lowered"
    assert fc["features"][2]["geometry"]["coordinates"][0] == [8.532, 52.022]


def test_summarize_fixture():
    data = json.loads((FIXTURES / "overpass_sample.json").read_text())
    fc = fetch.elements_to_geojson(data["elements"])
    kerbs = {"type": "FeatureCollection",
             "features": [f for f in fc["features"] if "kerb" in f["properties"]]}
    steps = {"type": "FeatureCollection",
             "features": [f for f in fc["features"] if f["properties"].get("highway") == "steps"]}
    empty = {"type": "FeatureCollection", "features": []}
    G_full = fixture_graph()
    s = fetch.summarize(G_full, largest_component(G_full),
                        {"kerbs": kerbs, "crossings": empty, "barriers": empty, "steps": steps})

    assert s["graph"]["edges_undirected"] == 3
    assert s["graph"]["length_km"] == pytest.approx(0.16)
    assert s["graph"]["edges_containing_steps"] == 1
    assert s["components"]["count"] == 2
    assert s["components"]["largest_share_of_length"] == pytest.approx(320 / 334, abs=1e-3)
    assert s["tag_share"]["surface"]["edges"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["tag_share"]["footway"]["edges"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["surface_values"] == {"asphalt": 1, "sett": 1, "untagged": 1}
    assert s["graph"]["graph_nodes_with_kerb"] == 1
    assert s["layers"]["kerb_values"] == {"lowered": 1, "raised": 1}
    assert s["layers"]["steps_with_step_count"] == 1
    assert s["layers"]["steps_with_ramp"] == 1
    md = fetch.summary_markdown(s, "Fixture")
    assert "network length | 0.16 km" in md


def test_split_layers_assigns_elements_to_every_matching_layer():
    els = [
        {"type": "node", "id": 1, "lat": 0, "lon": 0,
         "tags": {"kerb": "lowered", "highway": "crossing"}},
        {"type": "node", "id": 2, "lat": 0, "lon": 0, "tags": {"barrier": "bollard"}},
        {"type": "way", "id": 3, "tags": {"highway": "steps"},
         "geometry": [{"lat": 0, "lon": 0}, {"lat": 1, "lon": 1}]},
    ]
    layers = fetch.split_layers(els)
    assert {n: len(fc["features"]) for n, fc in layers.items()} == {
        "kerbs": 1, "crossings": 1, "barriers": 1, "steps": 1}
