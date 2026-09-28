import math

import networkx as nx
import pytest
from pyproj import Geod

from accessmap.geo import project
from accessmap.geo.cluster import cluster, combined_confidence
from accessmap.geo.snap import NetworkIndex

GEOD = Geod(ellps="WGS84")
CAM = (8.5335, 52.0240)
M_PER_DEG_LAT = 111_257.0  # at 52° N
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(52.024))


def offset(lon, lat, east_m, north_m):
    """Flat-earth offset; accurate to millimetres over tens of metres."""
    return lon + east_m / M_PER_DEG_LON, lat + north_m / M_PER_DEG_LAT


def err_m(a, b):
    return GEOD.inv(a[0], a[1], b[0], b[1])[2]


def feat(box, dist=None, conf=0.8):
    return {"box_2d": box, "confidence": conf, "estimated_distance_m": dist}


@pytest.mark.parametrize("heading,east,north", [(0, 0, 10), (90, 10, 0), (180, 0, -10),
                                                (270, -10, 0)])
def test_centred_box_lands_at_known_point(heading, east, north):
    p = project.place(feat([400, 450, 600, 550], dist=10), *CAM, heading, 90, 1000, 1000)
    assert err_m((p.lon, p.lat), offset(*CAM, east, north)) < 0.5
    assert p.distance_source == "gemini"


def test_box_at_right_quarter_uses_pinhole_bearing():
    # x = 0.75 in a 90° image → atan(0.5) = 26.57° right of the heading.
    p = project.place(feat([400, 700, 600, 800], dist=10), *CAM, 0, 90, 1000, 1000)
    b = math.radians(26.565)
    assert err_m((p.lon, p.lat), offset(*CAM, 10 * math.sin(b), 10 * math.cos(b))) < 0.5


def test_bearing_offset_edges_match_half_fov():
    assert project.bearing_offset(0.5, 90) == pytest.approx(0)
    assert project.bearing_offset(1.0, 90) == pytest.approx(45)
    assert project.bearing_offset(0.0, 70) == pytest.approx(-35)


def test_distance_from_box_bottom():
    # Square 90° image: bottom edge at y = 0.75 → 26.57° below horizon → 2 / tan = 4 m.
    assert project.distance_from_box_bottom(0.75, 90) == pytest.approx(4.0)
    assert project.distance_from_box_bottom(0.5, 90) is None
    assert project.distance_from_box_bottom(0.3, 90) is None


def test_missing_distance_falls_back_to_box_bottom_then_default():
    p = project.place(feat([500, 450, 750, 550]), *CAM, 0, 90, 1000, 1000)
    assert p.distance_source == "box_bottom" and p.distance_m == pytest.approx(4.0)
    q = project.place(feat([100, 450, 300, 550], conf=0.8), *CAM, 0, 90, 1000, 1000)
    assert q.distance_source == "fallback" and q.distance_m == 8.0
    assert q.confidence == pytest.approx(0.56)


def test_distance_is_clamped():
    assert project.place(feat([400, 450, 600, 550], dist=60), *CAM, 0, 90, 10, 10
                         ).distance_m == 25
    assert project.place(feat([400, 450, 600, 550], dist=0.5), *CAM, 0, 90, 10, 10
                         ).distance_m == 2


def test_vertical_fov_of_16_9_image():
    assert project.vertical_fov(90, 1920, 1080) == pytest.approx(58.72, abs=0.01)


def det(t, east, north, conf, frame, image=None, idx=0):
    lon, lat = offset(*CAM, east, north)
    return {"type": t, "lon": lon, "lat": lat, "confidence": conf, "frame_id": frame,
            "image_id": image or frame, "feature_index": idx}


def test_clustering_merges_nearby_views_and_keeps_types_apart():
    dets = [det("raised_curb", 0, 0, 0.6, "f1"), det("raised_curb", 3, 1, 0.7, "f2"),
            det("raised_curb", 1, -2, 0.5, "f3"),
            det("raised_curb", 40, 0, 0.9, "f4"),       # far away → own cluster
            det("curb_ramp", 1, 0, 0.8, "f1", idx=1)]   # same spot, other type
    out = {(c["type"], c["n_views"]): c for c in cluster(dets)}
    assert len(out) == 3
    group = out[("raised_curb", 3)]
    assert group["confidence"] == pytest.approx(1 - 0.4 * 0.3 * 0.5)
    assert err_m((group["lon"], group["lat"]), offset(*CAM, 1.44, -0.17)) < 0.5
    assert ("curb_ramp", 1) in out and ("raised_curb", 1) in out


def test_clustering_chains_within_eps_but_splits_beyond():
    # 5 m steps chain into one cluster (DBSCAN, min_samples 1); a 10 m gap splits.
    dets = [det("step", 0, 0, 0.8, "a"), det("step", 5, 0, 0.8, "b"),
            det("step", 10, 0, 0.8, "c"), det("step", 20, 0, 0.8, "d")]
    sizes = sorted(c["n_detections"] for c in cluster(dets))
    assert sizes == [1, 3]


def test_single_low_confidence_cluster_dropped():
    dets = [det("stairs", 0, 0, 0.45, "a"), det("stairs", 50, 0, 0.3, "b"),
            det("stairs", 52, 0, 0.3, "c")]
    out = cluster(dets)
    assert len(out) == 1 and out[0]["n_detections"] == 2
    assert out[0]["confidence"] == pytest.approx(0.51)


def test_confidence_counts_one_frame_once_and_is_capped():
    same_frame = [{"frame_id": "a", "confidence": 0.6}, {"frame_id": "a", "confidence": 0.7}]
    assert combined_confidence(same_frame) == pytest.approx(0.7)
    many = [{"frame_id": str(i), "confidence": 0.9} for i in range(5)]
    assert combined_confidence(many) == 0.97


def small_graph() -> nx.MultiDiGraph:
    """Footway a-b-c along the east axis from CAM (0, 50, 100 m); b is a crossing."""
    G = nx.MultiDiGraph(crs="epsg:4326")
    for n, e in (("a", 0), ("b", 50), ("c", 100)):
        lon, lat = offset(*CAM, e, 0)
        G.add_node(n, x=lon, y=lat)
    G.nodes["b"]["highway"] = "crossing"
    for u, v in (("a", "b"), ("b", "c")):
        G.add_edge(u, v, 0)
        G.add_edge(v, u, 0)
    return G


def test_snap_other_types_to_nearest_edge_point():
    idx = NetworkIndex(small_graph())
    s = idx.snap(*offset(*CAM, 30, 6), "step", curb_m=12, other_m=10)
    assert s.target == "edge" and s.distance_m == pytest.approx(6, abs=0.05)
    assert set(s.edge[:2]) == {"a", "b"}
    assert err_m((s.lon, s.lat), offset(*CAM, 30, 0)) < 0.1
    assert idx.snap(*offset(*CAM, 30, 11), "step", curb_m=12, other_m=10) is None


def test_snap_curbs_to_nodes_and_extra_crossings():
    extra = [offset(*CAM, 75, 3)]
    idx = NetworkIndex(small_graph(), extra)
    s = idx.snap(*offset(*CAM, 45, 5), "curb_ramp", curb_m=12, other_m=10)
    assert s.target == "crossing" and s.node == "b"
    s = idx.snap(*offset(*CAM, 78, 5), "raised_curb", curb_m=12, other_m=10)
    assert s.target == "crossing" and s.node is None and s.edge is not None
    # 25 m from any node: too far for a curb even though an edge is 2 m away.
    assert idx.snap(*offset(*CAM, 25, 2), "curb_ramp", curb_m=12, other_m=10) is None
