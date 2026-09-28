"""Offline demo fixture (SPEC §11) and OSM suggestions (SPEC §10); no network, no keys."""

import json

import pytest
from fastapi.testclient import TestClient
from shapely import LineString, Point

from accessmap.config import Settings
from accessmap.demo import fixture_dir, unpack
from accessmap.osm.suggestions import suggestions


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    from accessmap.web.app import create_app

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
    root = unpack(fixture_dir(), tmp_path / "demo")
    return TestClient(create_app(Settings(root)))


def test_demo_serves_barriers_photos_routes_and_stats(demo_client):
    cfg = demo_client.get("/api/config").json()
    assert cfg["area"]["name"].startswith("Demo") and cfg["routing"]
    feats = demo_client.get("/api/barriers").json()["features"]
    assert len(feats) >= 20
    assert any(not f["properties"]["snapped"] for f in feats)  # both files unpacked
    # every photo a barrier points to is in the fixture
    for fid in {v["frame_id"] for f in feats for v in f["properties"]["views"]}:
        r = demo_client.get(f"/api/frames/{fid}/image")
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8", fid
    g = demo_client.get("/api/graph").json()
    a, b = g["nodes"][g["edges"][0]["u"]], g["nodes"][g["edges"][-1]["v"]]
    route = demo_client.get("/api/route", params={
        "from": f"{a[1]},{a[0]}", "to": f"{b[1]},{b[0]}", "profile": "wheelchair"})
    assert route.status_code == 200 and route.json()["accessible"]["length_m"] >= 0
    s = demo_client.get("/api/stats").json()
    assert s["barriers"] == len(feats) and s["precision"] is not None
    assert s["gemini_calls"]["used"] > 0
    assert demo_client.get("/stats.html").status_code == 200


def test_demo_feedback_goes_to_the_temporary_copy(demo_client, tmp_path):
    bid = demo_client.get("/api/barriers").json()["features"][0]["properties"]["id"]
    r = demo_client.post("/api/feedback", json={"barrier_id": bid, "verdict": "reject"})
    assert r.status_code == 201
    assert (tmp_path / "demo" / "data" / "feedback.sqlite").is_file()
    assert not (fixture_dir() / "data").exists()


def barrier(type_="curb_ramp", conf=0.9, views=2, lon=8.53, lat=52.02, **kw):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"id": f"{type_}-1", "type": type_, "confidence": conf,
                           "n_views": views, "best_frame_id": "123_front",
                           "description_en": "x", "last_seen": "2025-01-01 10:00", **kw}}


def test_osm_suggestions_rules():
    osm = {"kerb": [Point(8.53, 52.02008)],  # ~9 m north of the default barrier
           "steps": [LineString([(8.5302, 52.0210), (8.5302, 52.0212)])]}
    far = dict(lon=8.535, lat=52.025)
    feats, skipped = suggestions([
        barrier(**far),                                   # suggested
        barrier(),                                        # kerb already mapped within 10 m
        barrier(conf=0.7, **far),                         # low confidence
        barrier(views=1, **far),                          # one photo only
        barrier("rough_surface", **far),                  # no clear OSM tag
        barrier("curb_ramp", permanence="temporary", **far),
        barrier("stairs", lon=8.5302, lat=52.02135),      # ~17 m from mapped steps: same stairs
        barrier("stairs", **far),                         # suggested
    ], osm)
    assert [f["properties"]["suggested_tags"] for f in feats] == [
        {"kerb": "lowered"}, {"highway": "steps"}]
    assert skipped == {"no_tag": 1, "temporary": 1, "low_confidence": 1, "few_views": 1,
                       "already_in_osm": 2}
    p = feats[0]["properties"]
    assert p["mapillary"].endswith("pKey=123") and "check the photo" in p["instruction"]
    json.dumps(feats)  # plain JSON (MapRoulette challenge file)
