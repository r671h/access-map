import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from accessmap.web.app import create_app


def barrier(bid, btype, conf, snapped, frame="f1"):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [8.53, 52.02]},
            "properties": {"id": bid, "type": btype, "confidence": conf, "snapped": snapped,
                           "n_views": 1, "best_frame_id": frame,
                           "views": [{"frame_id": frame, "box_2d": [1, 2, 3, 4],
                                      "confidence": conf}]}}


@pytest.fixture
def client(settings):
    s = settings
    s.paths.ensure()
    (s.paths.raw / "osm").mkdir(parents=True)
    fc = lambda feats: json.dumps({"type": "FeatureCollection", "features": feats})  # noqa: E731
    (s.paths.processed / "barriers.geojson").write_text(fc([
        barrier("stairs-1", "stairs", 0.9, True),
        barrier("curb_ramp-1", "curb_ramp", 0.6, True)]), encoding="utf-8")
    (s.paths.processed / "barriers_unsnapped.geojson").write_text(fc([
        barrier("raised_curb-1", "raised_curb", 0.8, False)]), encoding="utf-8")
    (s.paths.raw / "osm" / "walk_edges.geojson").write_text(fc([]), encoding="utf-8")
    (s.paths.images).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30)).save(s.paths.images / "f1.jpg")
    pd.DataFrame([{"frame_id": "f1", "image_id": "123", "lon": 8.53, "lat": 52.02,
                   "heading": 90.0, "captured_at": pd.Timestamp("2025-06-01", tz="UTC"),
                   "is_pano": False, "path": "data/images/f1.jpg"}]
                 ).to_parquet(s.paths.processed / "frames.parquet")
    return TestClient(create_app(s))


def ids(resp):
    return sorted(f["properties"]["id"] for f in resp.json()["features"])


def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "maplibre" in r.text


def test_config(client):
    c = client.get("/api/config").json()
    assert c["types"] == ["curb_ramp", "raised_curb", "stairs"]
    assert c["area"]["bbox"] == [8.526, 52.016, 8.544, 52.030]


def test_barriers_filters(client):
    assert ids(client.get("/api/barriers")) == ["curb_ramp-1", "raised_curb-1", "stairs-1"]
    assert ids(client.get("/api/barriers?types=stairs,curb_ramp")) == ["curb_ramp-1",
                                                                        "stairs-1"]
    assert ids(client.get("/api/barriers?min_conf=0.7")) == ["raised_curb-1", "stairs-1"]
    assert ids(client.get("/api/barriers?include_unsnapped=false")) == ["curb_ramp-1",
                                                                         "stairs-1"]
    assert client.get("/api/barriers?min_conf=2").status_code == 422


def test_network(client):
    assert client.get("/api/network").json()["type"] == "FeatureCollection"


def test_frame_and_image(client):
    f = client.get("/api/frames/f1").json()
    assert f["image_id"] == "123" and f["mapillary_url"].endswith("pKey=123")
    r = client.get("/api/frames/f1/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/frames/nope").status_code == 404
    assert client.get("/api/frames/..%2F..%2Fconfig/image").status_code == 404


def test_feedback_flow_and_rejected_filter(client):
    r = client.post("/api/feedback", json={"barrier_id": "stairs-1", "verdict": "reject",
                                           "comment": "only a kerb"})
    assert r.status_code == 201 and r.json()["barrier"]["status"] == "rejected"
    feats = {f["properties"]["id"]: f for f in client.get("/api/barriers").json()["features"]}
    assert feats["stairs-1"]["properties"]["feedback"]["reject"] == 1
    assert ids(client.get("/api/barriers?include_rejected=false")) == ["curb_ramp-1",
                                                                        "raised_curb-1"]
    client.post("/api/feedback", json={"barrier_id": "stairs-1", "verdict": "confirm"})
    client.post("/api/feedback", json={"barrier_id": "stairs-1", "verdict": "confirm"})
    assert client.get("/api/stats").json()["feedback"] == {"barriers": 1, "confirmed": 1,
                                                          "rejected": 0}


def test_feedback_validation(client):
    assert client.post("/api/feedback", json={"barrier_id": "nope",
                                              "verdict": "reject"}).status_code == 404
    assert client.post("/api/feedback", json={"barrier_id": "stairs-1",
                                              "verdict": "maybe"}).status_code == 422
    assert client.post("/api/feedback", json={"barrier_id": "stairs-1", "verdict": "confirm",
                                              "comment": "x" * 501}).status_code == 422


def test_stats(client):
    s = client.get("/api/stats").json()
    assert s["barriers"] == 3 and s["snapped"] == 2 and s["frames"] == 1
    assert s["by_type"]["raised_curb"] == {"total": 1, "unsnapped": 1}
    assert s["gemini_calls"]["used"] == 0
