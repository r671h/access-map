import pytest

from accessmap import http
from accessmap.imagery import mapillary
from accessmap.osm import overpass


def test_with_retries_retries_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise http.HttpError("busy", 503)
        return "ok"

    assert http.with_retries(flaky, retries=4, sleep=lambda s: None) == "ok"
    assert len(calls) == 3


def test_with_retries_gives_up_on_non_retryable():
    calls = []

    def bad():
        calls.append(1)
        raise http.HttpError("nope", 401)

    with pytest.raises(http.HttpError):
        http.with_retries(bad, retries=4, is_retryable=http._retryable_http,
                          sleep=lambda s: None)
    assert len(calls) == 1


def test_backoff_is_capped():
    assert all(0 <= http.backoff_delay(a, cap=10) <= 10 for a in range(20))


def test_tiles_cover_bbox_exactly():
    bbox = (8.526, 52.016, 8.544, 52.030)
    ts = list(mapillary.tiles(bbox))
    area = sum((t[2] - t[0]) * (t[3] - t[1]) for t in ts)
    assert area == pytest.approx((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    assert all((t[2] - t[0]) * (t[3] - t[1]) < 0.01 for t in ts)


def test_search_bbox_splits_full_tiles(monkeypatch):
    full_tile = (0.0, 0.0, 0.005, 0.005)

    def fake_search(token, bbox, fields, limit=mapillary.MAX_LIMIT, **kw):
        if bbox == full_tile:
            return [{"id": str(i)} for i in range(mapillary.MAX_LIMIT)]
        # Each quarter returns one unique image plus one shared duplicate.
        return [{"id": f"q{bbox}"}, {"id": "dup"}]

    monkeypatch.setattr(mapillary, "search_tile", fake_search)
    imgs = mapillary.search_bbox("MLY|t", full_tile)
    ids = {i["id"] for i in imgs}
    assert "dup" in ids and len(ids) == 5  # 4 quarters + 1 shared duplicate


def test_position_prefers_computed_geometry():
    img = {"geometry": {"coordinates": [1, 2]}, "computed_geometry": {"coordinates": [3, 4]}}
    assert mapillary.position(img) == (3.0, 4.0)
    assert mapillary.position({"geometry": {"coordinates": [1, 2]}}) == (1.0, 2.0)


class _Resp:
    def __init__(self, payload, ctype="application/json"):
        self._p, self.headers, self.text = payload, {"Content-Type": ctype}, str(payload)

    def json(self):
        return self._p


def test_overpass_falls_back_to_next_mirror(monkeypatch):
    seen = []

    def fake_request(method, url, **kw):
        seen.append(url)
        if url == overpass.ENDPOINTS[0]:
            return _Resp("<html>too busy</html>", ctype="text/html")
        return _Resp({"elements": [{"tags": {"total": "7"}}, {"tags": {"total": "2"}}]})

    monkeypatch.setattr(overpass, "request", fake_request)
    n = overpass.count((8.5, 52.0, 8.6, 52.1), {"kerb": 'node["kerb"]', "steps": 'way["x"]'})
    assert n == {"kerb": 7, "steps": 2}
    assert seen[:2] == list(overpass.ENDPOINTS[:2])


def test_overpass_bbox_order():
    assert overpass.overpass_bbox((8.5, 52.0, 8.6, 52.1)) == "52.0,8.5,52.1,8.6"
