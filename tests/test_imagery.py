import math

import numpy as np
import pytest
from PIL import Image

from accessmap.eval.contact import Box, Tile, box_to_pixels, contact_sheet
from accessmap.http import HttpError
from accessmap.imagery import fetch
from accessmap.imagery.pano import crop_offsets, perspective_crops
from accessmap.imagery.select import Candidate, hfov_deg, sector, spread, thin


def cand(i, x, y, heading=0.0, t=0, pano=False, on_foot=False):
    return Candidate(str(i), x, y, heading, t, pano, on_foot)


def test_hfov_from_focal_fraction():
    fov, defaulted = hfov_deg([0.5, 0, 0])  # focal = half the width -> 90°
    assert fov == pytest.approx(90) and not defaulted
    assert hfov_deg(None) == (70.0, True)
    assert hfov_deg([0.0]) == (70.0, True)


def test_sector_boundaries():
    assert sector(0) == 0 and sector(22) == 0 and sector(23) == 1
    assert sector(359) == 0 and sector(180) == 4 and sector(None) == -1


def test_thin_keeps_newest_per_cell_and_sector():
    cs = [cand(1, 1, 1, 10, t=1), cand(2, 2, 2, 20, t=5),  # same cell+sector -> keep t=5
          cand(3, 2, 2, 190, t=1),                          # same cell, opposite heading
          cand(4, 20, 20, 10, t=1),                         # other cell
          cand(5, 3, 3, pano=True, t=2), cand(6, 4, 4, 100, pano=True, t=3)]  # panos share
    kept = {c.image_id for c in thin(cs)}
    assert kept == {"2", "3", "4", "6"}


def test_thin_prefers_on_foot_on_ties():
    kept = thin([cand(1, 1, 1, t=5), cand(2, 1, 1, t=5, on_foot=True)])
    assert [c.image_id for c in kept] == ["2"]


def test_spread_round_robins_blocks():
    dense = [cand(f"d{i}", 1 + i * 0.1, 1, t=100 + i) for i in range(10)]  # one block
    sparse = [cand("s1", 100, 100, t=1), cand("s2", 200, 200, t=1)]
    out = spread(dense + sparse, cap=4)
    ids = [c.image_id for c in out]
    assert "s1" in ids and "s2" in ids and len(out) == 4
    assert ids[0] == "d9"  # newest overall first
    assert len(spread(dense + sparse, cap=100)) == 12


def _equirect() -> Image.Image:
    """Centre column red (forward), +90° green (right), -90° blue (left), edges yellow (back)."""
    w, h = 720, 360
    lon = (np.arange(w) + 0.5) / w * 360 - 180
    img = np.zeros((h, w, 3), np.uint8)
    img[:, np.abs(lon) < 30] = (255, 0, 0)
    img[:, np.abs(lon - 90) < 30] = (0, 255, 0)
    img[:, np.abs(lon + 90) < 30] = (0, 0, 255)
    img[:, np.abs(lon) > 150] = (255, 255, 0)
    return Image.fromarray(img)


def test_pano_crops_directions_and_bearings():
    crops = perspective_crops(_equirect(), heading=350, n=4, fov=60, out_px=32)
    centre = {name: tuple(np.asarray(im)[16, 16]) for name, _, im in crops}
    assert centre == {"forward": (255, 0, 0), "right": (0, 255, 0),
                      "back": (255, 255, 0), "left": (0, 0, 255)}
    assert [round(b) for _, b, _ in crops] == [350, 80, 170, 260]


def test_crop_offsets_generic():
    assert [o for _, o in crop_offsets(6)] == [0, 60, 120, 180, 240, 300]


def test_box_to_pixels_uses_gemini_order():
    # [ymin, xmin, ymax, xmax] -> (x0, y0, x1, y1)
    assert box_to_pixels([100, 200, 300, 400], 1000, 500) == (200, 50, 400, 150)


def test_contact_sheet_renders(tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (400, 300), (90, 90, 90)).save(p)
    out = contact_sheet([Tile(p, "cap", [Box([500, 100, 900, 600], "raised_curb 0.8",
                                              "raised_curb")])] * 3,
                        tmp_path / "sheet.jpg", cols=2, title="t")
    im = Image.open(out)
    assert im.size[0] == 2 * 480 and im.size[1] > 2 * 360


def test_angle_diff_wraps():
    assert fetch.angle_diff(350, 10) == 20
    assert fetch.angle_diff(0, 180) == 180


def test_travel_bearing_from_sequence():
    def img(i, lon, lat, t):
        return {"id": i, "sequence": "s", "captured_at": t,
                "geometry": {"coordinates": [lon, lat]}}
    seq = [img("a", 8.0, 52.0, 1), img("b", 8.0, 52.0001, 2), img("c", 8.0, 52.0002, 3)]
    b = fetch.travel_bearing(seq[1], {"s": seq})
    assert b == pytest.approx(0, abs=0.1) or b == pytest.approx(360, abs=0.1)
    east = [img("a", 8.0, 52.0, 1), img("b", 8.0002, 52.0, 2)]
    assert fetch.travel_bearing(east[0], {"s": east}) == pytest.approx(90, abs=0.1)


class _R:
    content = b"jpegbytes"


def test_download_refreshes_expired_url(tmp_path, monkeypatch):
    calls = []

    def fake_request(method, url, **kw):
        calls.append(url)
        if url == "https://old":
            raise HttpError("expired", 403)
        return _R()

    monkeypatch.setattr(fetch, "request", fake_request)
    monkeypatch.setattr(fetch, "refresh_thumb_url", lambda token, iid: "https://new")
    dest = tmp_path / "x.jpg"
    assert fetch.download_one({"id": 1, "thumb_2048_url": "https://old"}, dest, "MLY|t")
    assert dest.read_bytes() == b"jpegbytes" and calls == ["https://old", "https://new"]
    # Second run downloads nothing.
    assert fetch.download_one({"id": 1, "thumb_2048_url": "https://old"}, dest, "MLY|t") is False


def test_heading_prefers_computed():
    assert fetch.heading_of({"computed_compass_angle": 370.0, "compass_angle": 5}) == 10
    assert fetch.heading_of({"compass_angle": 5}) == 5
    assert fetch.heading_of({}) is None
    assert math.isclose(fetch.heading_of({"computed_compass_angle": 0.0}), 0.0)


def test_major_road_sequences():
    from accessmap.imagery.select import major_road_sequences

    seqs = ["a", "a", "a", "b", "b", None]
    d = np.array([1.0, 2.0, 50.0, 40.0, 3.0, 1.0])
    assert major_road_sequences(seqs, d) == {"a", "b"}  # b is exactly 50% -> dropped
    assert major_road_sequences(seqs, d, share=0.6) == {"a"}
