from datetime import UTC, datetime

import numpy as np
import pytest

from accessmap import checks, coverage


def test_checks_report_missing_keys(settings):
    g = checks.check_gemini(settings)
    m = checks.check_mapillary(settings)
    assert not g.ok and "GEMINI_API_KEY" in g.detail
    assert not m.ok and "MAPILLARY_TOKEN" in m.detail


def test_mapillary_token_format(settings):
    settings.secrets.mapillary_token = "abc"
    r = checks.check_mapillary(settings)
    assert not r.ok and "MLY|" in r.detail


def test_format_table():
    t = checks.format_table([checks.CheckResult("Gemini", True, "fine"),
                             checks.CheckResult("Overpass", False, "down")])
    assert "OK" in t and "FAIL" in t and "down" in t


def test_utm_epsg_bielefeld():
    assert coverage.utm_epsg(8.53, 52.02) == 32632


def test_image_age():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    assert coverage.image_age_years(ts, now) == pytest.approx(2.0, abs=0.01)


def test_sample_lines_weights_sum_to_length():
    lines = [np.array([[0, 0], [100, 0]]), np.array([[0, 0], [30, 40]])]
    pts, w = coverage.sample_lines(lines, step=5)
    assert w.sum() == pytest.approx(150)
    assert len(pts) == 20 + 10


def test_covered_share():
    # 100 m line along x; one image at x=10 covers x in [0, 25] -> 25%.
    pts, w = coverage.sample_lines([np.array([[0, 0], [100, 0]])], step=1)
    share = coverage.covered_share(pts, w, np.array([[10.0, 0.0]]), near_m=15)
    assert share == pytest.approx(0.25, abs=0.02)
    assert coverage.covered_share(pts, w, np.empty((0, 2))) == 0.0


def test_in_bbox():
    bbox = (8.5, 52.0, 8.6, 52.1)
    assert coverage.in_bbox((8.55, 52.05), bbox)
    assert not coverage.in_bbox((8.45, 52.05), bbox)
