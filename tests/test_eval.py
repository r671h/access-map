from shapely import LineString, Point

from accessmap.eval import evaluate, labels
from accessmap.vision.prompt import render

TYPES = ["curb_ramp", "raised_curb", "stairs"]


def rec(usable=True, feats=()):
    return {"response": {"image_usable": usable, "features": [
        {"type": t, "confidence": c, "permanence": p} for t, c, p in feats]}}


def test_selection_is_deterministic_and_disjoint():
    ids = [f"f{i:03d}" for i in range(300)]
    dets = [{"frame_id": f"f{i:03d}", "features": [{"type": "stairs",
                                                    "permanence": "permanent"}]}
            for i in range(0, 300, 7)]
    a = labels.select(ids, dets)
    assert a == labels.select(ids, dets)
    assert len(a["labelled"]) == 60 and len(a["holdout"]) == 18
    assert len(a["tuning_subset"]) == 150
    assert not set(a["holdout"]) & set(a["tuning_subset"])
    assert set(a["tuning_labelled"]) <= set(a["tuning_subset"])
    assert len(a["flagged_by_v1"]) >= 20


def test_frame_metrics_counts_and_filters():
    lab = {"a": {"usable": True, "present": ["stairs"]},
           "b": {"usable": True, "present": []},
           "c": {"usable": False, "present": []},
           "d": {"usable": True, "present": ["curb_ramp"]}}
    records = {
        "a": rec(feats=[("stairs", 0.9, "permanent")]),                  # TP
        "b": rec(feats=[("raised_curb", 0.8, "permanent"),               # FP
                        ("curb_ramp", 0.3, "permanent"),                 # below MIN_CONF
                        ("stairs", 0.9, "temporary")]),                  # temporary
        "c": rec(usable=True, feats=[("curb_ramp", 0.9, "permanent")]),  # FP + usable error
        "d": None,                                                        # no answer: skipped
    }
    m = evaluate.frame_metrics(records, lab, TYPES)
    assert m["frames"] == 3
    assert m["per_type"]["stairs"] == evaluate.prf(1, 0, 0)
    assert m["per_type"]["raised_curb"]["fp"] == 1
    assert m["per_type"]["curb_ramp"]["fp"] == 1
    assert m["micro"]["tp"] == 1 and m["micro"]["fp"] == 2
    assert m["usable_confusion"] == {"tp": 2, "fp": 1, "fn": 0, "tn": 0}


def test_unusable_prediction_predicts_nothing():
    assert evaluate.predicted_types(rec(False, [("stairs", 0.9, "permanent")])["response"]
                                    ) == set()


def test_prf_edge_cases():
    assert evaluate.prf(0, 0, 0)["precision"] is None
    assert evaluate.prf(0, 3, 0)["precision"] == 0.0
    assert evaluate.prf(2, 2, 2)["f1"] == 0.5


def test_in_view_uses_distance_and_fov():
    cam = {"image_usable": True, "lon": 8.53, "lat": 52.02, "heading": 0.0, "fov": 90.0}
    ahead = Point(8.53, 52.02 + 10 / 111_257)
    behind = Point(8.53, 52.02 - 10 / 111_257)
    far = Point(8.53, 52.02 + 40 / 111_257)
    assert evaluate.in_view(ahead, [cam])
    assert not evaluate.in_view(behind, [cam])
    assert not evaluate.in_view(far, [cam])
    assert evaluate.in_view(LineString([(8.53, 52.02 - 0.001), (8.53, 52.02 + 0.0001)]), [cam])


def test_holdout_split_is_stable():
    assert evaluate.is_holdout(123) == evaluate.is_holdout(123)
    share = sum(evaluate.is_holdout(i) for i in range(2000)) / 2000
    assert 0.25 < share < 0.35


def test_prompt_versions_have_their_own_definitions():
    kw = dict(active_types=TYPES, fov_deg=90, captured_at="2025-01-01")
    v1, v2, v3 = (render(v, **kw) for v in ("v1", "v2", "v3"))
    assert "ALONG the street" not in v1 and "ALONG the street" in v2
    assert "striped paving" in v2 and "striped paving" not in v3
    assert "about 15 m" in v3 and "about 15 m" not in v2
