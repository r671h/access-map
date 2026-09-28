import json
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from accessmap.vision import analyze, client
from accessmap.vision.client import BudgetExceeded, CallLedger, GeminiAnalyzer, RateLimiter
from accessmap.vision.prompt import render

TYPES = ["curb_ramp", "raised_curb", "stairs"]

GOOD = {
    "image_usable": True, "sidewalk_visible": True, "sidewalk_surface": "paving_stones",
    "features": [{"type": "raised_curb", "box_2d": [600, 100, 700, 400], "confidence": 0.8,
                  "estimated_distance_m": 6, "estimated_height_cm": 12,
                  "estimated_width_m": None, "permanence": "permanent",
                  "description_en": "Raised curb.", "description_de": "Hoher Bordstein."}],
}


class FakeModels:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        usage = SimpleNamespace(prompt_token_count=1800, candidates_token_count=300,
                                thoughts_token_count=100)
        return SimpleNamespace(text=r if isinstance(r, str) else json.dumps(r),
                               usage_metadata=usage)


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("accessmap.http.time.sleep", lambda s: None)
    monkeypatch.setattr(client.time, "sleep", lambda s: None)


@pytest.fixture
def frame(tmp_path):
    p = tmp_path / "img.jpg"
    Image.new("RGB", (2048, 1536), (100, 100, 100)).save(p)
    return {"frame_id": "f1", "path": "img.jpg", "fov": 80.0,
            "captured_at": pd.Timestamp("2025-06-01", tz="UTC")}


def make(tmp_path, replies, max_calls=100):
    fake = FakeModels(replies)
    a = GeminiAnalyzer(client=SimpleNamespace(models=fake), model="m", prompt_version="v1",
                       active_types=TYPES, cache_dir=tmp_path / "cache",
                       ledger=CallLedger(tmp_path / "ledger.jsonl", max_calls),
                       limiter=RateLimiter(6000), failed_log=tmp_path / "failed.jsonl",
                       thinking_level=None)
    return a, fake


def test_prompt_render_keeps_only_active_definitions():
    p = render("v1", active_types=["curb_ramp", "stairs"], fov_deg=82.4,
               captured_at="2025-06-01")
    assert "- curb_ramp:" in p and "- stairs:" in p and "- raised_curb:" not in p
    assert "about 82 degrees" in p and "2025-06-01" in p
    assert "{" not in p.replace("{camera", "")  # all placeholders filled
    assert not any(line.startswith("#") for line in p.splitlines())


def test_ledger_budget_and_release(tmp_path):
    led = CallLedger(tmp_path / "l.jsonl", max_calls=2)
    led.reserve()
    led.record({"ok": True})
    led.reserve()
    led.release()
    led.record({"ok": False, "calls": 0})
    led.reserve()
    led.record({"ok": True})
    with pytest.raises(BudgetExceeded):
        led.reserve()
    assert CallLedger(tmp_path / "l.jsonl", 2).used == 2  # persisted


def test_analyze_caches(tmp_path, frame):
    a, fake = make(tmp_path, [GOOD])
    rec = a.analyze(frame, tmp_path)
    assert rec["response"]["features"][0]["type"] == "raised_curb"
    assert a.analyze(frame, tmp_path) == rec
    assert fake.calls == 1 and a.ledger.used == 1


def test_invalid_then_valid_retries_once(tmp_path, frame):
    a, fake = make(tmp_path, ["not json", GOOD])
    assert a.analyze(frame, tmp_path) is not None
    assert fake.calls == 2 and a.ledger.used == 2


BAD = GOOD | {"features": [GOOD["features"][0] | {"box_2d": [700, 0, 600, 10]}]}


def test_invalid_twice_is_logged_and_cached_as_failed(tmp_path, frame):
    a, fake = make(tmp_path, [BAD, BAD])
    assert a.analyze(frame, tmp_path) is None
    assert json.loads((tmp_path / "failed.jsonl").read_text())["frame_id"] == "f1"
    assert a.lookup(frame, tmp_path)[0] == "failed"
    assert a.analyze(frame, tmp_path) is None      # no new calls on a rerun
    assert fake.calls == 2 and a.ledger.used == 2


def test_retry_failed_calls_again(tmp_path, frame):
    a, fake = make(tmp_path, [BAD, BAD, GOOD])
    a.analyze(frame, tmp_path)
    assert a.analyze(frame, tmp_path, retry_failed=True) is not None
    assert a.lookup(frame, tmp_path)[0] == "hit" and fake.calls == 3


def test_changed_prompt_makes_cache_stale_and_keeps_old_record(tmp_path, frame, monkeypatch):
    a, fake = make(tmp_path, [GOOD, GOOD])
    a.analyze(frame, tmp_path)
    monkeypatch.setattr(client, "render", lambda *args, **kw: "an edited prompt")
    assert a.lookup(frame, tmp_path)[0] == "stale"
    assert a.analyze(frame, tmp_path) is not None
    assert fake.calls == 2 and a.cache_stats["stale"] == 1
    assert len(list((tmp_path / "cache" / "_stale").rglob("f1.*.json"))) == 1
    assert a.lookup(frame, tmp_path)[0] == "hit"


def test_changed_image_or_types_invalidate(tmp_path, frame):
    a, _ = make(tmp_path, [GOOD])
    a.analyze(frame, tmp_path)
    Image.new("RGB", (2048, 1536), (90, 100, 100)).save(tmp_path / "img.jpg")
    assert a.lookup(frame, tmp_path)[0] == "stale"
    b, _ = make(tmp_path, [])
    b.active_types = ["curb_ramp"]
    assert b.lookup(frame, tmp_path)[0] == "stale"


def test_thinking_fallback_does_not_invalidate_cache(tmp_path, frame):
    a, fake = make(tmp_path, [ValueError("Thinking level MINIMAL is not supported"), GOOD])
    a.thinking_level = a.requested_thinking = "MINIMAL"
    a.analyze(frame, tmp_path)
    b, _ = make(tmp_path, [])
    b.thinking_level = b.requested_thinking = "MINIMAL"
    assert b.lookup(frame, tmp_path)[0] == "hit"


def test_legacy_record_without_fingerprint_is_adopted(tmp_path, frame):
    a, fake = make(tmp_path, [GOOD])
    rec = a.analyze(frame, tmp_path)
    legacy = {k: v for k, v in rec.items() if k not in ("fingerprint", "inputs")}
    a.cache_path("f1").write_text(json.dumps(legacy), encoding="utf-8")
    state, adopted, _ = a.lookup(frame, tmp_path)
    assert state == "hit" and adopted["fingerprint_backfilled"]
    assert "fingerprint" in json.loads(a.cache_path("f1").read_text())
    assert fake.calls == 1


def test_rate_limit_error_is_retried_and_not_billed(tmp_path, frame):
    a, fake = make(tmp_path, [ApiError(429), ApiError(503), GOOD])
    assert a.analyze(frame, tmp_path) is not None
    assert fake.calls == 3 and a.ledger.used == 1


def test_budget_stops_calls(tmp_path, frame):
    a, fake = make(tmp_path, [GOOD], max_calls=0)
    with pytest.raises(BudgetExceeded):
        a.analyze(frame, tmp_path)
    assert fake.calls == 0


def test_image_is_resized(tmp_path, frame):
    import io

    data = client.prepare_image(tmp_path / "img.jpg")
    assert max(Image.open(io.BytesIO(data)).size) == 1536


def test_pilot_ids_deterministic_with_panos():
    frames = pd.DataFrame({"frame_id": [f"r{i}" for i in range(100)] + ["p1_forward",
                                                                        "p1_back", "p2_left"],
                           "is_pano": [False] * 100 + [True] * 3})
    a = analyze.pilot_frame_ids(frames, n=30)
    assert a == analyze.pilot_frame_ids(frames, n=30) and len(a) == 30
    assert sum(i.startswith("p") for i in a) == 3


def test_summarize_costs():
    rec = {"response": GOOD, "usage": [{"prompt_tokens": 2000, "output_tokens": 300,
                                        "thinking_tokens": 100}]}
    s = analyze.summarize({"a": rec, "b": None}, "gemini-3.8-flash", "v1", 1, 3, 1200, 333)
    assert s["valid_share"] == 0.5 and s["features_by_type"] == {"raised_curb": 1}
    assert s["usd_per_call_paid"] == pytest.approx((2000 * 0.75 + 400 * 3.75) / 1e6)


def test_thinking_level_steps_up_when_rejected(tmp_path, frame):
    a, fake = make(tmp_path, [ValueError("Thinking level MINIMAL is not supported"), GOOD])
    a.thinking_level = "MINIMAL"
    assert a.analyze(frame, tmp_path) is not None
    assert a.thinking_level == "LOW" and a.ledger.used == 1
