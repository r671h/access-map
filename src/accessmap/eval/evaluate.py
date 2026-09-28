"""`accessmap evaluate` (phase 5, SPEC §7).

Two views per prompt version:
- frame level against the hand labels (labels/frame_labels.json): per type, is the type
  predicted (permanent, confidence ≥ MIN_CONF, usable frame) where the label says present;
- object level against OSM (kerb=lowered|flush → curb_ramp, kerb=raised → raised_curb,
  highway=steps → stairs), barriers within 10 m, GT restricted to objects in view of the
  evaluated frames. OSM is incomplete, so OSM precision is a lower bound.

Hold-out: 30% of labelled frames and 30% of OSM objects (by id hash) are only scored with
--final, so tuning never looks at them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pyproj import Geod
from shapely import LineString, Point

from accessmap.config import Settings
from accessmap.eval.labels import load_labels, load_selection
from accessmap.geo.snap import to_metric

log = logging.getLogger(__name__)

GEOD = Geod(ellps="WGS84")
MIN_CONF = 0.5
MATCH_M = 10.0
VIEW_M = 25.0
OSM_TYPES = ("curb_ramp", "raised_curb", "stairs")
HOLDOUT_SHARE = 0.3


def load_records(settings: Settings, model: str, prompt: str, frame_ids: list[str]
                 ) -> dict[str, dict | None]:
    """Cached Gemini records; None for frames with no (valid) answer yet."""
    from accessmap.vision.client import model_dir

    base = settings.paths.cache / "gemini" / model_dir(model) / prompt
    out = {}
    for fid in frame_ids:
        p = base / f"{fid}.json"
        rec = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
        out[fid] = rec if rec and rec.get("response") is not None else None
    return out


def predicted_types(response: dict, min_conf: float = MIN_CONF) -> set[str]:
    if not response["image_usable"]:
        return set()
    return {f["type"] for f in response["features"]
            if f["permanence"] == "permanent" and f["confidence"] >= min_conf}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    rnd = lambda x: round(x, 3) if x is not None else None  # noqa: E731
    return {"tp": tp, "fp": fp, "fn": fn, "precision": rnd(p), "recall": rnd(r),
            "f1": rnd(f1)}


def frame_metrics(records: dict[str, dict | None], labels: dict[str, dict],
                  types: list[str], min_conf: float = MIN_CONF) -> dict:
    """Per-type and micro-averaged P/R/F1 over labelled frames with a valid record."""
    counts = {t: [0, 0, 0] for t in types}
    usable = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    errors = []
    n = 0
    for fid, lab in sorted(labels.items()):
        rec = records.get(fid)
        if rec is None:
            continue
        n += 1
        resp = rec["response"]
        pred, truth = predicted_types(resp, min_conf), set(lab["present"])
        key = ("t" if resp["image_usable"] == lab["usable"] else "f") + (
            "p" if resp["image_usable"] else "n")
        usable[key] += 1
        for t in types:
            if t in pred and t in truth:
                counts[t][0] += 1
            elif t in pred:
                counts[t][1] += 1
                errors.append({"frame_id": fid, "type": t, "kind": "fp"})
            elif t in truth:
                counts[t][2] += 1
                errors.append({"frame_id": fid, "type": t, "kind": "fn"})
    per_type = {t: prf(*c) for t, c in counts.items() if any(c)}
    micro = prf(*(sum(c[i] for c in counts.values()) for i in range(3)))
    return {"frames": n, "per_type": per_type, "micro": micro,
            "usable_confusion": usable, "errors": errors}


def is_holdout(osm_id) -> bool:
    return int(hashlib.sha1(str(osm_id).encode()).hexdigest(), 16) % 10 < HOLDOUT_SHARE * 10


def osm_ground_truth(settings: Settings) -> list[dict]:
    osm = settings.paths.raw / "osm"
    gt = []
    kerbs = json.loads((osm / "kerbs.geojson").read_text(encoding="utf-8"))["features"]
    for f in kerbs:
        kerb = f["properties"].get("kerb")
        t = {"lowered": "curb_ramp", "flush": "curb_ramp", "raised": "raised_curb"}.get(kerb)
        if t:
            gt.append({"type": t, "osm_id": f["properties"]["osm_id"],
                       "geom": Point(f["geometry"]["coordinates"])})
    steps = json.loads((osm / "steps.geojson").read_text(encoding="utf-8"))["features"]
    for f in steps:
        if f["geometry"]["type"] == "LineString":
            gt.append({"type": "stairs", "osm_id": f["properties"]["osm_id"],
                       "geom": LineString(f["geometry"]["coordinates"])})
    for g in gt:
        g["holdout"] = is_holdout(g["osm_id"])
    return gt


def in_view(geom, frames: list[dict]) -> bool:
    """Some point of the object within VIEW_M of a usable camera and inside its FOV."""
    pts = [geom] if geom.geom_type == "Point" else [
        geom.interpolate(i / 4, normalized=True) for i in range(5)]
    for fr in frames:
        if not fr["image_usable"]:
            continue
        for p in pts:
            az, _, d = GEOD.inv(fr["lon"], fr["lat"], p.x, p.y)
            if d <= VIEW_M and abs((az - fr["heading"] + 540) % 360 - 180) <= fr["fov"] / 2:
                return True
    return False


def _dist_m(geom, lon: float, lat: float) -> float:
    from shapely.ops import transform

    g = transform(lambda x, y, z=None: to_metric(x, y), geom)
    return g.distance(Point(*to_metric(lon, lat)))


def osm_metrics(barriers: list[dict], gt: list[dict], frames: list[dict],
                holdout: bool | None) -> dict:
    """Greedy one-to-one matching within MATCH_M per type. Predictions whose nearest GT
    candidate belongs to the other split are left out of that split's precision.
    holdout=None scores all GT objects (no split)."""
    if holdout is None:
        gt = [g | {"holdout": False} for g in gt]
        holdout = False
    visible = [g for g in gt if in_view(g["geom"], frames)]
    out = {}
    for t in OSM_TYPES:
        g_t = [g for g in visible if g["type"] == t]
        preds = [b for b in barriers if b["properties"]["type"] == t]
        pairs = sorted((_dist_m(g["geom"], *b["geometry"]["coordinates"]), i, j)
                       for i, g in enumerate(g_t) for j, b in enumerate(preds))
        used_g, used_p, tp = set(), set(), 0
        other_split = set()
        for d, i, j in pairs:
            if d > MATCH_M:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            if g_t[i]["holdout"] == holdout:
                tp += 1
            else:
                other_split.add(j)
        n_gt = sum(g["holdout"] == holdout for g in g_t)
        n_pred = len(preds) - len(other_split)
        out[t] = prf(tp, n_pred - tp, n_gt - tp) | {"gt_in_view": n_gt}
    return out


def evaluate(settings: Settings, model: str, prompt: str, final: bool = False) -> dict:
    from accessmap.geo.geolocate import build_barriers, load_index
    from accessmap.vision.analyze import detection_rows, load_frames

    sel = load_selection(settings.root)
    labels = load_labels(settings.root)
    types = settings.project.detection.types
    frames_df = load_frames(settings).set_index("frame_id", drop=False)
    index = load_index(settings)
    g = settings.project.geo
    gt = osm_ground_truth(settings)

    splits = {"tuning": sel["tuning_subset"]}
    if final:
        splits["holdout"] = sel["holdout"]
        splits["all_frames"] = sorted(frames_df.frame_id)
    result = {"model": model, "prompt": prompt, "min_conf": MIN_CONF}
    for split, ids in splits.items():
        records = load_records(settings, model, prompt, ids)
        missing = [f for f, r in records.items() if r is None]
        rows = [detection_rows(r, frames_df.loc[f].to_dict()) for f, r in records.items()
                if r is not None]
        lab_ids = sel["tuning_labelled"] if split == "tuning" else (
            sel["holdout"] if split == "holdout" else sel["labelled"])
        fm = frame_metrics(records, {f: labels[f] for f in lab_ids}, types)
        _, snapped, unsnapped = build_barriers(rows, settings.root, index, g.snap_curb_m,
                                               g.snap_other_m)
        result[split] = {
            "frames": len(ids), "frames_without_answer": len(missing),
            "labels": fm,
            "osm": osm_metrics(snapped + unsnapped, gt, rows,
                               holdout={"tuning": False, "holdout": True}.get(split)),
            "barriers": len(snapped) + len(unsnapped),
            "unsnapped_share": round(len(unsnapped) / max(1, len(snapped) + len(unsnapped)),
                                     3),
            "detections_by_type": _count_types(rows),
        }
    return result


def _count_types(rows: list[dict]) -> dict:
    from collections import Counter

    return dict(Counter(f["type"] for r in rows if r["image_usable"] for f in r["features"]
                        if f["permanence"] == "permanent").most_common())


def save(settings: Settings, result: dict) -> Path:
    path = settings.paths.reports / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data[f"{result['model']}/{result['prompt']}"] = result
    path.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
    return path


def markdown_table(result: dict, split: str = "tuning") -> str:
    r = result[split]
    lines = [f"**{result['model']} / {result['prompt']} — {split}** "
             f"({r['labels']['frames']} labelled frames, {r['frames']} frames, "
             f"{r['barriers']} barriers, unsnapped {r['unsnapped_share']:.0%})", "",
             "| type | label TP | label FP | label FN | precision | recall | "
             "OSM TP/pred/GT |", "|---|---|---|---|---|---|---|"]
    fmt = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731
    types = sorted(set(r["labels"]["per_type"]) | set(OSM_TYPES))
    for t in types:
        m = r["labels"]["per_type"].get(t, {"tp": 0, "fp": 0, "fn": 0, "precision": None,
                                            "recall": None})
        o = r["osm"].get(t)
        osm = f"{o['tp']}/{o['tp'] + o['fp']}/{o['gt_in_view']}" if o else ""
        lines.append(f"| {t} | {m['tp']} | {m['fp']} | {m['fn']} | {fmt(m['precision'])} | "
                     f"{fmt(m['recall'])} | {osm} |")
    mi = r["labels"]["micro"]
    lines.append(f"| **all (micro)** | {mi['tp']} | {mi['fp']} | {mi['fn']} | "
                 f"{fmt(mi['precision'])} | {fmt(mi['recall'])} | |")
    u = r["labels"]["usable_confusion"]
    lines += ["", f"image_usable vs label: correct {u['tp'] + u['tn']}, "
                  f"usable-but-labelled-unusable {u['fp']}, missed-usable {u['fn']}"]
    return "\n".join(lines)


def error_sheets(settings: Settings, result: dict, split: str = "tuning",
                 per_sheet: int = 8) -> list[Path]:
    """Frames where the model and the labels disagree: the model's boxes for the
    disagreeing types, the label in the caption. Doubles as the user's spot-check sheet."""
    from collections import defaultdict

    from accessmap.eval.contact import Box, Tile, contact_sheet
    from accessmap.vision.analyze import load_frames

    labels = load_labels(settings.root)
    frames = load_frames(settings).set_index("frame_id")
    by_frame = defaultdict(list)
    for e in result[split]["labels"]["errors"]:
        by_frame[e["frame_id"]].append(e)
    records = load_records(settings, result["model"], result["prompt"], list(by_frame))
    tiles = []
    for fid, errs in sorted(by_frame.items()):
        resp = records[fid]["response"]
        fp_types = {e["type"] for e in errs if e["kind"] == "fp"}
        boxes = [Box(f["box_2d"], f"{f['type']} {f['confidence']:.2f}", f["type"])
                 for f in resp["features"] if f["type"] in fp_types]
        label = ", ".join(labels[fid]["present"]) or "none"
        what = " ".join(f"{e['kind'].upper()}:{e['type']}" for e in errs)
        tiles.append(Tile(settings.root / frames.loc[fid, "path"],
                          f"{fid[-12:]} | label: {label} | {what}", boxes))
    out = []
    for i in range(0, len(tiles), per_sheet):
        p = settings.paths.qa / f"errors_{result['prompt']}_{split}_{i // per_sheet + 1}.jpg"
        contact_sheet(tiles[i:i + per_sheet], p, cols=4,
                      title=f"{result['prompt']} {split}: model vs labels "
                            f"(boxes = false-positive types)")
        out.append(p)
    return out
