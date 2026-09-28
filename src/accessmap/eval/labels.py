"""Frame labels for prompt tuning (phase 5, DECISIONS.md 2026-09-28).

OSM has too little ground truth in view (15 objects), so ~60 frames are labelled by hand
at frame level: for each barrier type, is at least one instance present that the prompt's
definition covers (on or directly next to pedestrian space, within ~25 m, permanent)?

Split: 30% of the labelled frames are a hold-out that is only scored for the final
prompt. Prompt versions run on a fixed tuning subset (~150 frames) that contains the
labelled tuning frames and none of the hold-out frames.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

LABELS_DIR = Path("labels")
SELECTION_FILE = "selection.json"
LABELS_FILE = "frame_labels.json"
SEED = 11
RARE_TYPES = ("curb_ramp", "stairs", "step", "no_sidewalk", "narrow_passage")


def select(frame_ids: list[str], detections: list[dict], n_labelled: int = 60,
           n_flagged: int = 20, holdout_share: float = 0.3, subset_size: int = 150,
           seed: int = SEED) -> dict:
    """Deterministic selection. `detections` are v1 rows from detections.jsonl; they are only
    used to make sure rare types have positives (n_flagged frames with a rare-type v1
    detection), the rest is uniform random."""
    rng = random.Random(seed)
    ids = sorted(frame_ids)
    flagged = sorted({d["frame_id"] for d in detections
                      if any(f["type"] in RARE_TYPES and f["permanence"] == "permanent"
                             for f in d["features"])})
    chosen = rng.sample(flagged, min(n_flagged, len(flagged)))
    rest = [i for i in ids if i not in set(chosen)]
    chosen += rng.sample(rest, n_labelled - len(chosen))
    chosen.sort()
    holdout = sorted(rng.sample(chosen, round(n_labelled * holdout_share)))
    tuning = sorted(set(chosen) - set(holdout))
    others = [i for i in ids if i not in set(chosen)]
    subset = sorted(tuning + rng.sample(others, subset_size - len(tuning)))
    return {"seed": seed, "labelled": chosen, "tuning_labelled": tuning, "holdout": holdout,
            "tuning_subset": subset, "flagged_by_v1": sorted(set(chosen) & set(flagged))}


def load_selection(root: Path) -> dict:
    return json.loads((root / LABELS_DIR / SELECTION_FILE).read_text(encoding="utf-8"))


def load_labels(root: Path) -> dict[str, dict]:
    """frame_id -> {"usable": bool, "present": [types], "notes": str}."""
    data = json.loads((root / LABELS_DIR / LABELS_FILE).read_text(encoding="utf-8"))
    return data["frames"]


def labelling_images(root: Path, frames_by_id: dict[str, dict], ids: list[str], out_dir: Path,
                     per_image: int = 2, width: int = 1280) -> list[Path]:
    """Frames stacked vertically at a size where kerbs are readable, captioned with the
    frame id only (no model output, so the labels stay independent of the model)."""
    from PIL import Image, ImageDraw

    from accessmap.eval.contact import _font

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(ids), per_image):
        tiles = []
        for fid in ids[i:i + per_image]:
            im = Image.open(root / frames_by_id[fid]["path"]).convert("RGB")
            im.thumbnail((width, width))
            tile = Image.new("RGB", (width, im.height + 30), (0, 0, 0))
            tile.paste(im, (0, 30))
            ImageDraw.Draw(tile).text((8, 4), fid, fill=(255, 255, 0), font=_font(20))
            tiles.append(tile)
        sheet = Image.new("RGB", (width, sum(t.height for t in tiles)), (0, 0, 0))
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y))
            y += t.height
        p = out_dir / f"label_{i // per_image + 1:02d}.jpg"
        sheet.save(p, "JPEG", quality=88)
        paths.append(p)
    return paths
