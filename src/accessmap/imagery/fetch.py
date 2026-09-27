"""`accessmap fetch-images` (phase 2): metadata, selection, downloads, panorama crops, manifest."""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from accessmap.config import Settings
from accessmap.coverage import image_age_years, in_bbox, utm_epsg
from accessmap.http import HttpError, get_json, request
from accessmap.imagery import mapillary
from accessmap.imagery.pano import crop_offsets, perspective_crops
from accessmap.imagery.select import (
    MAJOR_ROAD_M,
    MAX_PANO_HEADING_DIFF,
    MAX_POSITION_SHIFT_M,
    NEAR_EDGE_M,
    SKIP_CAMERA_TYPES,
    Candidate,
    distance_to_network,
    hfov_deg,
    major_road_sequences,
    spread,
    thin,
)

log = logging.getLogger(__name__)

PANO_FOV = 90.0
PANO_CROP_PX = 768
MANIFEST_COLUMNS = ["frame_id", "image_id", "lon", "lat", "heading", "fov", "captured_at",
                    "is_pano", "crop", "sequence", "source", "path"]


def load_metadata(settings: Settings, refresh: bool = False) -> list[dict]:
    """Full image metadata for the bbox, cached (thumbnail URLs inside expire; see download)."""
    out = settings.paths.raw / "mapillary" / "images.json"
    bbox = list(settings.project.area.bbox)
    if out.is_file() and not refresh:
        cached = json.loads(out.read_text(encoding="utf-8"))
        if cached.get("bbox") == bbox:
            return cached["images"]
    token = settings.secrets.mapillary_token
    if not token:
        raise RuntimeError("MAPILLARY_TOKEN is not set (SETUP.md §3)")
    images = mapillary.search_bbox(token, tuple(bbox))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bbox": bbox, "fetched_at": datetime.now(UTC).isoformat(),
                               "images": images}), encoding="utf-8")
    return images


def heading_of(img: dict) -> float | None:
    h = img.get("computed_compass_angle")
    if h is None:
        h = img.get("compass_angle")
    return None if h is None else float(h) % 360


def build_candidates(settings: Settings, images: list[dict]) -> tuple[list[Candidate], dict]:
    """Filter by bbox, camera type, heading, age and distance to the walk network."""
    import geopandas as gpd
    from pyproj import Transformer

    p = settings.project
    bbox = p.area.bbox
    stats = {"metadata": len(images)}
    imgs = [i for i in images if in_bbox(mapillary.position(i), bbox)]
    stats["in_bbox"] = len(imgs)
    imgs = [i for i in imgs if i.get("camera_type") not in SKIP_CAMERA_TYPES]
    stats["not_fisheye"] = len(imgs)
    imgs = [i for i in imgs if i.get("is_pano") or heading_of(i) is not None]
    stats["with_heading"] = len(imgs)
    imgs = [i for i in imgs if image_age_years(i["captured_at"]) <= p.imagery.max_age_years]
    stats["fresh"] = len(imgs)
    if not imgs:
        return [], stats

    epsg = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    to_m = Transformer.from_crs(4326, epsg, always_xy=True)

    def project(lonlat: np.ndarray) -> np.ndarray:
        return np.column_stack(to_m.transform(lonlat[:, 0], lonlat[:, 1]))

    # Positions whose raw GPS and SfM correction disagree a lot are unreliable either way.
    xy = project(np.array([mapillary.position(i) for i in imgs]))
    raw_xy = project(np.array([i["geometry"]["coordinates"][:2] for i in imgs]))
    shift = np.hypot(*(xy - raw_xy).T)
    keep = shift <= MAX_POSITION_SHIFT_M
    stats["position_consistent"] = int(keep.sum())

    # Frames shot from motorways/trunk roads show no pedestrian space.
    major_path = settings.paths.raw / "osm" / "major_roads.geojson"
    if major_path.is_file():
        major = gpd.read_file(major_path).to_crs(epsg)
    else:
        major = gpd.GeoDataFrame(geometry=[], crs=epsg)
    if len(major):
        d_major = np.minimum(distance_to_network(xy, major.geometry.values),
                             distance_to_network(raw_xy, major.geometry.values))
        bad_seq = major_road_sequences([i.get("sequence") for i in imgs], d_major)
        on_major = (d_major <= MAJOR_ROAD_M) | np.array(
            [i.get("sequence") in bad_seq for i in imgs])
        keep &= ~on_major
        stats["major_road_sequences"] = len(bad_seq)
    stats["not_on_major_road"] = int(keep.sum())

    # Panoramas: the heading must agree with the direction of travel, or every crop is off.
    by_seq: dict[str, list[dict]] = {}
    for i in images:
        by_seq.setdefault(i.get("sequence"), []).append(i)
    for k, i in enumerate(imgs):
        if keep[k] and i.get("is_pano"):
            tb = travel_bearing(i, by_seq)
            if tb is None or angle_diff(tb, heading_of(i) or 0.0) > MAX_PANO_HEADING_DIFF:
                keep[k] = False
    stats["pano_heading_ok"] = int(keep.sum())

    edges = gpd.read_file(settings.paths.raw / "osm" / "walk_edges.geojson").to_crs(epsg)
    dist = distance_to_network(xy, edges.geometry.values)
    keep &= dist <= NEAR_EDGE_M
    stats["near_network"] = int(keep.sum())

    cands = [
        Candidate(image_id=str(i["id"]), x=float(x), y=float(y), heading=heading_of(i),
                  captured_at=int(i["captured_at"]), is_pano=bool(i.get("is_pano")),
                  on_foot=bool(i.get("on_foot")), quality=float(i.get("quality_score") or 0))
        for i, (x, y), k in zip(imgs, xy, keep, strict=True) if k
    ]
    return cands, stats


def refresh_thumb_url(token: str, image_id: str) -> str:
    data = get_json(f"{mapillary.API}/{image_id}", params={"fields": "thumb_2048_url"},
                    headers=mapillary._auth(token), timeout=30)
    return data["thumb_2048_url"]


def download_one(img: dict, dest: Path, token: str | None) -> bool:
    """Download the 2048 px thumbnail; re-fetch the URL once if it has expired."""
    if dest.is_file() and dest.stat().st_size > 0:
        return False
    url = img.get("thumb_2048_url")
    try:
        if not url:
            raise HttpError("no url", 403)
        r = request("GET", url, timeout=60, retries=3)
    except HttpError as e:
        if e.status not in (403, 404, 410) or not token:
            raise
        r = request("GET", refresh_thumb_url(token, str(img["id"])), timeout=60, retries=3)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return True


def download_all(settings: Settings, imgs: list[dict]) -> tuple[int, list[str]]:
    from tqdm import tqdm

    token = settings.secrets.mapillary_token
    root = settings.paths.images / "mapillary"
    new, failed = 0, []

    def job(img):
        return download_one(img, root / f"{img['id']}.jpg", token)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(job, i): str(i["id"]) for i in imgs}
        for f in tqdm(futures, desc="download", unit="img"):
            try:
                new += bool(f.result())
            except Exception as e:  # noqa: BLE001 - logged, frame skipped
                log.warning("download %s failed: %s", futures[f], e)
                failed.append(futures[f])
    return new, failed


def make_frames(settings: Settings, imgs: list[dict]) -> pd.DataFrame:
    """One row per analysable frame; panoramas become `pano_crops` perspective crops."""
    from PIL import Image

    rows = []
    defaulted_fov = 0
    n_crops = settings.project.imagery.pano_crops
    root = settings.paths.images
    for img in imgs:
        iid = str(img["id"])
        src = root / "mapillary" / f"{iid}.jpg"
        if not src.is_file():
            continue
        lon, lat = mapillary.position(img)
        base = {"image_id": iid, "lon": lon, "lat": lat,
                "captured_at": pd.Timestamp(img["captured_at"], unit="ms", tz="UTC"),
                "is_pano": bool(img.get("is_pano")), "sequence": img.get("sequence"),
                "source": "mapillary"}
        heading = heading_of(img) or 0.0
        if img.get("is_pano"):
            crop_dir = root / "crops"
            offsets = crop_offsets(n_crops)
            paths = [crop_dir / f"{iid}_{name}.jpg" for name, _ in offsets]
            if all(p.is_file() for p in paths):  # already cropped: idempotent rerun
                crops = [(name, (heading + off) % 360, None) for name, off in offsets]
            else:
                with Image.open(src) as eq:
                    crops = perspective_crops(eq, heading, n=n_crops, fov=PANO_FOV,
                                              out_px=PANO_CROP_PX)
                crop_dir.mkdir(parents=True, exist_ok=True)
                for (_, _, im), path in zip(crops, paths, strict=True):
                    im.save(path, "JPEG", quality=90)
            for name, bearing, _ in crops:
                rows.append(base | {"frame_id": f"{iid}_{name}", "heading": bearing,
                                    "fov": PANO_FOV, "crop": name,
                                    "path": f"data/images/crops/{iid}_{name}.jpg"})
        else:
            fov, defaulted = hfov_deg(img.get("camera_parameters"), img.get("width"))
            defaulted_fov += defaulted
            rows.append(base | {"frame_id": iid, "heading": heading, "fov": fov, "crop": None,
                                "path": f"data/images/mapillary/{iid}.jpg"})
    if defaulted_fov:
        log.info("FOV defaulted to 70° for %d images without camera_parameters", defaulted_fov)
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def travel_bearing(img: dict, by_seq: dict[str, list[dict]]) -> float | None:
    """Bearing of travel at `img`, from its nearest neighbours in the same sequence."""
    from pyproj import Geod

    seq = sorted(by_seq.get(img.get("sequence"), []), key=lambda i: i["captured_at"])
    ids = [i["id"] for i in seq]
    if img["id"] not in ids or len(seq) < 2:
        return None
    k = ids.index(img["id"])
    a, b = seq[max(0, k - 1)], seq[min(len(seq) - 1, k + 1)]
    (lon1, lat1), (lon2, lat2) = mapillary.position(a), mapillary.position(b)
    if (lon1, lat1) == (lon2, lat2):
        return None
    az, _, _ = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
    return az % 360


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def qa_sheets(settings: Settings, frames: pd.DataFrame, meta: dict[str, dict],
              all_images: list[dict]) -> dict:
    from accessmap.eval.contact import Tile, contact_sheet

    qa = settings.paths.qa
    root = settings.paths.root
    rng = random.Random(42)
    sample = frames.sample(min(12, len(frames)), random_state=42)
    contact_sheet(
        [Tile(root / r.path, f"{r.frame_id} · {r.captured_at:%Y-%m} · {r.heading:.0f}° "
                             f"fov {r.fov:.0f}") for r in sample.itertuples()],
        qa / "frames_sample.jpg", title="Random sample of selected frames")

    # Panorama "forward" check: forward crop vs direction of travel in the sequence.
    by_seq: dict[str, list[dict]] = {}
    for i in all_images:
        by_seq.setdefault(i.get("sequence"), []).append(i)
    panos = sorted(frames.loc[frames.is_pano, "image_id"].unique())
    diffs = {}
    for iid in panos:
        tb = travel_bearing(meta[iid], by_seq)
        if tb is not None:
            diffs[iid] = (tb, heading_of(meta[iid]) or 0.0)
    result = {"panos": len(panos), "with_travel_bearing": len(diffs)}
    if diffs:
        d = np.array([angle_diff(t, h) for t, h in diffs.values()])
        result |= {"median_heading_vs_travel_deg": round(float(np.median(d)), 1),
                   "share_within_30deg": round(float((d <= 30).mean()), 2),
                   "share_reversed_150deg": round(float((d >= 150).mean()), 2)}
        tiles = []
        for iid in rng.sample(sorted(diffs), min(3, len(diffs))):
            tb, h = diffs[iid]
            for r in frames[frames.image_id == iid].itertuples():
                tiles.append(Tile(root / r.path,
                                  f"{iid[-6:]} {r.crop} {r.heading:.0f}° (travel {tb:.0f}°)"))
        contact_sheet(tiles, qa / "pano_check.jpg",
                      title="Panorama crops: 'forward' should look along the travel direction")
    return result


def selection_map(settings: Settings, cands, chosen, path: Path) -> None:
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bbox = settings.project.area.bbox
    epsg = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    edges = gpd.read_file(settings.paths.raw / "osm" / "walk_edges.geojson").to_crs(epsg)
    fig, ax = plt.subplots(figsize=(8, 12), dpi=110)
    edges.plot(ax=ax, color="#666", linewidth=0.5)
    ax.scatter([c.x for c in cands], [c.y for c in cands], s=2, c="#bbb",
               label=f"candidates ({len(cands)})")
    reg = [c for c in chosen if not c.is_pano]
    pan = [c for c in chosen if c.is_pano]
    ax.scatter([c.x for c in reg], [c.y for c in reg], s=10, c="#1f77b4",
               label=f"selected images ({len(reg)})")
    ax.scatter([c.x for c in pan], [c.y for c in pan], s=30, marker="*", c="#d62728",
               label=f"selected panoramas ({len(pan)})")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", markerscale=2)
    ax.set_title("Frame selection over the walking network")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(settings: Settings, refresh: bool = False, reselect: bool = False) -> dict:
    p = settings.project
    settings.paths.ensure()
    images = load_metadata(settings, refresh=refresh)
    meta = {str(i["id"]): i for i in images}

    sel_path = settings.paths.processed / "selection.json"
    cands, stats = build_candidates(settings, images)
    if sel_path.is_file() and not (refresh or reselect):
        chosen_ids = json.loads(sel_path.read_text(encoding="utf-8"))["image_ids"]
        by_id = {c.image_id: c for c in cands}
        chosen = [by_id[i] for i in chosen_ids if i in by_id]
        log.info("Reusing selection of %d images (%s)", len(chosen), sel_path)
    else:
        thinned = thin(cands)
        stats["after_thinning"] = len(thinned)
        chosen = spread(thinned, p.budget.max_images)
        sel_path.write_text(json.dumps({"created_at": datetime.now(UTC).isoformat(),
                                        "image_ids": [c.image_id for c in chosen]}),
                            encoding="utf-8")
    stats["selected_images"] = len(chosen)
    stats["selected_panoramas"] = sum(c.is_pano for c in chosen)

    chosen_meta = [meta[c.image_id] for c in chosen]
    new, failed = download_all(settings, chosen_meta)
    stats |= {"downloaded_new": new, "download_failed": len(failed)}

    frames = make_frames(settings, chosen_meta)
    frames.to_parquet(settings.paths.processed / "frames.parquet", index=False)
    stats["frames"] = len(frames)
    stats["pano_check"] = qa_sheets(settings, frames, meta, images)
    selection_map(settings, cands, chosen, settings.paths.qa / "frames_map.png")

    (settings.paths.reports / "imagery_summary.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return stats
