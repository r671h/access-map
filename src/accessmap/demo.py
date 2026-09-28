"""Offline demo (SPEC §11): a small excerpt of the real results that runs with no keys.

`accessmap build-demo` (maintainer, needs the pipeline outputs) cuts a ~500 m square out of
the pipeline results into tests/fixtures/demo/ — a flat layout, because data/, reports/ and
*.parquet are gitignored:

  project.yaml            project config with the demo bbox
  barriers.geojson        barriers inside the bbox (attached + not attached, `snapped` flag)
  network.geojson         walking network edges inside the bbox (geometry only)
  routing_graph.json      routing graph built from the clipped walking graph
  frames.json             manifest rows of the photos the barriers were seen in
  photos/<frame>.jpg      those photos, downscaled (Mapillary, CC BY-SA 4.0)
  metrics.json, calls.json  evaluation of the real run and its Gemini call count (stats page)

`accessmap demo` unpacks that into a temporary project root and serves it: no keys, and the
server makes no network calls (the browser still loads MapLibre and map tiles online).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import yaml

from accessmap.config import Settings, find_root

log = logging.getLogger(__name__)

DEMO_BBOX = [8.5305, 52.0255, 8.5378, 52.0300]  # ~500 m x 500 m north of Jahnplatz
PHOTO_PX = 640
PHOTO_QUALITY = 72


def fixture_dir(root: Path | None = None) -> Path:
    return (root or find_root()) / "tests" / "fixtures" / "demo"


def _inside(lon: float, lat: float, bbox: list[float]) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8", newline="\n")


def build_fixtures(settings: Settings, bbox: list[float] = DEMO_BBOX,
                   out: Path | None = None) -> dict:
    import networkx as nx
    import osmnx as ox
    import pandas as pd
    from PIL import Image

    from accessmap.routing.graph import build
    from accessmap.web.app import SiteData
    from accessmap.web.export import slim_network

    out = out or fixture_dir(settings.root)
    if out.exists():
        shutil.rmtree(out)
    (out / "photos").mkdir(parents=True)
    data = SiteData(settings)

    barriers = [f for f in data.barriers()
                if _inside(*f["geometry"]["coordinates"], bbox)]
    for f in barriers:
        f["properties"].pop("feedback", None)
    _write(out / "barriers.geojson", {"type": "FeatureCollection", "features": barriers})

    network = slim_network(data.network())
    network["features"] = [f for f in network["features"]
                           if any(_inside(*c, bbox) for c in f["geometry"]["coordinates"])]
    _write(out / "network.geojson", network)

    # Routing graph from the clipped walking graph (largest connected piece) and the
    # barriers/kerbs inside the bbox, with the same builder as `accessmap build-graph`.
    osm = settings.paths.raw / "osm"
    G = ox.load_graphml(osm / "walk.graphml")
    keep = [n for n, d in G.nodes(data=True) if _inside(d["x"], d["y"], bbox)]
    sub = G.subgraph(keep)
    largest = max(nx.weakly_connected_components(sub), key=len)
    sub = G.subgraph(largest).copy()
    kerbs = [f for f in json.loads((osm / "kerbs.geojson").read_text(encoding="utf-8"))
             ["features"] if _inside(*f["geometry"]["coordinates"], bbox)]
    snapped = [f for f in barriers if f["properties"]["snapped"]
               and f["properties"].get("edge_u") in largest
               and f["properties"].get("edge_v") in largest]
    graph = build(sub, snapped, kerbs, settings)
    _write(out / "routing_graph.json", graph)

    frames = pd.read_parquet(settings.paths.processed / "frames.parquet")
    wanted = sorted({v["frame_id"] for f in barriers for v in f["properties"]["views"]})
    rows = frames[frames.frame_id.isin(wanted)].copy()
    for fid, rel in zip(rows.frame_id, rows.path, strict=True):
        with Image.open(settings.root / rel) as im:
            im = im.convert("RGB")
            im.thumbnail((PHOTO_PX, PHOTO_PX))
            im.save(out / "photos" / f"{fid}.jpg", "JPEG", quality=PHOTO_QUALITY,
                    optimize=True, progressive=True)
    rows["path"] = [f"data/images/demo/{fid}.jpg" for fid in rows.frame_id]
    rows["captured_at"] = rows.captured_at.astype(str)
    _write(out / "frames.json", rows.to_dict("records"))
    (out / "photos" / "ATTRIBUTION.md").write_text(
        "Photos: Mapillary contributors, CC BY-SA 4.0 (https://www.mapillary.com).\n"
        "Downscaled excerpts; open one with https://www.mapillary.com/app/?pKey=<image id>\n"
        "(the image id is the file name up to the first underscore).\n",
        encoding="utf-8", newline="\n")

    g = settings.project.gemini
    key = f"{g.model}/{g.prompt}"
    metrics = json.loads((settings.paths.reports / "metrics.json").read_text(encoding="utf-8"))
    _write(out / "metrics.json", {key: metrics[key]} if key in metrics else {})
    _write(out / "calls.json", {"calls": data.stats()["gemini_calls"]["used"],
                                "note": "Gemini calls of the real run the demo was cut from"})

    project = yaml.safe_load((settings.paths.config / "project.yaml").read_text(encoding="utf-8"))
    project["area"] = {"name": "Demo: Bielefeld, north of Jahnplatz (excerpt)",
                       "bbox": bbox, "confirmed": True}
    (out / "project.yaml").write_text(
        "# Demo project (accessmap build-demo): an excerpt of the real run. Serve it with\n"
        "# `accessmap demo`; no keys and no pipeline run needed.\n"
        + yaml.safe_dump(project, sort_keys=False, allow_unicode=True),
        encoding="utf-8", newline="\n")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return {"out": str(out), "barriers": len(barriers), "photos": len(rows),
            "network_edges": len(network["features"]),
            "graph_nodes": len(graph["nodes"]), "graph_edges": len(graph["edges"]),
            "size_mb": round(size / 1e6, 2)}


def unpack(fixtures: Path, root: Path) -> Path:
    """Lay the flat fixture out as a project root that Settings/SiteData understand."""
    import pandas as pd

    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(fixtures / "project.yaml", root / "config" / "project.yaml")
    data = root / "data"
    proc, osm, images = data / "processed", data / "raw" / "osm", data / "images" / "demo"
    cache, reports = data / "cache", root / "reports"
    for p in (proc, osm, images, cache, reports):
        p.mkdir(parents=True, exist_ok=True)

    fc = json.loads((fixtures / "barriers.geojson").read_text(encoding="utf-8"))
    for snapped, name in ((True, "barriers.geojson"), (False, "barriers_unsnapped.geojson")):
        _write(proc / name, {"type": "FeatureCollection", "features": [
            f for f in fc["features"] if f["properties"]["snapped"] is snapped]})
    shutil.copy(fixtures / "routing_graph.json", proc / "routing_graph.json")
    shutil.copy(fixtures / "network.geojson", osm / "walk_edges.geojson")
    rows = json.loads((fixtures / "frames.json").read_text(encoding="utf-8"))
    frames = pd.DataFrame(rows)
    frames["captured_at"] = pd.to_datetime(frames.captured_at, utc=True, format="mixed")
    frames.to_parquet(proc / "frames.parquet")
    for p in (fixtures / "photos").glob("*.jpg"):
        shutil.copy(p, images / p.name)
    shutil.copy(fixtures / "metrics.json", reports / "metrics.json")
    calls = json.loads((fixtures / "calls.json").read_text(encoding="utf-8"))
    (cache / "gemini_calls.jsonl").write_text(json.dumps(calls) + "\n", encoding="utf-8")
    return root


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    from accessmap.web.app import create_app

    with tempfile.TemporaryDirectory(prefix="accessmap-demo-") as tmp:
        root = unpack(fixture_dir(), Path(tmp))
        settings = Settings(root)
        settings.secrets.gemini_api_key = settings.secrets.mapillary_token = None
        print(f"access-map demo ({settings.project.area.name}) on http://{host}:{port}"
              "  (Ctrl+C to stop; feedback is kept only until then)")
        uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")
