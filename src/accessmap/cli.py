"""Command line: `accessmap <command>` or `python -m accessmap.cli <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_check(args) -> int:
    from accessmap.checks import format_table, run_checks
    from accessmap.config import get_settings

    settings = get_settings()
    settings.paths.ensure()
    results = run_checks(settings)
    print(format_table(results))
    gem = next(r for r in results if r.service == "Gemini")
    if gem.extra.get("calls"):
        from accessmap.vision.analyze import ledger_for

        ledger_for(settings).record({"model": "check", "ok": True, "calls": gem.extra["calls"]})
    if gem.extra.get("models"):
        out = settings.paths.reports / "gemini_models.json"
        out.write_text(json.dumps(gem.extra, indent=2), encoding="utf-8")
        print(f"\nGemini models available to this key: {out}")
    return 0 if all(r.ok for r in results) else 1


def cmd_coverage(args) -> int:
    from accessmap.config import get_settings
    from accessmap.coverage import build_report, to_markdown

    report = build_report(get_settings(), refresh=args.refresh)
    print(to_markdown(report))
    return 0


def cmd_fetch_osm(args) -> int:
    from accessmap.config import get_settings
    from accessmap.osm.fetch import run

    s = run(get_settings(), refresh=args.refresh)
    g = s["graph"]
    print(f"walk graph: {g['edges_undirected']} edges, {g['length_km']} km; "
          f"{s['components']['count']} components; kerb nodes {s['layers']['kerb_nodes']}, "
          f"steps {s['layers']['steps_ways']} -> reports/osm_summary.md")
    return 0


def cmd_fetch_images(args) -> int:
    from accessmap.config import get_settings
    from accessmap.imagery.fetch import run

    stats = run(get_settings(), refresh=args.refresh, reselect=args.reselect)
    print(json.dumps(stats, indent=2, default=str))
    return 0


def cmd_analyze(args) -> int:
    from accessmap.config import get_settings
    from accessmap.vision.analyze import load_frames, run

    settings = get_settings()
    model = args.model or settings.project.gemini.model
    args.prompt = args.prompt or settings.project.gemini.prompt
    if not model:
        print("No model chosen yet: pass --model or set gemini.model in config/project.yaml")
        return 2
    out = None
    if args.subset:
        from accessmap.eval.labels import load_selection

        ids = load_selection(settings.root)[f"{args.subset}_subset"]
        out = settings.paths.processed / "runs" / f"{model}_{args.prompt}_{args.subset}.jsonl"
    else:
        ids = None if args.all else sorted(load_frames(settings).frame_id)[: args.limit]
    stats = run(settings, model, ids, prompt_version=args.prompt, out=out,
                retry_failed=args.retry_failed)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_pilot(args) -> int:
    from accessmap.config import get_settings
    from accessmap.vision.analyze import compare_sheets, load_frames, pilot_frame_ids, run

    settings = get_settings()
    models = args.models or settings.project.gemini.pilot_candidates
    ids = pilot_frame_ids(load_frames(settings), n=args.n)
    all_stats = {}
    for m in models:
        out = settings.paths.processed / "pilot" / f"{m}_{args.prompt}.jsonl"
        all_stats[m] = run(settings, m, ids, prompt_version=args.prompt, out=out)
    sheets = compare_sheets(settings, models, ids, prompt_version=args.prompt)
    report = {"frame_ids": ids, "stats": all_stats, "sheets": [str(p) for p in sheets]}
    (settings.paths.reports / f"pilot_{args.prompt}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(all_stats, indent=2))
    print("sheets:")
    for path in sheets:
        print(f"  {path}")
    return 0


def cmd_geolocate(args) -> int:
    from accessmap.config import get_settings
    from accessmap.geo.geolocate import run

    print(json.dumps(run(get_settings()), indent=2))
    return 0


def cmd_evaluate(args) -> int:
    from accessmap.config import get_settings
    from accessmap.eval.evaluate import evaluate, markdown_table, save

    settings = get_settings()
    model = args.model or settings.project.gemini.model
    result = evaluate(settings, model, args.prompt or settings.project.gemini.prompt,
                      final=args.final)
    print(f"saved {save(settings, result)}")
    for split in ("tuning", "holdout", "all_frames"):
        if split in result:
            print(markdown_table(result, split) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="accessmap", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify keys and API access (Gemini, Mapillary, Overpass)"
                   ).set_defaults(fn=cmd_check)
    c = sub.add_parser("coverage", help="imagery and OSM coverage report for the area")
    c.add_argument("--refresh", action="store_true", help="re-fetch the Mapillary index")
    c.set_defaults(fn=cmd_coverage)
    f = sub.add_parser("fetch-osm", help="walking graph, kerbs, steps, barriers + tag summary")
    f.add_argument("--refresh", action="store_true", help="re-download instead of using cache")
    f.set_defaults(fn=cmd_fetch_osm)
    i = sub.add_parser("fetch-images", help="select, download and crop Mapillary frames")
    i.add_argument("--refresh", action="store_true",
                   help="re-fetch metadata and redo the selection")
    i.add_argument("--reselect", action="store_true",
                   help="redo the selection from cached metadata")
    i.set_defaults(fn=cmd_fetch_images)
    a = sub.add_parser("analyze", help="run Gemini on frames -> data/processed/detections.jsonl")
    a.add_argument("--model")
    a.add_argument("--prompt", help="default: gemini.prompt in config/project.yaml")
    a.add_argument("--retry-failed", action="store_true",
                   help="call again for frames whose cached outcome is 'no valid answer'")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="all frames in the manifest")
    g.add_argument("--limit", type=int, help="first N frames (for quick tests)")
    g.add_argument("--subset", choices=["tuning"],
                   help="phase 5 prompt-tuning subset (labels/selection.json); writes to "
                        "data/processed/runs/ instead of detections.jsonl")
    a.set_defaults(fn=cmd_analyze)
    pl = sub.add_parser("pilot", help="same N frames on each pilot model + comparison sheets")
    pl.add_argument("--models", nargs="+")
    pl.add_argument("-n", type=int, default=30)
    pl.add_argument("--prompt", default="v1")
    pl.set_defaults(fn=cmd_pilot)
    e = sub.add_parser("evaluate", help="metrics vs hand labels and OSM -> reports/metrics.json")
    e.add_argument("--model")
    e.add_argument("--prompt", help="default: gemini.prompt in config/project.yaml")
    e.add_argument("--final", action="store_true",
                   help="also score the hold-out (only for the chosen final prompt)")
    e.set_defaults(fn=cmd_evaluate)
    sub.add_parser("geolocate", help="place, cluster and snap detections -> barriers.geojson"
                   ).set_defaults(fn=cmd_geolocate)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page (cp1251/cp1252) that can't print "²".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
