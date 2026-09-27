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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="accessmap", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify keys and API access (Gemini, Mapillary, Overpass)"
                   ).set_defaults(fn=cmd_check)
    c = sub.add_parser("coverage", help="imagery and OSM coverage report for the area")
    c.add_argument("--refresh", action="store_true", help="re-fetch the Mapillary index")
    c.set_defaults(fn=cmd_coverage)
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
