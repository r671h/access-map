"""Prompt templates (vision/prompts/vN.txt); the version is part of the cache key."""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
CAMERA_HEIGHT_M = 2.0

DEFINITIONS = {
    "curb_ramp": "a lowered or flush curb at a crossing or driveway that allows rolling\n"
                 "  from sidewalk to street. This is a POSITIVE feature; report it.",
    "raised_curb": "a curb at a crossing point or corner that is NOT lowered\n"
                   "  (a step down to the street, typically 8-15 cm).",
    "step": "a single step or threshold (> 2 cm) that interrupts a sidewalk or path.",
    "stairs": "two or more steps forming a staircase on a pedestrian route.",
    "narrow_passage": "an obstacle (pole, bollard, sign, tree, bench, bin) leaving less\n"
                      "  than about 1 m of clear width on the sidewalk.",
    "steep_slope": "a clearly steep sidewalk or ramp (noticeably steeper than normal streets).",
    "rough_surface": "cobblestones, setts, gravel, broken or heavily cracked pavement.",
    "no_sidewalk": "pedestrians would have to walk on the roadway.",
    "construction": "construction works, fences or barriers blocking the sidewalk.",
    "parked_vehicle": "cars, scooters or bikes parked on the sidewalk blocking passage.",
}


# Per-version definition overrides. v1 uses DEFINITIONS unchanged; editing DEFINITIONS
# itself would change every version's rendered prompt and invalidate its cache.
DEFINITION_OVERRIDES: dict[str, dict[str, str]] = {
    "v2": {
        "curb_ramp": "a lowered or flush curb exactly where a pedestrian crossing (zebra,\n"
                     "  signalised crossing, or a sidewalk corner leading to a crossing) or a\n"
                     "  driveway meets the roadway. This is a POSITIVE feature; report it.\n"
                     "  NOT a curb_ramp: grass verges or traffic islands, building gateways or\n"
                     "  doorsteps, flush pedestrian zones with no roadway edge, cycle lanes.",
        "raised_curb": "a curb that is NOT lowered at a pedestrian crossing point: where a\n"
                       "  marked crossing, or a sidewalk corner a pedestrian would cross from,\n"
                       "  meets the roadway (a step down, typically 8-15 cm).\n"
                       "  NOT a raised_curb: ordinary curbs running ALONG the street between the\n"
                       "  sidewalk and the roadway, parking lane or median. Never report these.",
        "step": "a single step or threshold (> 2 cm) that interrupts the walking route along\n"
                "  a sidewalk or path. NOT a step: a step up into a door or shop entrance.",
        "stairs": "two or more steps that are themselves part of a public pedestrian route\n"
                  "  (e.g. between two plaza levels or up an embankment).\n"
                  "  NOT stairs: stairs leading into a building entrance, loading docks,\n"
                  "  bicycle racks, railings or striped paving.",
    },
}
# v3 = v2 with the stairs regression undone ("striped paving" made the model miss real
# plaza stairs) and tighter curb_ramp / stairs distance rules.
DEFINITION_OVERRIDES["v3"] = DEFINITION_OVERRIDES["v2"] | {
    "curb_ramp": "a lowered or flush curb exactly where a pedestrian crossing (zebra,\n"
                 "  signalised crossing, or a sidewalk corner leading to a crossing) or a\n"
                 "  driveway meets the roadway, clearly visible within about 15 m.\n"
                 "  This is a POSITIVE feature; report it.\n"
                 "  NOT a curb_ramp: grass verges or traffic islands, the threshold of a\n"
                 "  building gateway or doorstep, flush pedestrian zones with no roadway\n"
                 "  edge, cycle lanes, crossings further than about 15 m away.",
    "stairs": "two or more steps that are themselves part of a public pedestrian route,\n"
              "  e.g. wide outdoor stairs between two plaza levels or up an embankment\n"
              "  (these DO count, even seen from above or behind a railing).\n"
              "  NOT stairs: stairs leading into a building entrance (steps far away\n"
              "  next to a facade are almost always entrances), loading docks,\n"
              "  bicycle racks.",
}


def load_template(version: str) -> str:
    text = (PROMPT_DIR / f"{version}.txt").read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.startswith("#")).strip()


def render(version: str, *, active_types: list[str], fov_deg: float, captured_at: str,
           camera_height_m: float = CAMERA_HEIGHT_M) -> str:
    """Fill the template; definitions of inactive types are left out."""
    defs = DEFINITIONS | DEFINITION_OVERRIDES.get(version, {})
    definitions = "\n".join(f"- {t}: {defs[t]}" for t in active_types)
    return load_template(version).format(
        camera_height_m=camera_height_m,
        fov_deg=round(fov_deg),
        captured_at=captured_at,
        active_types=", ".join(active_types),
        definitions=definitions,
    )
