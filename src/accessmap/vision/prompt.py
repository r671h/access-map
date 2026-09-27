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


def load_template(version: str) -> str:
    text = (PROMPT_DIR / f"{version}.txt").read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.startswith("#")).strip()


def render(version: str, *, active_types: list[str], fov_deg: float, captured_at: str,
           camera_height_m: float = CAMERA_HEIGHT_M) -> str:
    """Fill the template; definitions of inactive types are left out."""
    definitions = "\n".join(f"- {t}: {DEFINITIONS[t]}" for t in active_types)
    return load_template(version).format(
        camera_height_m=camera_height_m,
        fov_deg=round(fov_deg),
        captured_at=captured_at,
        active_types=", ".join(active_types),
        definitions=definitions,
    )
