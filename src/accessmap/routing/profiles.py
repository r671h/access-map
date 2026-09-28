"""Routing profiles from config/profiles.yaml (SPEC §8)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# OSM surface values that the profile tables do not list, mapped to the closest listed one
# (DECISIONS.md, phase 6). Anything else unknown uses the profile's `unknown` multiplier.
SURFACE_ALIASES = {
    "paved": "asphalt", "metal": "concrete", "concrete:plates": "concrete",
    "concrete:lanes": "concrete", "wood": "paving_stones", "grass_paver": "cobblestone",
    "unhewn_cobblestone": "cobblestone", "pebblestone": "gravel", "fine_gravel": "gravel",
    "compacted": "gravel", "dirt": "unpaved", "ground": "unpaved", "earth": "unpaved",
    "grass": "unpaved", "mud": "unpaved", "sand": "unpaved",
}


class Defaults(BaseModel):
    forbid_threshold: float = 0.6
    fallback_penalty_m: float = 400
    temporary_factor: float = 0.5
    temporary_max_age_days: int = 365


class Profile(BaseModel):
    label: str
    forbidden: list[str] = []
    penalty_m: dict[str, float] = {}
    surface_multiplier: dict[str, float] = {}
    osm_rules: dict[str, Literal["forbid", "ignore"]] = Field(default_factory=dict)


class ProfilesFile(BaseModel):
    defaults: Defaults = Field(default_factory=Defaults)
    profiles: dict[str, Profile]


def load_profiles(path: Path, active: list[str] | None = None) -> ProfilesFile:
    with open(path, encoding="utf-8") as f:
        pf = ProfilesFile.model_validate(yaml.safe_load(f))
    if active is not None:
        missing = [p for p in active if p not in pf.profiles]
        if missing:
            raise ValueError(f"profiles {missing} are not defined in {path}")
        pf.profiles = {k: pf.profiles[k] for k in active}
    return pf
