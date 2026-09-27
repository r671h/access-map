"""Configuration: secrets from the environment / .env, everything else from config/project.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BARRIER_TYPES = (
    "curb_ramp",
    "raised_curb",
    "step",
    "stairs",
    "narrow_passage",
    "steep_slope",
    "rough_surface",
    "no_sidewalk",
    "construction",
    "parked_vehicle",
)


def find_root(start: Path | None = None) -> Path:
    """Project root: $ACCESSMAP_ROOT, else the first parent containing config/project.yaml."""
    if env := os.environ.get("ACCESSMAP_ROOT"):
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / "config" / "project.yaml").is_file():
            return p
    return Path(__file__).resolve().parents[2]


class Secrets(BaseSettings):
    """API keys. Read from environment variables, falling back to .env. Never logged."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str | None = None
    mapillary_token: str | None = None


class Area(BaseModel):
    name: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    confirmed: bool = False

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: tuple[float, float, float, float]):
        min_lon, min_lat, max_lon, max_lat = v
        if not (min_lon < max_lon and min_lat < max_lat):
            raise ValueError("bbox must be min_lon, min_lat, max_lon, max_lat")
        if not (-180 <= min_lon <= 180 and -90 <= min_lat <= 90):
            raise ValueError("bbox out of range")
        return v


class UI(BaseModel):
    languages: list[str] = ["en"]
    language: str = "en"


class Gemini(BaseModel):
    model: str | None = None
    pilot_candidates: list[str] = []
    prefer_free_tier: bool = True
    rpm: int = 60
    thinking: str = "minimal"


class Budget(BaseModel):
    max_images: int = 800
    max_gemini_calls: int = 4000


class Imagery(BaseModel):
    sources: list[Literal["mapillary", "own_photos"]] = ["mapillary"]
    max_age_years: float = 6
    pano_crops: int = 4


class Detection(BaseModel):
    types: list[str] = list(BARRIER_TYPES)
    preference: Literal["precision", "recall", "balanced"] = "precision"

    @field_validator("types")
    @classmethod
    def _known_types(cls, v: list[str]):
        unknown = set(v) - set(BARRIER_TYPES)
        if unknown:
            raise ValueError(f"unknown barrier types: {sorted(unknown)}")
        return v


class Geo(BaseModel):
    snap_curb_m: float = 12
    snap_other_m: float = 10


class Routing(BaseModel):
    max_detour: float | None = 0.5
    temporary_barriers: Literal["half_weight", "ignore", "permanent"] = "half_weight"


class App(BaseModel):
    features: list[str] = ["map_routes", "feedback"]
    layout: str = "side_panel"
    basemap: str = "light"
    run_mode: Literal["local", "docker", "hosting_ready"] = "local"


class ProjectConfig(BaseModel):
    goal: Literal["portfolio", "study", "real_users"] = "portfolio"
    autonomy: Literal["guided", "checkpoints", "autonomous"] = "checkpoints"
    area: Area
    profiles: list[str] = ["wheelchair", "stroller", "suitcase"]
    ui: UI = Field(default_factory=UI)
    gemini: Gemini = Field(default_factory=Gemini)
    budget: Budget = Field(default_factory=Budget)
    imagery: Imagery = Field(default_factory=Imagery)
    detection: Detection = Field(default_factory=Detection)
    geo: Geo = Field(default_factory=Geo)
    routing: Routing = Field(default_factory=Routing)
    app: App = Field(default_factory=App)


class Paths:
    """Standard locations under the project root."""

    def __init__(self, root: Path):
        self.root = root
        self.config = root / "config"
        self.data = root / "data"
        self.raw = self.data / "raw"
        self.images = self.data / "images"
        self.cache = self.data / "cache"
        self.processed = self.data / "processed"
        self.reports = root / "reports"
        self.qa = self.reports / "qa"

    def ensure(self) -> None:
        for p in (self.raw, self.images, self.cache, self.processed, self.qa):
            p.mkdir(parents=True, exist_ok=True)


def load_project(path: Path) -> ProjectConfig:
    with open(path, encoding="utf-8") as f:
        return ProjectConfig.model_validate(yaml.safe_load(f))


class Settings:
    def __init__(self, root: Path | None = None):
        self.root = root or find_root()
        self.paths = Paths(self.root)
        self.project = load_project(self.root / "config" / "project.yaml")
        self.secrets = Secrets(_env_file=self.root / ".env")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
