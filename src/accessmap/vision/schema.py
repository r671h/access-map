"""Gemini response schema (docs/GEMINI_PROMPTS.md), also used for strict validation.

Deviation from the doc: `description` is split into `description_en` / `description_de`
because the UI is bilingual (DECISIONS.md, 2026-09-27).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, create_model, field_validator

SidewalkSurface = Literal[
    "asphalt", "concrete", "paving_stones", "sett", "cobblestone", "gravel", "unpaved",
    "unknown", "none",
]


class FeatureBase(BaseModel):
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] normalized to 0-1000")
    confidence: float = Field(ge=0, le=1)
    estimated_distance_m: float | None = None
    estimated_height_cm: float | None = None
    estimated_width_m: float | None = None
    permanence: Literal["permanent", "temporary"]
    description_en: str
    description_de: str

    @field_validator("box_2d")
    @classmethod
    def _box(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError("box_2d must have 4 values")
        ymin, xmin, ymax, xmax = v
        if not all(0 <= c <= 1000 for c in v):
            raise ValueError("box_2d values must be within 0-1000")
        if ymin > ymax or xmin > xmax:
            raise ValueError("box_2d must be [ymin, xmin, ymax, xmax]")
        return v


@lru_cache(maxsize=8)
def response_model(active_types: tuple[str, ...]) -> type[BaseModel]:
    """Response model with the `type` enum restricted to the active barrier types."""
    type_enum = Enum("BarrierType", {t: t for t in active_types}, type=str)
    feature = create_model("Feature", __base__=FeatureBase, type=(type_enum, ...))
    return create_model(
        "FrameAnalysis",
        image_usable=(bool, ...),
        sidewalk_visible=(bool, ...),
        sidewalk_surface=(SidewalkSurface, ...),
        features=(list[feature], ...),
    )
