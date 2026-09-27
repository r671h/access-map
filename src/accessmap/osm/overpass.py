"""Overpass API client that falls back across public mirrors (the main one is often busy)."""

from __future__ import annotations

import logging

from accessmap.http import HttpError, request

log = logging.getLogger(__name__)

ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)


def overpass_bbox(bbox: tuple[float, float, float, float]) -> str:
    """Overpass uses (south, west, north, east)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return f"{min_lat},{min_lon},{max_lat},{max_lon}"


def query(ql: str, *, timeout: float = 180, endpoints: tuple[str, ...] = ENDPOINTS) -> dict:
    """Run an Overpass QL query, trying each endpoint in turn. Returns parsed JSON."""
    errors = []
    for url in endpoints:
        try:
            r = request("POST", url, data={"data": ql}, timeout=timeout, retries=2)
            if "json" not in r.headers.get("Content-Type", ""):
                # Overpass reports overload as an HTML page, sometimes with status 200.
                raise HttpError(f"non-JSON response: {r.text[:200]}")
            return r.json()
        except Exception as e:  # noqa: BLE001 - try the next mirror
            log.warning("Overpass %s failed: %s", url, e)
            errors.append(f"{url}: {e}")
    raise HttpError("all Overpass endpoints failed:\n" + "\n".join(errors))


def count(bbox: tuple[float, float, float, float], selectors: dict[str, str]) -> dict[str, int]:
    """Count OSM elements per named selector, e.g. {"kerb": 'node["kerb"]'}."""
    b = overpass_bbox(bbox)
    body = "".join(f"{sel}({b});out count;" for sel in selectors.values())
    data = query(f"[out:json][timeout:120];{body}")
    elements = data.get("elements", [])
    if len(elements) != len(selectors):
        raise HttpError(f"expected {len(selectors)} counts, got {len(elements)}")
    return {name: int(el["tags"]["total"]) for name, el in zip(selectors, elements, strict=True)}
