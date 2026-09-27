"""Equirectangular panorama -> perspective crops (SPEC §3.4).

The centre column of a Mapillary panorama faces `compass_angle`; moving right in the image
turns clockwise, which matches py360convert's positive u_deg (verified in tests).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

CROP_NAMES = {4: ["forward", "right", "back", "left"]}


def crop_offsets(n: int) -> list[tuple[str, float]]:
    """(name, yaw offset from heading) for n evenly spaced crops."""
    names = CROP_NAMES.get(n) or [f"yaw{int(i * 360 / n):03d}" for i in range(n)]
    return [(name, i * 360.0 / n) for i, name in enumerate(names)]


def perspective_crops(equirect: Image.Image, heading: float, n: int = 4, fov: float = 90.0,
                      out_px: int = 768) -> list[tuple[str, float, Image.Image]]:
    """Returns (name, absolute bearing of the crop centre, image) per crop."""
    import py360convert

    e = np.asarray(equirect.convert("RGB"))
    out = []
    for name, off in crop_offsets(n):
        u = ((off + 180) % 360) - 180  # py360convert wants [-180, 180]
        p = py360convert.e2p(e, fov_deg=(fov, fov), u_deg=u, v_deg=0.0,
                             out_hw=(out_px, out_px))
        out.append((name, (heading + off) % 360, Image.fromarray(p.astype(np.uint8))))
    return out
