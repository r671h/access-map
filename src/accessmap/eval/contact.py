"""Contact sheets: grids of frames with captions and optional Gemini boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE_W, TILE_H = 480, 360
CAPTION_H = 34

TYPE_COLORS = {
    "curb_ramp": (46, 204, 113),
    "raised_curb": (231, 76, 60),
    "step": (230, 126, 34),
    "stairs": (192, 57, 43),
    "narrow_passage": (155, 89, 182),
    "steep_slope": (241, 196, 15),
    "rough_surface": (52, 152, 219),
    "no_sidewalk": (236, 64, 122),
    "construction": (255, 152, 0),
    "parked_vehicle": (0, 188, 212),
}


@dataclass
class Box:
    box_2d: list[int]  # [ymin, xmin, ymax, xmax], 0-1000 (Gemini order)
    label: str
    type: str = ""


@dataclass
class Tile:
    path: Path
    caption: str
    boxes: list[Box] = field(default_factory=list)


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def box_to_pixels(box_2d: list[int], w: int, h: int) -> tuple[float, float, float, float]:
    """Gemini [ymin, xmin, ymax, xmax] (0-1000) -> PIL (x0, y0, x1, y1) pixels."""
    ymin, xmin, ymax, xmax = box_2d
    return xmin / 1000 * w, ymin / 1000 * h, xmax / 1000 * w, ymax / 1000 * h


def render_tile(t: Tile) -> Image.Image:
    img = Image.open(t.path).convert("RGB")
    img.thumbnail((TILE_W, TILE_H))
    canvas = Image.new("RGB", (TILE_W, TILE_H + CAPTION_H), (20, 20, 20))
    ox, oy = (TILE_W - img.width) // 2, (TILE_H - img.height) // 2
    canvas.paste(img, (ox, oy))
    d = ImageDraw.Draw(canvas)
    small = _font(13)
    for b in t.boxes:
        x0, y0, x1, y1 = box_to_pixels(b.box_2d, img.width, img.height)
        color = TYPE_COLORS.get(b.type, (255, 255, 255))
        d.rectangle([ox + x0, oy + y0, ox + x1, oy + y1], outline=color, width=3)
        ty = max(oy, oy + y0 - 16)
        tw = d.textlength(b.label, font=small)
        d.rectangle([ox + x0, ty, ox + x0 + tw + 6, ty + 16], fill=color)
        d.text((ox + x0 + 3, ty + 1), b.label, fill=(0, 0, 0), font=small)
    d.text((6, TILE_H + 4), t.caption[:70], fill=(230, 230, 230), font=_font(14))
    return canvas


def contact_sheet(tiles: list[Tile], out: Path, cols: int = 4, title: str = "") -> Path:
    rows = max(1, -(-len(tiles) // cols))
    head = 40 if title else 0
    sheet = Image.new("RGB", (cols * TILE_W, head + rows * (TILE_H + CAPTION_H)), (0, 0, 0))
    if title:
        ImageDraw.Draw(sheet).text((10, 8), title, fill=(255, 255, 255), font=_font(22))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(render_tile(t), (c * TILE_W, head + r * (TILE_H + CAPTION_H)))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=85)
    return out
