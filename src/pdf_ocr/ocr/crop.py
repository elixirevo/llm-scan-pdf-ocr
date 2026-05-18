"""Crop figure/table regions from page images and save as PNGs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL.Image import Image

from ..llm.schema import BBox, PageLayout
from ..pdf.page import Page


@dataclass(slots=True)
class Asset:
    """A saved image asset referenced from the final markdown."""

    page_no: int
    order: int            # block order on the page
    kind: str             # "fig" or "tab"
    idx: int              # 1-indexed per page+kind
    path: Path            # absolute path on disk
    rel_path: str         # path relative to the .md (used in the link)


def _bbox_to_pixels(bbox: BBox, w: int, h: int, *, pad: float = 0.01) -> tuple[int, int, int, int]:
    """Convert a normalized bbox to (left, top, right, bottom) pixel coords with a small pad."""
    x = max(0.0, bbox.x - pad)
    y = max(0.0, bbox.y - pad)
    x2 = min(1.0, bbox.x + bbox.w + pad)
    y2 = min(1.0, bbox.y + bbox.h + pad)
    return (round(x * w), round(y * h), round(x2 * w), round(y2 * h))


def crop_assets(
    page: Page,
    layout: PageLayout,
    *,
    images_dir: Path,
    asset_name: str,
) -> list[Asset]:
    """Crop every figure/table block on the page and write PNGs.

    Returns the list of :class:`Asset`s in block order. Mutates nothing.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    page_img: Image = page.image
    w, h = page.width, page.height

    assets: list[Asset] = []
    fig_idx = 0
    tab_idx = 0

    for block in layout.blocks:
        if block.type not in ("figure", "table") or block.bbox is None:
            continue
        if block.type == "figure":
            fig_idx += 1
            kind, idx = "fig", fig_idx
        else:
            tab_idx += 1
            kind, idx = "tab", tab_idx

        left, top, right, bottom = _bbox_to_pixels(block.bbox, w, h)
        # Guard against degenerate boxes from the model.
        if right - left < 4 or bottom - top < 4:
            continue
        crop = page_img.crop((left, top, right, bottom))

        filename = asset_name.format(page=page.page_no, kind=kind, idx=idx)
        out_path = images_dir / filename
        crop.save(out_path, format="PNG", optimize=True)

        assets.append(
            Asset(
                page_no=page.page_no,
                order=block.order,
                kind=kind,
                idx=idx,
                path=out_path,
                rel_path=f"{images_dir.name}/{filename}",
            )
        )

    return assets
