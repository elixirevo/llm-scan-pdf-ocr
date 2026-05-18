"""Verify bbox→pixel conversion and PNG writing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pdf_ocr.llm.schema import BBox, Block, PageLayout
from pdf_ocr.ocr.crop import crop_assets
from pdf_ocr.pdf.page import Page


def _solid_page(w: int = 1000, h: int = 1400) -> Page:
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    return Page.from_image(page_no=1, image=img)


def test_crop_assets_writes_figure_and_table(tmp_path: Path):
    page = _solid_page()
    layout = PageLayout(
        page_no=1,
        blocks=[
            Block(type="text", order=1, text="hi"),
            Block(type="figure", order=2, bbox=BBox(x=0.1, y=0.1, w=0.3, h=0.2)),
            Block(type="table",  order=3, bbox=BBox(x=0.5, y=0.5, w=0.4, h=0.3),
                  caption="표 1"),
        ],
    )
    assets = crop_assets(
        page, layout,
        images_dir=tmp_path / "imgs",
        asset_name="p{page:03d}_{kind}{idx:02d}.png",
    )
    assert len(assets) == 2
    assert assets[0].kind == "fig" and assets[0].path.exists()
    assert assets[1].kind == "tab" and assets[1].path.exists()
    # Sanity: cropped image is non-empty and smaller than the page.
    fig = Image.open(assets[0].path)
    assert 0 < fig.width < page.width
    assert 0 < fig.height < page.height


def test_crop_skips_blocks_without_bbox(tmp_path: Path):
    page = _solid_page()
    layout = PageLayout(
        page_no=1,
        blocks=[Block(type="figure", order=1)],  # no bbox
    )
    assets = crop_assets(
        page, layout,
        images_dir=tmp_path / "imgs",
        asset_name="p{page:03d}_{kind}{idx:02d}.png",
    )
    assert assets == []


def test_crop_skips_degenerate_bbox_after_clamping(tmp_path: Path):
    """A bbox that gets squashed to zero area by clamping should not produce a file."""
    page = _solid_page()
    # x=1.0 forces w to clamp to 0 → degenerate
    layout = PageLayout(
        page_no=1,
        blocks=[Block(type="figure", order=1, bbox=BBox(x=1.0, y=0.0, w=0.5, h=0.5))],
    )
    assets = crop_assets(
        page, layout,
        images_dir=tmp_path / "imgs",
        asset_name="p{page:03d}_{kind}{idx:02d}.png",
    )
    assert assets == []
