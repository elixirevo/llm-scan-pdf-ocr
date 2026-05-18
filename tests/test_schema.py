"""Schema / postprocess sanity tests — no VLM required."""

from __future__ import annotations

from pytest import approx

from pdf_ocr.llm.schema import BBox, Block, PageLayout


def test_bbox_clamps_overflow_instead_of_rejecting():
    """VLMs often overshoot edge-touching figures by a few percent.

    The schema used to ``raise ValidationError`` on these, which killed entire
    pages — now we clamp w/h so the box fits.
    """
    b = BBox(x=0.9, y=0.0, w=0.5, h=0.1)
    assert b.x + b.w <= 1.0001
    assert b.w == approx(0.1)  # 1.0 - 0.9


def test_bbox_real_world_failure_case():
    """Regression: ``y=0.55, h=0.65`` produced by Qwen3-VL on a real page."""
    b = BBox(x=0.0, y=0.55, w=0.5, h=0.65)
    assert b.y + b.h <= 1.0001
    assert b.h == approx(0.45)


def test_bbox_negative_coords_clamped_to_zero():
    b = BBox(x=-0.1, y=-0.05, w=0.3, h=0.2)
    assert b.x == 0.0 and b.y == 0.0


def test_bbox_full_page_ok():
    b = BBox(x=0.0, y=0.0, w=1.0, h=1.0)
    assert not b.is_degenerate


def test_bbox_degenerate_after_clamp():
    # x already at 1.0 → w gets clamped to 0
    b = BBox(x=1.0, y=0.0, w=0.5, h=0.5)
    assert b.is_degenerate


def test_page_layout_normalization_sorts_and_compacts():
    layout = PageLayout(
        page_no=2,
        blocks=[
            Block(type="text", order=10, text="second"),
            Block(type="text", order=3, text="first"),
        ],
    ).normalized()
    assert [b.text for b in layout.blocks] == ["first", "second"]
    assert [b.order for b in layout.blocks] == [1, 2]


def test_figure_without_bbox_is_allowed_at_schema_level():
    """Downstream crop will skip it; schema doesn't enforce."""
    b = Block(type="figure", order=1, caption="x")
    assert b.bbox is None
