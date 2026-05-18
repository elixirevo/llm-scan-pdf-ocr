"""Schema / postprocess sanity tests — no VLM required."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pdf_ocr.llm.schema import BBox, Block, PageLayout


def test_bbox_rejects_overflow():
    with pytest.raises(ValidationError):
        BBox(x=0.9, y=0.0, w=0.5, h=0.1)  # x+w > 1


def test_bbox_accepts_full_page():
    BBox(x=0.0, y=0.0, w=1.0, h=1.0)


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


def test_figure_requires_bbox_at_runtime():
    """Schema permits bbox=None but downstream crop will skip it."""
    b = Block(type="figure", order=1, caption="x")
    assert b.bbox is None
