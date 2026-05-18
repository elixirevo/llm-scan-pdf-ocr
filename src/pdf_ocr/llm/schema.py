"""Pydantic schema for the VLM's per-page response.

Coordinates are **[0, 1] normalized**, origin top-left:
    x, y, w, h ∈ [0, 1], x+w ≤ 1, y+h ≤ 1
We intentionally *clamp* slightly out-of-range boxes (rather than reject them)
because VLMs routinely overshoot edge-touching figures by a few percent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _clip01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

    @model_validator(mode="after")
    def _clamp(self) -> "BBox":
        # Clip individual coordinates to [0, 1].
        self.x = _clip01(self.x)
        self.y = _clip01(self.y)
        self.w = _clip01(self.w)
        self.h = _clip01(self.h)
        # Ensure the box fits inside the page; shrink width/height if it overflows.
        if self.x + self.w > 1.0:
            self.w = max(0.0, 1.0 - self.x)
        if self.y + self.h > 1.0:
            self.h = max(0.0, 1.0 - self.y)
        return self

    @property
    def is_degenerate(self) -> bool:
        """True if the box has zero (or near-zero) area after clamping."""
        return self.w <= 0.001 or self.h <= 0.001


BlockType = Literal["text", "figure", "table"]


class Block(BaseModel):
    """A single content block on the page, in reading order."""

    type: BlockType
    order: int = Field(ge=1, description="1-indexed reading order on the page.")
    # text blocks: required. figure/table: optional caption text.
    text: str | None = None
    caption: str | None = None
    # required for figure/table; ignored for text.
    bbox: BBox | None = None


class PageLayout(BaseModel):
    page_no: int = Field(ge=1)
    blocks: list[Block] = Field(default_factory=list)

    def normalized(self) -> "PageLayout":
        """Return a copy with blocks sorted by ``order`` and order numbers compacted."""
        sorted_blocks = sorted(self.blocks, key=lambda b: b.order)
        for i, b in enumerate(sorted_blocks, start=1):
            b.order = i
        return PageLayout(page_no=self.page_no, blocks=sorted_blocks)


# JSON schema dict suitable for OpenAI-style ``response_format``.
PAGE_LAYOUT_JSON_SCHEMA = {
    "name": "PageLayout",
    "schema": PageLayout.model_json_schema(),
    "strict": True,
}
