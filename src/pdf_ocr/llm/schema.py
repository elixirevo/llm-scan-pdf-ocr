"""Pydantic schema for the VLM's per-page response.

Coordinates are **[0, 1] normalized**, origin top-left:
    x, y, w, h ∈ [0, 1], x+w ≤ 1, y+h ≤ 1
This avoids ambiguity when the model resizes the input internally.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    @field_validator("w")
    @classmethod
    def _w_fits(cls, v: float, info) -> float:  # noqa: ANN001
        x = info.data.get("x", 0.0)
        if x + v > 1.0001:
            raise ValueError(f"bbox overflows width: x={x}, w={v}")
        return v

    @field_validator("h")
    @classmethod
    def _h_fits(cls, v: float, info) -> float:  # noqa: ANN001
        y = info.data.get("y", 0.0)
        if y + v > 1.0001:
            raise ValueError(f"bbox overflows height: y={y}, h={v}")
        return v


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
