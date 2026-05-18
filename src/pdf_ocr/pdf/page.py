"""Page data model passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

from PIL.Image import Image


@dataclass(slots=True)
class Page:
    """A single rendered PDF page.

    ``width`` / ``height`` are the rendered image's pixel dimensions and are the
    reference for any bbox the VLM returns (which we treat as [0, 1] normalized).
    """

    page_no: int            # 1-indexed
    image: Image
    width: int
    height: int

    @classmethod
    def from_image(cls, page_no: int, image: Image) -> "Page":
        return cls(page_no=page_no, image=image, width=image.width, height=image.height)
