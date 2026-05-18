"""Render PDF pages to PIL images using pypdfium2.

pypdfium2 is preferred over PyMuPDF for licensing reasons (Apache/BSD vs AGPL)
and produces high-quality scans suitable for VLM ingestion.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from .page import Page


def _scale_for_dpi(dpi: int) -> float:
    # pypdfium2 renders at 72 DPI by default; ``scale`` multiplies that.
    return dpi / 72.0


def _maybe_downscale(img: Image.Image, max_side: int | None) -> Image.Image:
    if max_side is None or max(img.size) <= max_side:
        return img
    w, h = img.size
    if w >= h:
        new_w = max_side
        new_h = round(h * (max_side / w))
    else:
        new_h = max_side
        new_w = round(w * (max_side / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def render_pdf(
    pdf_path: str | Path,
    *,
    dpi: int = 200,
    max_side: int | None = 2048,
) -> Iterator[Page]:
    """Yield :class:`Page` objects one at a time.

    Uses a generator so we never hold every page in memory — important for
    multi-hundred-page scans.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    scale = _scale_for_dpi(dpi)
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        for i, pdf_page in enumerate(doc, start=1):
            bitmap = pdf_page.render(scale=scale)
            img = bitmap.to_pil()
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = _maybe_downscale(img, max_side)
            yield Page.from_image(page_no=i, image=img)
            pdf_page.close()
    finally:
        doc.close()


def page_count(pdf_path: str | Path) -> int:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()
