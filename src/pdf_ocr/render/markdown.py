"""Assemble per-page :class:`PageLayout`s + cropped assets into a single .md file."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from ..llm.schema import Block, PageLayout
from ..ocr.crop import Asset


@dataclass(slots=True)
class PageResult:
    layout: PageLayout
    assets: list[Asset]


def _assets_by_order(assets: list[Asset]) -> dict[int, Asset]:
    return {a.order: a for a in assets}


def _render_block(block: Block, asset: Asset | None, page_no: int) -> str:
    if block.type == "text":
        return (block.text or "").rstrip() + "\n"

    # figure / table
    if asset is None:
        # Model returned a figure/table block but we couldn't crop it (bad bbox, etc.).
        # Emit a placeholder so the reader knows something was there.
        kind_ko = "그림" if block.type == "figure" else "표"
        cap = block.caption or f"{kind_ko} (p.{page_no})"
        return f"> *[{kind_ko}: {cap}]* (영역을 추출하지 못했습니다)\n"

    alt = block.caption or f"{block.type} p.{page_no}"
    md = f"![{alt}]({asset.rel_path})\n"
    if block.caption:
        md += f"\n*{block.caption.strip()}*\n"
    return md


def render_document(
    pages: list[PageResult],
    *,
    title: str | None = None,
    page_separator: bool = True,
) -> str:
    """Return the full markdown document as a string."""
    out = StringIO()
    if title:
        out.write(f"# {title}\n\n")

    for i, page in enumerate(pages):
        layout = page.layout
        assets = _assets_by_order(page.assets)

        if page_separator and i > 0:
            out.write("\n---\n\n")
        out.write(f"<!-- page {layout.page_no} -->\n\n")

        prev_was_text = False
        for block in layout.blocks:
            chunk = _render_block(block, assets.get(block.order), layout.page_no)
            if prev_was_text and block.type == "text":
                # paragraph break between consecutive text blocks
                out.write("\n")
            out.write(chunk)
            out.write("\n")
            prev_was_text = block.type == "text"

    return out.getvalue()


def write_document(
    pages: list[PageResult],
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_document(pages, title=title), encoding="utf-8")
    return out_path
