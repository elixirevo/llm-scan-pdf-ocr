"""Markdown assembly tests."""

from __future__ import annotations

from pathlib import Path

from pdf_ocr.llm.schema import BBox, Block, PageLayout
from pdf_ocr.ocr.crop import Asset
from pdf_ocr.render.markdown import PageResult, render_document


def test_render_interleaves_text_and_images():
    layout = PageLayout(
        page_no=1,
        blocks=[
            Block(type="text", order=1, text="제목"),
            Block(type="figure", order=2,
                  bbox=BBox(x=0.0, y=0.0, w=0.5, h=0.5),
                  caption="그림 1. 흐름도"),
            Block(type="text", order=3, text="본문입니다."),
        ],
    )
    asset = Asset(
        page_no=1, order=2, kind="fig", idx=1,
        path=Path("/tmp/x.png"), rel_path="doc_images/p001_fig01.png",
    )
    md = render_document([PageResult(layout=layout, assets=[asset])], title="doc")
    assert "# doc" in md
    assert "제목" in md
    assert "![그림 1. 흐름도](doc_images/p001_fig01.png)" in md
    assert "*그림 1. 흐름도*" in md
    assert "본문입니다." in md


def test_page_separator_between_pages():
    p1 = PageResult(
        layout=PageLayout(page_no=1, blocks=[Block(type="text", order=1, text="a")]),
        assets=[],
    )
    p2 = PageResult(
        layout=PageLayout(page_no=2, blocks=[Block(type="text", order=1, text="b")]),
        assets=[],
    )
    md = render_document([p1, p2])
    assert "<!-- page 1 -->" in md
    assert "<!-- page 2 -->" in md
    assert "\n---\n" in md
