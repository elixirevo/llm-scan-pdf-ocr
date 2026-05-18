"""When one page errors, the rest should still finish and be written.

Exercises the VLM backend directly with a fake client so no server is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pdf_ocr.backends import vlm as vlm_backend
from pdf_ocr.backends.vlm import VLMBackend
from pdf_ocr.pdf.page import Page


class _FakeClient:
    """Mimics VLMClient. Page 2 always fails."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat_image_json(self, image, *, system, user, max_tokens=None):
        self.calls += 1
        page_no = int(user.splitlines()[0].split(":")[1].strip())
        if page_no == 2:
            raise RuntimeError("simulated VLM failure on page 2")
        return (
            '{"page_no": %d, "blocks": ['
            '{"type":"text","order":1,"text":"page %d body"}'
            "]}"
        ) % (page_no, page_no)

    async def aclose(self) -> None:
        pass


def _make_pages(n: int) -> list[Page]:
    return [
        Page.from_image(page_no=i, image=Image.new("RGB", (400, 600), "white"))
        for i in range(1, n + 1)
    ]


@pytest.mark.asyncio
async def test_failed_page_does_not_kill_pipeline(tmp_path: Path, monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vlm_backend, "VLMClient", lambda cfg: fake)

    cfg = {
        "llm": {
            "base_url": "x", "api_key": "x", "model": "x",
            "use_response_format": False, "timeout": 5, "max_retries": 0,
            "max_tokens": 256,
        },
        "render": {"dpi": 100, "max_side": 800},
        "pipeline": {"concurrency": 2},
        "output": {
            "markdown_path": "${stem}.md",
            "images_dir": "${stem}_images",
            "asset_name": "p{page:03d}_{kind}{idx:02d}.png",
        },
    }
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.touch()
    pages = _make_pages(3)

    result = await VLMBackend().run(pdf_path, tmp_path / "out", cfg, pages=pages)
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "page 1 body" in text
    assert "page 3 body" in text
    assert "페이지 2 OCR 실패" in text
    assert result.pages_total == 3
    assert result.pages_failed == 1
