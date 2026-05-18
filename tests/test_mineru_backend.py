"""MinerU backend tests.

We don't actually run MinerU here — we monkeypatch the subprocess call and
verify that:
  * the generated <stem>.md is moved into the project's standard out_dir,
  * the images directory is renamed to <stem>_images,
  * image links inside the markdown are rewritten to point at the new dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_ocr.backends import mineru as mineru_mod
from pdf_ocr.backends.mineru import MinerUBackend, _rewrite_image_links


def test_rewrite_image_links_basic():
    md = "see ![fig](images/a.jpg) and ![](images/b.jpg \"t\")"
    out = _rewrite_image_links(md, "doc_images")
    assert "![fig](doc_images/a.jpg)" in out
    assert '![](doc_images/b.jpg "t")' in out
    # No bare ``(images/...`` link should survive — but ``doc_images/`` contains
    # the substring ``images/``, so check for the link prefix specifically.
    assert "(images/" not in out


def test_rewrite_image_links_leaves_external_urls_alone():
    md = "![x](https://example.com/foo.png) and ![y](images/local.jpg)"
    out = _rewrite_image_links(md, "doc_images")
    assert "https://example.com/foo.png" in out
    assert "doc_images/local.jpg" in out


@pytest.mark.asyncio
async def test_mineru_backend_collects_output(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    out_dir = tmp_path / "out"

    # Pretend `mineru` exists on PATH.
    monkeypatch.setattr(mineru_mod, "_which_mineru", lambda explicit: "/usr/bin/mineru")

    async def fake_run_cli(cmd):
        # Recover the staging directory MinerU was told to write to.
        staging = Path(cmd[cmd.index("-o") + 1])
        produced = staging / "doc" / "auto"
        (produced / "images").mkdir(parents=True)
        (produced / "images" / "fig1.jpg").write_bytes(b"\x89PNG fake")
        (produced / "doc.md").write_text(
            "# doc\n\nbody text\n\n![](images/fig1.jpg)\n",
            encoding="utf-8",
        )
        return 0, "", ""

    monkeypatch.setattr(mineru_mod, "_run_cli", fake_run_cli)

    cfg = {"mineru": {"backend": "pipeline", "lang": "korean", "method": "auto"}}
    result = await MinerUBackend().run(pdf, out_dir, cfg)

    assert result.markdown_path == out_dir / "doc.md"
    assert result.images_dir == out_dir / "doc_images"
    assert (out_dir / "doc_images" / "fig1.jpg").exists()

    md = result.markdown_path.read_text(encoding="utf-8")
    assert "![](doc_images/fig1.jpg)" in md
    assert "(images/fig1.jpg)" not in md   # raw mineru link must be gone

    # Staging dir should be cleaned up.
    assert not (out_dir / ".mineru_staging").exists()


@pytest.mark.asyncio
async def test_mineru_backend_surfaces_cli_error(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(mineru_mod, "_which_mineru", lambda explicit: "/usr/bin/mineru")

    async def failing(cmd):
        return 1, "", "boom: model file missing"

    monkeypatch.setattr(mineru_mod, "_run_cli", failing)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        await MinerUBackend().run(pdf, tmp_path / "out", {})
