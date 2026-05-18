"""MinerU backend: shell out to the ``mineru`` CLI and normalize its output.

We prefer the CLI over MinerU's Python API because the API signature
(``do_parse`` / ``aio_do_parse``) churns between releases. The CLI contract
(``mineru -p <pdf> -o <dir> -b <backend> -l <lang>``) is more stable.

MinerU produces a directory tree like::

    <out_dir>/<pdf_stem>/auto/
        ├── <pdf_stem>.md
        ├── images/
        │   ├── abc123...jpg
        │   └── ...
        ├── layout.pdf
        ├── middle.json
        └── ...

We move the markdown + images into the layout this project uses elsewhere::

    <out_dir>/<pdf_stem>.md
    <out_dir>/<pdf_stem>_images/<original-image-names>

and rewrite the image links in the markdown accordingly.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .base import BackendResult

log = logging.getLogger(__name__)


class MinerUNotInstalledError(RuntimeError):
    pass


def _which_mineru(explicit: str | None) -> str:
    binary = explicit or shutil.which("mineru")
    if not binary:
        raise MinerUNotInstalledError(
            "The `mineru` CLI was not found on PATH.\n"
            "Install it with:  uv pip install --extra-index-url "
            "https://wheels.myhloli.com 'mineru[core]'\n"
            "or follow https://opendatalab.github.io/MinerU/quick_start/"
        )
    return binary


# Image links in MinerU markdown look like:
#   ![](images/abc.jpg)            (relative)
#   ![alt](images/abc.jpg "title") (with alt/title)
_IMG_LINK = re.compile(r"!\[([^\]]*)\]\((images/[^)\s]+)(\s+\"[^\"]*\")?\)")


def _rewrite_image_links(md_text: str, new_dir_name: str) -> str:
    """Replace ``images/...`` with ``<new_dir_name>/...`` everywhere in the markdown."""
    def repl(m: re.Match[str]) -> str:
        alt, path, title = m.group(1), m.group(2), m.group(3) or ""
        new_path = path.replace("images/", f"{new_dir_name}/", 1)
        return f"![{alt}]({new_path}{title})"
    return _IMG_LINK.sub(repl, md_text)


async def _run_cli(cmd: list[str]) -> tuple[int, str, str]:
    log.debug("running: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return proc.returncode or 0, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")


class MinerUBackend:
    """Shell out to MinerU and normalize the output layout."""

    name = "mineru"

    async def run(self, pdf_path: Path, out_dir: Path, cfg: dict) -> BackendResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mcfg = cfg.get("mineru", {})
        binary = _which_mineru(mcfg.get("binary"))
        backend = mcfg.get("backend", "pipeline")            # pipeline | vlm-transformers | vlm-vllm-async-engine | ...
        lang = mcfg.get("lang", "korean")
        method = mcfg.get("method", "auto")                  # auto | txt | ocr  (pipeline backend only)
        extra_args: list[str] = list(mcfg.get("extra_args", []))

        # MinerU writes into a temp staging dir we control, then we hoist out
        # only the bits we care about. This keeps the user-facing out_dir clean.
        staging = out_dir / ".mineru_staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        cmd = [
            binary,
            "-p", str(pdf_path),
            "-o", str(staging),
            "-b", backend,
            "-l", lang,
        ]
        # `method` is only meaningful for the pipeline backend.
        if backend == "pipeline":
            cmd += ["-m", method]
        cmd += extra_args

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task(f"MinerU ({backend})", total=None)
            code, stdout, stderr = await _run_cli(cmd)

        if code != 0:
            tail = (stderr or stdout).strip().splitlines()[-20:]
            raise RuntimeError(
                f"mineru exited with code {code}. Last lines:\n  " + "\n  ".join(tail)
            )

        return self._collect(pdf_path, out_dir, staging, backend)

    # ------------------------------------------------------------------
    # output collection
    # ------------------------------------------------------------------

    def _collect(
        self, pdf_path: Path, out_dir: Path, staging: Path, backend: str
    ) -> BackendResult:
        stem = pdf_path.stem
        # MinerU's "method" subdir is "auto"/"txt"/"ocr" for pipeline, and "vlm"
        # for VLM backends. We don't assume — just find the only .md.
        produced_md = list(staging.rglob(f"{stem}.md"))
        if not produced_md:
            raise RuntimeError(
                f"MinerU ran but produced no '{stem}.md' under {staging}"
            )
        src_md = produced_md[0]
        src_images = src_md.parent / "images"

        # Final destinations: same layout as the VLM backend uses.
        dst_md = out_dir / f"{stem}.md"
        dst_images_dir_name = f"{stem}_images"
        dst_images = out_dir / dst_images_dir_name

        if dst_images.exists():
            shutil.rmtree(dst_images)
        if src_images.exists():
            shutil.copytree(src_images, dst_images)
        else:
            dst_images.mkdir(exist_ok=True)

        md_text = src_md.read_text(encoding="utf-8")
        md_text = _rewrite_image_links(md_text, dst_images_dir_name)
        dst_md.write_text(md_text, encoding="utf-8")

        # Clean the staging dir; keep nothing the user didn't ask for.
        shutil.rmtree(staging, ignore_errors=True)

        # MinerU doesn't expose a clean "pages" count without parsing middle.json;
        # we approximate with the number of HR separators it emits (one per page).
        pages = max(1, md_text.count("\n---\n") + 1)
        log.info("MinerU (%s) wrote %s (%d pages approx)", backend, dst_md, pages)

        return BackendResult(
            markdown_path=dst_md,
            images_dir=dst_images,
            pages_total=pages,
            pages_failed=0,
        )
