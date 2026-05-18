"""Orchestrate PDF → Markdown with bounded concurrency."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .llm.openai_compat import VLMClient, VLMConfig
from .ocr.crop import crop_assets
from .ocr.extractor import extract_page
from .pdf.page import Page
from .pdf.render import render_pdf
from .render.markdown import PageResult, write_document

log = logging.getLogger(__name__)


def _build_paths(pdf_path: Path, out_dir: Path, output_cfg: dict) -> tuple[Path, Path]:
    stem = pdf_path.stem
    md_path = out_dir / output_cfg["markdown_path"].replace("${stem}", stem)
    images_dir = out_dir / output_cfg["images_dir"].replace("${stem}", stem)
    return md_path, images_dir


async def _process_one(
    page: Page,
    *,
    client: VLMClient,
    images_dir: Path,
    asset_name: str,
    sem: asyncio.Semaphore,
) -> PageResult:
    async with sem:
        layout = await extract_page(client, page)
    # Cropping is CPU-bound and fast; do it outside the semaphore.
    assets = crop_assets(page, layout, images_dir=images_dir, asset_name=asset_name)
    log.info(
        "page %d done: %d blocks, %d assets",
        page.page_no, len(layout.blocks), len(assets),
    )
    return PageResult(layout=layout, assets=assets)


async def run_pipeline(
    pdf_path: Path,
    out_dir: Path,
    cfg: dict,
    *,
    pages: Iterable[Page] | None = None,
) -> Path:
    """Process ``pdf_path`` end-to-end and return the path of the written .md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path, images_dir = _build_paths(pdf_path, out_dir, cfg["output"])
    asset_name = cfg["output"]["asset_name"]

    if pages is None:
        pages = render_pdf(
            pdf_path,
            dpi=cfg["render"]["dpi"],
            max_side=cfg["render"].get("max_side"),
        )

    vlm_cfg = cfg["llm"]
    client = VLMClient(VLMConfig(
        base_url=vlm_cfg["base_url"],
        api_key=vlm_cfg["api_key"],
        model=vlm_cfg["model"],
        use_response_format=vlm_cfg.get("use_response_format", True),
        timeout=float(vlm_cfg.get("timeout", 180)),
        max_retries=int(vlm_cfg.get("max_retries", 2)),
        max_tokens=int(vlm_cfg.get("max_tokens", 2048)),
    ))

    sem = asyncio.Semaphore(int(cfg["pipeline"]["concurrency"]))

    # Materialize pages lazily into tasks as they're rendered so we don't hold
    # the whole document in memory at once. asyncio.gather preserves order.
    tasks: list[asyncio.Task[PageResult]] = []
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]
    try:
        with Progress(*progress_columns) as progress:
            scan_task = progress.add_task("Rendering & extracting", total=None)
            for page in pages:
                tasks.append(asyncio.create_task(
                    _process_one(
                        page,
                        client=client,
                        images_dir=images_dir,
                        asset_name=asset_name,
                        sem=sem,
                    )
                ))
                progress.update(scan_task, total=len(tasks))
            # Wait for all in submission order.
            results: list[PageResult] = []
            for t in tasks:
                results.append(await t)
                progress.update(scan_task, advance=1)
    finally:
        await client.aclose()

    results.sort(key=lambda r: r.layout.page_no)
    write_document(results, md_path, title=pdf_path.stem)
    log.info("wrote %s (%d pages)", md_path, len(results))
    return md_path
