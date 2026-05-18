"""Typer-based CLI: ``pdf-ocr run <input.pdf> -o <out_dir>``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from .pipeline import run_pipeline
from .utils.config import load_config
from .utils.logging import setup_logging

app = typer.Typer(add_completion=False, help="Scanned PDF → Markdown via a local VLM.")


@app.command()
def run(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Input PDF."),
    out: Path = typer.Option(Path("output"), "-o", "--out", help="Output directory."),
    config: Optional[Path] = typer.Option(
        None, "-c", "--config", help="Config YAML. Defaults to configs/default.yaml."
    ),
    dpi: Optional[int] = typer.Option(None, "--dpi", help="Override render DPI."),
    concurrency: Optional[int] = typer.Option(
        None, "--concurrency", help="Override max in-flight VLM requests."
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging."),
) -> None:
    """OCR a scanned PDF into Markdown using a local VLM."""
    load_dotenv()
    setup_logging(verbose=verbose)
    cfg = load_config(config)

    if dpi is not None:
        cfg["render"]["dpi"] = dpi
    if concurrency is not None:
        cfg["pipeline"]["concurrency"] = concurrency

    md_path = asyncio.run(run_pipeline(pdf, out, cfg))
    typer.secho(f"✓ wrote {md_path}", fg=typer.colors.GREEN, bold=True)


@app.command()
def info() -> None:
    """Print the resolved default config (for debugging env interpolation)."""
    load_dotenv()
    import json

    cfg = load_config(None)
    typer.echo(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
