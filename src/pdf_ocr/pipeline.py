"""Backend dispatcher.

The interesting per-backend logic lives in :mod:`pdf_ocr.backends`. This module
just picks one and runs it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .backends import BackendResult, get_backend

log = logging.getLogger(__name__)


async def run_pipeline(pdf_path: Path, out_dir: Path, cfg: dict) -> Path:
    """Process ``pdf_path`` end-to-end and return the path of the written .md.

    Selects the backend from ``cfg['backend']`` (defaults to 'vlm').
    """
    name = (cfg.get("backend") or "vlm").lower()
    cls = get_backend(name)
    backend = cls()
    result: BackendResult = await backend.run(Path(pdf_path), Path(out_dir), cfg)
    return result.markdown_path
