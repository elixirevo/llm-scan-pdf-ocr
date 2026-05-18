"""Backend protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class BackendResult:
    """The deliverables a backend produces.

    ``markdown_path`` is the final .md file the user opens.
    ``images_dir`` is where any extracted figure/table assets live (relative
    paths in the markdown refer to this directory).
    """

    markdown_path: Path
    images_dir: Path
    pages_total: int
    pages_failed: int = 0


@runtime_checkable
class Backend(Protocol):
    """All backends implement this single async method.

    Implementations are free to use whatever internal pipeline they want — the
    VLM backend renders pages and calls a remote model per page, MinerU shells
    out to its own CLI — as long as the contract holds:

    * Write a single markdown file at the path derived from ``cfg['output']``.
    * Write all referenced images inside the configured ``images_dir`` and use
      paths *relative to the markdown* in the markdown body.
    * Return a :class:`BackendResult` pointing at those paths.
    """

    name: str

    async def run(self, pdf_path: Path, out_dir: Path, cfg: dict) -> BackendResult: ...
