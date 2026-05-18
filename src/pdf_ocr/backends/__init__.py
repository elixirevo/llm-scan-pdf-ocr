"""Pluggable backends that turn a PDF into a markdown file + image assets.

Currently shipped:
  - ``vlm``    : local Qwen3-VL via OpenAI-compatible server (per-page VLM grounding).
  - ``mineru`` : OpenDataLab MinerU (layout model + OCR; best CJK figure accuracy).
"""

from __future__ import annotations

from .base import Backend, BackendResult

__all__ = ["Backend", "BackendResult", "get_backend"]


def get_backend(name: str) -> type[Backend]:
    """Return the backend class for ``name``. Imports lazily so MinerU's heavy
    dependencies aren't required when the user only uses the VLM backend.
    """
    name = name.lower()
    if name == "vlm":
        from .vlm import VLMBackend
        return VLMBackend
    if name == "mineru":
        from .mineru import MinerUBackend
        return MinerUBackend
    raise ValueError(f"Unknown backend: {name!r} (choose 'vlm' or 'mineru')")
