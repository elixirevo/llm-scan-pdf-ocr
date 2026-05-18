"""YAML config loader with ${VAR:-default} env interpolation and ``extends:`` support."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2) or ""
            return os.environ.get(name, default)
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str | Path | None) -> dict:
    """Load a YAML config, resolving ``extends:`` chains and env interpolation.

    ``path=None`` returns the bundled ``configs/default.yaml``.
    """
    if path is None:
        path = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    extends = data.pop("extends", None)
    if extends:
        parent_path = (path.parent / extends).resolve()
        parent = load_config(parent_path)
        data = _deep_merge(parent, data)

    return _interpolate(data)
