"""Config loader: env interpolation + extends merging."""

from __future__ import annotations

from pathlib import Path

from pdf_ocr.utils.config import load_config


def test_default_config_loads_without_env(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    cfg = load_config(None)
    assert cfg["llm"]["base_url"].endswith("/v1")
    assert cfg["llm"]["model"]  # has a default


def test_env_interpolation(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "fake/model")
    cfg = load_config(None)
    assert cfg["llm"]["model"] == "fake/model"


def test_extends_merging():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "llamacpp.yaml")
    assert "8080" in cfg["llm"]["base_url"]
    # inherited from default
    assert "dpi" in cfg["render"]
    # overridden
    assert cfg["render"]["max_side"] == 1536
