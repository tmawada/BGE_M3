"""Load the single YAML config file.

Everything editable lives in config/config.yaml so code never hard-codes paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Read YAML config into a dict."""
    # Resolve path relative to repo root so CLI works from anywhere.
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        # pipeline.py lives in src/, so repo root is one level up.
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / cfg_path
        if candidate.exists():
            cfg_path = candidate
    # Fail fast with a clear message if config is missing.
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    # safe_load avoids executing arbitrary YAML tags.
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Loaded config from {cfg_path}")
    return config
