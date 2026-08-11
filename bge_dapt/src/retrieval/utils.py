"""Shared utilities for the interactive retrieval module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml


def setup_logging() -> logging.Logger:
    """Configure and return the module logger."""
    logger = logging.getLogger("src.retrieval")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def format_and_print(
    model_name: str,
    query: str,
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    timings: Dict[str, float],
) -> None:
    """Print a human-friendly retrieval report."""
    top_k = config.get("top_k", 5)
    show_score = bool(config.get("show_similarity_score", True))
    show_text = bool(config.get("show_document_text", True))
    show_title = bool(config.get("show_document_title", True))

    print("\n" + "=" * 55)
    print("Semantic Retrieval Result")
    print("=" * 55)
    print("\nModel\n")
    print(model_name)
    print("\n" + "-" * 55)
    print("\nQuery\n")
    print(query)
    print("\n" + "-" * 55)
    print(f"Top {top_k} Documents\n")

    for r in results:
        print("-" * 55)
        print(f"Rank {r['rank']}")
        if show_score:
            print(f"\nSimilarity : {r['score']:.4f}")
        print(f"\nDocument ID : {r['document_id']}")
        if show_title:
            print(f"\nTitle : {r.get('title', '')}")
        if show_text:
            print("\nContent\n")
            print(r.get("text", ""))
        print()

    print("-" * 55)
    print("\nEmbedding Time\n")
    print(f"{timings['embedding_time']:.4f} sec")
    print("\nSearch Time\n")
    print(f"{timings['search_time']:.4f} sec")
    print("\nTotal Time\n")
    print(f"{timings['total_time']:.4f} sec")
    print("\nRetrieved Documents\n")
    print(len(results))
    print("\n" + "=" * 55 + "\n")


def save_results_json(path: Path, payload: Dict[str, Any]) -> None:
    """Save retrieval results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
