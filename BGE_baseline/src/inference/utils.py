"""Utility helpers for the inference CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


LOGGER_NAME = "src.inference"


def setup_logging() -> logging.Logger:
    """Configure a simple logger for inference runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration for inference."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    return config


def resolve_output_dir(config: Dict[str, Any]) -> Path:
    """Resolve the output directory from configuration."""
    return Path(config.get("outputs_dir", config.get("output_dir", "outputs")))


def resolve_numpy_dtype(precision: Any) -> np.dtype:
    """Map a configured precision value to a NumPy dtype."""
    precision_str = str(precision).lower().strip()
    if precision_str in {"float16", "fp16", "half"}:
        return np.float16
    if precision_str in {"float32", "fp32", "single"}:
        return np.float32
    if precision_str in {"float64", "fp64", "double"}:
        return np.float64
    return np.float32


def format_vector_preview(vector: np.ndarray, limit: int = 10) -> str:
    """Return a compact preview string for a dense vector."""
    preview = vector[:limit]
    return np.array2string(preview, precision=4, separator=", ", suppress_small=False)


def save_numpy_vector(path: Path, vector: np.ndarray) -> None:
    """Persist a single vector to .npy format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), vector)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist a JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def print_report(payload: Dict[str, Any]) -> None:
    """Print a human-readable inference report."""
    stats_1 = payload["query_1_stats"]
    stats_2 = payload["query_2_stats"]

    print("\n" + "=" * 53)
    print("Embedding Similarity Analysis")
    print("=" * 53)
    print("\nModel\n")
    print(payload["model"])
    print("\n" + "-" * 53)
    print("\nQuery 1\n")
    print(payload["query_1"])
    print("\nEmbedding Dimension\n")
    print(int(stats_1["dimension"]))
    print("\nL2 Norm\n")
    print(f"{stats_1['l2_norm']:.4f}")
    print("\nEmbedding Preview\n")
    print(payload["query_1_preview"])
    print("\n" + "-" * 53)
    print("\nQuery 2\n")
    print(payload["query_2"])
    print("\nEmbedding Dimension\n")
    print(int(stats_2["dimension"]))
    print("\nL2 Norm\n")
    print(f"{stats_2['l2_norm']:.4f}")
    print("\nEmbedding Preview\n")
    print(payload["query_2_preview"])
    print("\n" + "-" * 53)
    print("\nCosine Similarity\n")
    print(f"{payload['cosine_similarity']:.4f}")
    print("\n" + "-" * 53)
    print("\nEuclidean Distance\n")
    print(f"{payload['euclidean_distance']:.4f}")
    print("\n" + "-" * 53)
    print("\nDot Product\n")
    print(f"{payload['dot_product']:.4f}")
    print("\n" + "-" * 53)
    print("\nDevice\n")
    print(payload["device"])
    print("\nInference Time\n")
    print(f"{payload['inference_time']:.4f} sec")
    print("\nEmbedding Time\n")
    print(f"{payload['embedding_time']:.4f} sec")
    print("\nSimilarity Time\n")
    print(f"{payload['similarity_time']:.4f} sec")
    print("\n" + "=" * 53 + "\n")