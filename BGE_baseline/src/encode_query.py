"""
Query embedding generator.

Encodes all queries using the BGE-M3 model and saves the dense
embeddings to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .model_loader import Encoder


def encode_queries(
    queries: List[Dict[str, str]],
    encoder: Encoder,
    config: Dict[str, Any],
) -> np.ndarray:
    """Generate dense embeddings for all queries.

    Args:
        queries: List of {"query_id": str, "query": str} dictionaries.
        encoder: An Encoder instance with an .encode() method.
        config: Configuration dictionary with embedding parameters.

    Returns:
        np.ndarray of shape (num_queries, embedding_dim).
    """
    embeddings_dir = Path(config.get("embeddings_dir", "embeddings"))
    embeddings_path = embeddings_dir / "query.npy"
    ids_path = embeddings_dir / "query_ids.json"

    # Check for cached embeddings
    if embeddings_path.exists() and ids_path.exists():
        print(f"[INFO] Query embeddings already exist at {embeddings_path}, loading from disk.")
        return np.load(str(embeddings_path))

    texts = [q["query"] for q in queries]
    query_ids = [q["query_id"] for q in queries]

    batch_size = config.get("batch_size", 32)
    max_length = config.get("max_length", 512)
    normalize = config.get("normalize_embeddings", True)

    print(f"[INFO] Encoding {len(texts)} queries...")
    embeddings = encoder.encode(
        sentences=texts,
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=normalize,
    )

    # Save embeddings and query IDs
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(embeddings_path), embeddings)
    print(f"[INFO] Saved query embeddings to {embeddings_path} (shape={embeddings.shape})")

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(query_ids, f, ensure_ascii=False)
    print(f"[INFO] Saved query IDs to {ids_path}")

    return embeddings
