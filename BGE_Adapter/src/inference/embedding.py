"""Embedding generation helpers for inference."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np

from .model import Encoder
from .utils import resolve_numpy_dtype


def generate_query_embeddings(
    encoder: Encoder,
    query_1: str,
    query_2: str,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate embeddings for two queries in one model call."""
    batch_size = int(config.get("batch_size", 32))
    max_length = int(config.get("max_length", 512))
    normalize_embeddings = bool(config.get("normalize_embeddings", True))
    embedding_precision = config.get("embedding_precision", "float32")

    start_time = time.perf_counter()
    embeddings = encoder.encode(
        sentences=[query_1, query_2],
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=normalize_embeddings,
    )
    elapsed = time.perf_counter() - start_time

    dtype = resolve_numpy_dtype(embedding_precision)
    embeddings = embeddings.astype(dtype, copy=False)

    return embeddings[0], embeddings[1], elapsed


def embedding_statistics(vector: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics for an embedding vector."""
    vector_32 = vector.astype(np.float32, copy=False)
    return {
        "dimension": int(vector_32.shape[0]),
        "l2_norm": float(np.linalg.norm(vector_32)),
        "mean": float(np.mean(vector_32)),
        "std": float(np.std(vector_32)),
        "min": float(np.min(vector_32)),
        "max": float(np.max(vector_32)),
    }