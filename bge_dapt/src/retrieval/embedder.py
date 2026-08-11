"""Query embedding generation for interactive retrieval."""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from ..model_loader import DenseEncoder


def encode_query(encoder: DenseEncoder, query: str, config: Dict[str, Any]) -> tuple[np.ndarray, float]:
    """Encode a single query and return the vector and elapsed time.

    Args:
        encoder: A DenseEncoder instance.
        query: Query text.
        config: Configuration dictionary.

    Returns:
        Tuple of (query vector, elapsed time in seconds).
    """
    batch_size = int(config.get("batch_size", 32))
    max_length = int(config.get("max_length", 256))
    normalize = bool(config.get("normalize_embeddings", True))

    start = time.perf_counter()
    vectors = encoder.encode(
        sentences=[query],
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=normalize,
    )
    elapsed = time.perf_counter() - start

    return vectors[0], elapsed
