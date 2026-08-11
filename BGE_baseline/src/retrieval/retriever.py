"""High-level retrieval operations: encoding a query and searching the FAISS index."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from ..model_loader import Encoder, load_model


def encode_query(
    encoder: Encoder, query: str, config: Dict[str, Any]
) -> tuple[np.ndarray, float]:
    """Encode a single query and return the vector and elapsed time."""
    batch_size = int(config.get("batch_size", 32))
    max_length = int(config.get("max_length", 512))
    normalize = bool(config.get("normalize_embeddings", True))

    start = time.perf_counter()
    vecs = encoder.encode(
        sentences=[query],
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=normalize,
    )
    elapsed = time.perf_counter() - start

    return vecs[0], elapsed


def search_index(
    index, query_vector: np.ndarray, top_k: int = 5
) -> tuple[List[int], List[float]]:
    """Search FAISS index and return indices and scores."""
    import numpy as _np

    q = query_vector.astype(_np.float32, copy=False).reshape(1, -1)
    scores, indices = index.search(q, top_k)
    return indices[0].tolist(), scores[0].tolist()