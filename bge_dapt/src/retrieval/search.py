"""Low-level FAISS search wrapper for interactive retrieval."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def search_index(
    index,
    query_vector: np.ndarray,
    top_k: int = 5,
) -> Tuple[List[int], List[float]]:
    """Search a FAISS index and return indices and scores.

    Args:
        index: A FAISS index.
        query_vector: 1-D query embedding.
        top_k: Number of results to return.

    Returns:
        Tuple of (list of corpus indices, list of similarity scores).
    """
    query = query_vector.astype(np.float32, copy=False).reshape(1, -1)
    scores, indices = index.search(query, top_k)
    return indices[0].tolist(), scores[0].tolist()
