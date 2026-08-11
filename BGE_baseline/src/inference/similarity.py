"""Similarity metrics for embedding comparison."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def _as_float32(vector: np.ndarray) -> np.ndarray:
    return vector.astype(np.float32, copy=False).reshape(1, -1)


def cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """Compute cosine similarity in the range 0.0 to 1.0."""
    return float(sklearn_cosine_similarity(_as_float32(vector_1), _as_float32(vector_2))[0, 0])


def euclidean_distance(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    diff = vector_1.astype(np.float32, copy=False) - vector_2.astype(np.float32, copy=False)
    return float(np.linalg.norm(diff))


def dot_product(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """Compute the dot product between two vectors."""
    return float(np.dot(vector_1.astype(np.float32, copy=False), vector_2.astype(np.float32, copy=False)))