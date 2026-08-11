"""Inference utilities for BGE-M3 similarity testing."""

from .embedding import embedding_statistics, generate_query_embeddings
from .model import load_model
from .similarity import cosine_similarity, dot_product, euclidean_distance

__all__ = [
    "generate_query_embeddings",
    "embedding_statistics",
    "load_model",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
]