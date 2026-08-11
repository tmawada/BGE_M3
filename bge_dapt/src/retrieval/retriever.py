"""High-level retrieval orchestration for interactive search."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from ..model_loader import DenseEncoder, load_encoder
from .embedder import encode_query
from .faiss_index import load_index_and_corpus
from .search import search_index


def run_retrieval(config: Dict[str, Any], query: str, encoder: DenseEncoder) -> Dict[str, Any]:
    """Run one interactive retrieval query end-to-end.

    Args:
        config: Configuration dictionary.
        query: Query text (non-empty).
        encoder: A DenseEncoder instance (loaded once by the caller).

    Returns:
        Payload with model, query, ranked results, and timings.
    """
    index, corpus = load_index_and_corpus(config)
    top_k = int(config.get("top_k", 5))

    query_vector, embedding_time = encode_query(encoder, query, config)

    start = time.perf_counter()
    indices, scores = search_index(index, query_vector, top_k=top_k)
    search_time = time.perf_counter() - start

    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx < 0 or idx >= len(corpus):
            continue
        doc = corpus[idx]
        results.append(
            {
                "rank": rank,
                "document_id": doc.get("doc_id", ""),
                "score": float(score),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
        )

    total_time = embedding_time + search_time

    return {
        "model": config.get("retriever_model_dir", "models/bge-m3-dapt-retriever"),
        "query": query,
        "results": results,
        "timings": {
            "embedding_time": embedding_time,
            "search_time": search_time,
            "total_time": total_time,
        },
    }
