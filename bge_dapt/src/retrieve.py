"""
Dense retrieval module for Stage 2 evaluation.

Performs top-K retrieval using a FAISS index and query embeddings, saving the
ranked results to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from tqdm import tqdm


def retrieve(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    query_ids: List[str],
    corpus_ids: List[str],
    config: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Retrieve top-K documents for each query.

    Args:
        index: A FAISS index containing corpus vectors.
        query_embeddings: np.ndarray of shape (num_queries, embedding_dim).
        query_ids: List of query IDs aligned with query_embeddings rows.
        corpus_ids: List of document IDs aligned with FAISS index positions.
        config: Configuration dictionary with 'top_k' and 'evaluation_dir'.

    Returns:
        Dictionary mapping query_id to a ranked list of retrieved doc_ids.
    """
    top_k = int(config.get("top_k", 10))
    eval_dir = Path(config.get("evaluation_dir", "outputs"))

    if query_embeddings.dtype != np.float32:
        query_embeddings = query_embeddings.astype(np.float32)

    print(f"[INFO] Retrieving top-{top_k} documents for {len(query_ids)} queries...")

    scores, indices = index.search(query_embeddings, top_k)

    results: Dict[str, List[str]] = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Building results")):
        retrieved_docs: List[str] = []
        for j in range(top_k):
            idx = indices[i][j]
            if 0 <= idx < len(corpus_ids):
                retrieved_docs.append(corpus_ids[idx])
        results[qid] = retrieved_docs

    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "retrieval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved retrieval results to {results_path}")

    return results
