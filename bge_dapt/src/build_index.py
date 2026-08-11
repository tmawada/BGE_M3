"""
FAISS index builder for Stage 2 evaluation.

Creates a flat inner-product index over L2-normalized corpus embeddings and
saves it to disk (cached for reuse across evaluations).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import faiss
import numpy as np


def build_index(
    corpus_embeddings: np.ndarray,
    config: Dict[str, Any],
    force: bool = False,
) -> faiss.Index:
    """Build a FAISS IndexFlatIP from corpus embeddings.

    Inner product is equivalent to cosine similarity for L2-normalized
    embeddings produced by the retriever.

    Args:
        corpus_embeddings: np.ndarray of shape (num_docs, embedding_dim).
        config: Configuration dictionary with 'faiss_dir' key.
        force: Rebuild the index even if it already exists.

    Returns:
        A FAISS index containing all corpus vectors.
    """
    faiss_dir = Path(config.get("faiss_dir", "outputs/faiss"))
    index_path = faiss_dir / "index.bin"

    if not force and index_path.exists():
        print(f"[INFO] FAISS index already exists at {index_path}, loading from disk.")
        return faiss.read_index(str(index_path))

    embedding_dim = corpus_embeddings.shape[1]
    num_vectors = corpus_embeddings.shape[0]

    print(f"[INFO] Building FAISS IndexFlatIP (dim={embedding_dim}, vectors={num_vectors})...")

    index = faiss.IndexFlatIP(embedding_dim)

    if corpus_embeddings.dtype != np.float32:
        corpus_embeddings = corpus_embeddings.astype(np.float32)

    index.add(corpus_embeddings)
    print(f"[INFO] FAISS index built with {index.ntotal} vectors.")

    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"[INFO] Saved FAISS index to {index_path}")

    return index
