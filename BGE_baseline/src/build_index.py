"""
FAISS index builder.

Creates a FAISS index from corpus embeddings and saves it to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import faiss
import numpy as np


def build_index(
    corpus_embeddings: np.ndarray,
    config: Dict[str, Any],
) -> faiss.Index:
    """Build a FAISS index from corpus embeddings.

    Uses IndexFlatIP (inner product) since BGE-M3 embeddings are
    L2-normalized, making inner product equivalent to cosine similarity.

    Args:
        corpus_embeddings: np.ndarray of shape (num_docs, embedding_dim).
        config: Configuration dictionary with 'faiss_dir' key.

    Returns:
        A FAISS index containing all corpus vectors.
    """
    faiss_dir = Path(config.get("faiss_dir", "faiss"))
    index_path = faiss_dir / "index.bin"

    # Check for cached index
    if index_path.exists():
        print(f"[INFO] FAISS index already exists at {index_path}, loading from disk.")
        return faiss.read_index(str(index_path))

    embedding_dim = corpus_embeddings.shape[1]
    num_vectors = corpus_embeddings.shape[0]

    print(f"[INFO] Building FAISS IndexFlatIP (dim={embedding_dim}, vectors={num_vectors})...")

    # Inner product index (equivalent to cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(embedding_dim)

    # Ensure embeddings are float32 (FAISS requirement)
    if corpus_embeddings.dtype != np.float32:
        corpus_embeddings = corpus_embeddings.astype(np.float32)

    index.add(corpus_embeddings)
    print(f"[INFO] FAISS index built with {index.ntotal} vectors.")

    # Save index
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"[INFO] Saved FAISS index to {index_path}")

    return index
