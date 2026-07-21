"""
Corpus embedding generator.

Encodes all documents in the corpus using the BGE-M3 model and saves
the dense embeddings to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .model_loader import Encoder


def encode_corpus(
    corpus: List[Dict[str, str]],
    encoder: Encoder,
    config: Dict[str, Any],
) -> np.ndarray:
    """Generate dense embeddings for all corpus documents.

    Args:
        corpus: List of {"doc_id": str, "text": str} dictionaries.
        encoder: An Encoder instance with an .encode() method.
        config: Configuration dictionary with embedding parameters.

    Returns:
        np.ndarray of shape (num_docs, embedding_dim).
    """
    embeddings_dir = Path(config.get("embeddings_dir", "embeddings"))
    embeddings_path = embeddings_dir / "corpus.npy"
    ids_path = embeddings_dir / "corpus_ids.json"

    # Check for cached embeddings
    if embeddings_path.exists() and ids_path.exists():
        print(f"[INFO] Corpus embeddings already exist at {embeddings_path}, loading from disk.")
        return np.load(str(embeddings_path))

    texts = [doc["text"] for doc in corpus]
    doc_ids = [doc["doc_id"] for doc in corpus]

    batch_size = config.get("batch_size", 32)
    max_length = config.get("max_length", 512)
    normalize = config.get("normalize_embeddings", True)

    print(f"[INFO] Encoding {len(texts)} corpus documents...")
    embeddings = encoder.encode(
        sentences=texts,
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=normalize,
    )

    # Save embeddings and document IDs
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(embeddings_path), embeddings)
    print(f"[INFO] Saved corpus embeddings to {embeddings_path} (shape={embeddings.shape})")

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    print(f"[INFO] Saved corpus IDs to {ids_path}")

    return embeddings
