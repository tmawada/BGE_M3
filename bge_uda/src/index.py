"""FAISS index: exact inner-product search (== cosine on normalized vecs)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import faiss
import numpy as np


def _index_path(config: Dict[str, Any]) -> Path:
    """Resolve faiss/index.bin relative to repo root."""
    d = Path(config.get("faiss_dir", "faiss"))
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / d
    d.mkdir(parents=True, exist_ok=True)
    return d / "index.bin"


def build_index(corpus_vecs: np.ndarray, config: Dict[str, Any]) -> faiss.Index:
    """Build IndexFlatIP from corpus vectors, or load cache if exists."""
    path = _index_path(config)
    if path.exists():  # reuse: building is cheap but loading is clearer
        print(f"[INFO] Loading FAISS index: {path}")
        return faiss.read_index(str(path))
    dim = corpus_vecs.shape[1]  # 1024 for bge-m3
    print(f"[INFO] Building IndexFlatIP dim={dim} n={len(corpus_vecs)}...")
    index = faiss.IndexFlatIP(dim)  # exact brute-force, no training needed
    if corpus_vecs.dtype != np.float32:  # FAISS requires float32
        corpus_vecs = corpus_vecs.astype(np.float32)
    index.add(corpus_vecs)  # position i <-> corpus_ids[i]
    faiss.write_index(index, str(path))
    print(f"[INFO] Saved index ({index.ntotal} vecs) -> {path}")
    return index


def load_index(config: Dict[str, Any]) -> faiss.Index:
    """Load existing index; error clearly if baseline hasn't run yet."""
    path = _index_path(config)
    if not path.exists():
        raise FileNotFoundError(f"No index at {path}. Run --stage baseline first.")
    return faiss.read_index(str(path))
