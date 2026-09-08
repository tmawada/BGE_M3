"""FAISS index: exact inner-product search (== cosine on normalized vecs)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def _faiss():
    """Lazy faiss import (keeps --help / numpy-only evals working w/o faiss)."""
    try:
        import faiss
        return faiss
    except Exception as e:
        raise ImportError(
            f"faiss import failed ({e}). Fix env mismatch (e.g. numpy>=2 vs "
            f"faiss-cpu built for numpy1: pip install 'numpy<2' or upgrade faiss).")


def _index_path(config: Dict[str, Any]) -> Path:
    """Resolve faiss/index_<tag>.bin relative to repo root (per-encoder)."""
    from .encode import encoder_tag
    d = Path(config.get("faiss_dir", "faiss"))
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / d
    d.mkdir(parents=True, exist_ok=True)
    try:
        tag = encoder_tag(config)
    except Exception:
        tag = "model"
    return d / f"index_{tag}.bin"


def build_index(corpus_vecs: np.ndarray, config: Dict[str, Any],
                force: bool = False) -> faiss.Index:
    """Build IndexFlatIP from corpus vectors (per-encoder cache)."""
    path = _index_path(config)
    faiss = _faiss()
    if not force and path.exists():
        # Guard against stale cross-encoder reuse: dim/nvec must match.
        try:
            cached = faiss.read_index(str(path))
            if cached.d == int(corpus_vecs.shape[1]) and cached.ntotal == len(corpus_vecs):
                print(f"[INFO] Loading FAISS index: {path}")
                return cached
            print(f"[WARNING] Cached index dim/ntotal mismatch "
                  f"(cached d={cached.d},n={cached.ntotal} vs vecs {corpus_vecs.shape}) -> rebuild.")
        except Exception as e:  # corrupt cache -> rebuild
            print(f"[WARNING] Could not load {path} ({e}) -> rebuild.")
    dim = corpus_vecs.shape[1]  # 1024 for bge-m3
    print(f"[INFO] Building IndexFlatIP dim={dim} n={len(corpus_vecs)}...")
    index = faiss.IndexFlatIP(int(dim))  # exact brute-force, no training needed
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
