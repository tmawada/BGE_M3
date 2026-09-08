"""Encode corpus / queries to .npy files (with disk cache)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .model import Encoder


def _out_dir(config: Dict[str, Any]) -> Path:
    """Resolve embeddings_dir relative to repo root."""
    p = Path(config.get("embeddings_dir", "embeddings"))
    if not p.is_absolute():  # keep outputs inside BGE_UDA/ by default
        p = Path(__file__).resolve().parent.parent / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def encode_corpus(corpus: List[Dict[str, str]], encoder: Encoder, config: Dict[str, Any]) -> np.ndarray:
    """Encode doc passages -> embeddings/corpus.npy + corpus_ids.json."""
    out_dir = _out_dir(config)
    npy, ids = out_dir / "corpus.npy", out_dir / "corpus_ids.json"
    if npy.exists() and ids.exists():  # cache: skip re-encoding (slow on GPU)
        print(f"[INFO] Reusing cached {npy}")
        return np.load(str(npy))
    texts = [d["passage"] for d in corpus]  # title NOT encoded, display only
    doc_ids = [d["doc_id"] for d in corpus]  # order must match FAISS positions
    print(f"[INFO] Encoding {len(texts)} docs...")
    vecs = encoder.encode(texts, config.get("batch_size", 32),
                          config.get("max_length", 512), config.get("normalize_embeddings", True))
    np.save(str(npy), vecs)  # .npy preserves shape + dtype
    ids.write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved {vecs.shape} -> {npy}")
    return vecs


def encode_queries(queries: List[Dict[str, str]], encoder: Encoder,
                   config: Dict[str, Any], name: str = "query") -> tuple[np.ndarray, List[str]]:
    """Encode queries -> embeddings/<name>.npy. name='query' or 'informal_query'."""
    out_dir = _out_dir(config)
    npy, ids = out_dir / f"{name}.npy", out_dir / f"{name}_ids.json"
    if npy.exists() and ids.exists():
        print(f"[INFO] Reusing cached {npy}")
        return np.load(str(npy)), json.loads(ids.read_text(encoding="utf-8"))
    texts = [q["query"] for q in queries]
    qids = [q["query_id"] for q in queries]
    print(f"[INFO] Encoding {len(texts)} queries ({name})...")
    vecs = encoder.encode(texts, config.get("batch_size", 32),
                          config.get("max_length", 512), config.get("normalize_embeddings", True))
    np.save(str(npy), vecs)
    ids.write_text(json.dumps(qids, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved {vecs.shape} -> {npy}")
    return vecs, qids
