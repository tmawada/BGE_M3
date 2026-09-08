"""Encode corpus / queries to .npy files (with disk cache)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .model import Encoder


def encoder_tag(config: Dict[str, Any]) -> str:
    """Short filesystem-safe tag for the current encoder.

    Prevents the cross-encoder bug where adapter queries reused baseline
    corpus vectors. Example: 'BAAI_bge-m3', 'models_bge-m3-dapt',
    'BAAI_bge-m3__lora'.
    """
    base = str(config.get("model_name", "BAAI/bge-m3"))
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "model"
    if config.get("use_adapter", False):
        tag += "__lora"
    return tag


def _out_dir(config: Dict[str, Any]) -> Path:
    """Resolve embeddings_dir relative to repo root."""
    p = Path(config.get("embeddings_dir", "embeddings"))
    if not p.is_absolute():  # keep outputs inside BGE_UDA/ by default
        p = Path(__file__).resolve().parent.parent / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def encode_corpus(corpus: List[Dict[str, str]], encoder: Encoder, config: Dict[str, Any],
                  force: bool = False) -> np.ndarray:
    """Encode doc passages -> embeddings/corpus_<tag>.npy + corpus_<tag>_ids.json."""
    out_dir = _out_dir(config)
    tag = encoder_tag(config)
    npy, ids = out_dir / f"corpus_{tag}.npy", out_dir / f"corpus_{tag}_ids.json"
    if not force and npy.exists() and ids.exists():  # cache per encoder, not global
        print(f"[INFO] Reusing cached {npy} (tag={tag})")
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
                   config: Dict[str, Any], name: str = "query",
                   force: bool = False) -> tuple[np.ndarray, List[str]]:
    """Encode queries -> embeddings/<name>_<tag>.npy. name='query' or 'informal_query'."""
    out_dir = _out_dir(config)
    tag = encoder_tag(config)
    npy, ids = out_dir / f"{name}_{tag}.npy", out_dir / f"{name}_{tag}_ids.json"
    if not force and npy.exists() and ids.exists():
        print(f"[INFO] Reusing cached {npy} (tag={tag})")
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
