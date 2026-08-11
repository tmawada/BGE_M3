"""Load the FAISS index and aligned corpus metadata for interactive retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss

from ..load_miracl import load_corpus


def load_index_and_corpus(config: Dict[str, Any]) -> Tuple[faiss.Index, List[Dict[str, str]]]:
    """Load the FAISS index and corpus metadata aligned to index positions.

    The corpus documents are re-ordered to match the ``corpus_ids.json`` file
    written when the embeddings/index were created.

    Raises:
        FileNotFoundError: If the index, corpus IDs, or corpus are missing.
    """
    faiss_dir = Path(config.get("faiss_dir", "outputs/faiss"))
    index_path = faiss_dir / "index.bin"
    ids_path = faiss_dir.parent / "embeddings" / "corpus_ids.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run 'python -m src.main --step evaluate' to build it first."
        )
    index = faiss.read_index(str(index_path))

    if not ids_path.exists():
        raise FileNotFoundError(f"Corpus ID mapping not found at {ids_path}")

    with open(ids_path, "r", encoding="utf-8") as f:
        corpus_ids: List[str] = json.load(f)

    corpus = load_corpus(config)
    doc_map: Dict[str, Dict[str, str]] = {d["doc_id"]: d for d in corpus}

    aligned: List[Dict[str, str]] = []
    for doc_id in corpus_ids:
        if doc_id in doc_map:
            aligned.append(doc_map[doc_id])
        else:
            aligned.append({"doc_id": doc_id, "title": "", "text": ""})

    print(f"[INFO] Loaded FAISS index ({index.ntotal} vectors) and {len(aligned)} documents.")
    return index, aligned
