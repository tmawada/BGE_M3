"""Load FAISS index and corpus metadata for retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss


def load_index_and_corpus(config: Dict[str, Any]) -> Tuple[faiss.Index, List[Dict[str, str]]]:
    """Load FAISS index and corpus metadata from disk.

    Raises FileNotFoundError if resources are missing.
    """
    faiss_dir = Path(config.get("faiss_dir", "faiss"))
    index_path = faiss_dir / "index.bin"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(str(index_path))

    data_dir = Path(config.get("data_dir", "data"))
    corpus_path = data_dir / "corpus.json"
    corpus_jsonl_path = data_dir / "corpus.jsonl"

    if not corpus_path.exists() and not corpus_jsonl_path.exists():
        raise FileNotFoundError(
            f"Corpus metadata not found at {corpus_path} or {corpus_jsonl_path}"
        )

    corpus: List[Dict[str, str]] = []
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

    # Backfill titles from the original corpus.jsonl if the cached corpus lost them.
    if corpus_jsonl_path.exists():
        title_map: Dict[str, str] = {}
        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                doc_id = str(raw.get("docid") or raw.get("doc_id") or raw.get("id") or raw.get("_id") or "")
                if doc_id:
                    title_map[doc_id] = str(raw.get("title", ""))

        if corpus:
            updated = False
            for doc in corpus:
                doc_id = str(doc.get("doc_id") or doc.get("id") or doc.get("docid") or "")
                if doc_id and not doc.get("title"):
                    title = title_map.get(doc_id, "")
                    if title:
                        doc["title"] = title
                        updated = True

            if updated:
                with open(corpus_path, "w", encoding="utf-8") as f:
                    json.dump(corpus, f, ensure_ascii=False, indent=2)

    if not corpus:
        raise FileNotFoundError(f"Corpus metadata could not be loaded from {corpus_path}")

    return index, corpus