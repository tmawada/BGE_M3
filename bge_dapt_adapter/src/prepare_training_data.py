"""
Training data preparation for BGE-M3 + LoRA Adapter contrastive learning.

Constructs (query, positive_passage, hard_negative_passages) triplets from
the MIRACL dataset for training the LoRA adapter with InfoNCE loss.

Hard negatives are mined using the baseline BGE-M3 embeddings and FAISS index.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


def _build_qrels_lookup(qrels: List[Dict[str, Any]]) -> Dict[str, set]:
    """Build a fast lookup: query_id -> set of relevant doc_ids."""
    lookup: Dict[str, set] = {}
    for entry in qrels:
        qid = entry["query_id"]
        did = entry["relevant_doc"]
        if qid not in lookup:
            lookup[qid] = set()
        lookup[qid].add(did)
    return lookup


def _build_corpus_lookup(corpus: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Build a fast lookup: doc_id -> full document dict."""
    return {doc["doc_id"]: doc for doc in corpus}


def mine_hard_negatives(
    query_embeddings: np.ndarray,
    query_ids: List[str],
    corpus_ids: List[str],
    qrels_lookup: Dict[str, set],
    config: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Mine hard negatives for each query using FAISS."""
    import faiss

    faiss_dir = Path(config.get("faiss_dir", "faiss"))
    index_path = faiss_dir / "index.bin"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run the baseline pipeline first to build the index."
        )

    index = faiss.read_index(str(index_path))
    training_config = config.get("training", {})
    num_hard_negatives = training_config.get("num_hard_negatives", 7)
    search_k = num_hard_negatives + 50

    if query_embeddings.dtype != np.float32:
        query_embeddings = query_embeddings.astype(np.float32)

    print(f"[INFO] Mining hard negatives (top-{search_k} candidates per query)...")
    scores, indices = index.search(query_embeddings, search_k)

    hard_negatives: Dict[str, List[str]] = {}

    for i, qid in enumerate(tqdm(query_ids, desc="Mining negatives")):
        relevant = qrels_lookup.get(qid, set())
        negatives: List[str] = []
        for j in range(search_k):
            idx = indices[i][j]
            if 0 <= idx < len(corpus_ids):
                doc_id = corpus_ids[idx]
                if doc_id not in relevant:
                    negatives.append(doc_id)
                    if len(negatives) >= num_hard_negatives:
                        break
        hard_negatives[qid] = negatives

    return hard_negatives


def prepare_training_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare contrastive training data from MIRACL.

    Each training example:
    {
        "query_id": str, "query": str,
        "positive": {"doc_id", "title", "passage"},
        "negatives": [{"doc_id", "title", "passage"}, ...]
    }
    """
    from .load_dataset import load_all

    data_dir = Path(config.get("data_dir", "data"))
    training_data_path = data_dir / "training_pairs.json"

    if training_data_path.exists():
        print(f"[INFO] Training data already exists at {training_data_path}, loading.")
        with open(training_data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("[INFO] Preparing training data...")
    corpus, queries, qrels = load_all(config)
    qrels_lookup = _build_qrels_lookup(qrels)
    corpus_lookup = _build_corpus_lookup(corpus)

    embeddings_dir = Path(config.get("embeddings_dir", "embeddings"))
    query_emb_path = embeddings_dir / "query.npy"
    query_ids_path = embeddings_dir / "query_ids.json"
    corpus_ids_path = embeddings_dir / "corpus_ids.json"

    if not query_emb_path.exists():
        raise FileNotFoundError(
            f"Baseline query embeddings not found at {query_emb_path}. "
            "Run the baseline pipeline first."
        )

    query_embeddings = np.load(str(query_emb_path))
    with open(query_ids_path, "r", encoding="utf-8") as f:
        query_ids = json.load(f)
    with open(corpus_ids_path, "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)

    hard_negatives = mine_hard_negatives(
        query_embeddings, query_ids, corpus_ids, qrels_lookup, config
    )

    query_lookup = {q["query_id"]: q["query"] for q in queries}
    training_config = config.get("training", {})
    random.seed(training_config.get("seed", 42))

    training_data: List[Dict[str, Any]] = []
    skipped = 0

    for qid in tqdm(query_ids, desc="Building training pairs"):
        if qid not in qrels_lookup:
            skipped += 1
            continue

        query_text = query_lookup.get(qid, "")
        if not query_text:
            skipped += 1
            continue

        pos_doc_id = random.choice(list(qrels_lookup[qid]))
        pos_doc = corpus_lookup.get(pos_doc_id)
        if pos_doc is None:
            skipped += 1
            continue

        neg_docs = []
        for neg_id in hard_negatives.get(qid, []):
            neg_doc = corpus_lookup.get(neg_id)
            if neg_doc is not None:
                neg_docs.append({
                    "doc_id": neg_doc["doc_id"],
                    "title": neg_doc.get("title", ""),
                    "passage": neg_doc["passage"],
                })

        if not neg_docs:
            skipped += 1
            continue

        training_data.append({
            "query_id": qid,
            "query": query_text,
            "positive": {
                "doc_id": pos_doc["doc_id"],
                "title": pos_doc.get("title", ""),
                "passage": pos_doc["passage"],
            },
            "negatives": neg_docs,
        })

    print(f"[INFO] Prepared {len(training_data)} training examples (skipped {skipped}).")

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(training_data_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved training data to {training_data_path}")

    return training_data
