"""
Retrieval training sample builder (MIRACL).

Builds (query, positive-document) pairs from the MIRACL qrels. Hard negatives
are not mined in this stage; negatives come from other in-batch queries via
MultipleNegativesRankingLoss during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .load_miracl import load_all


def prepare_pairs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build (query, positive) training pairs and save them to disk.

    Args:
        config: Configuration dictionary with MIRACL paths and
            'dataset_path'.

    Returns:
        List of ``{"query_id", "query", "positive_doc_id", "positive_text"}``.
    """
    dataset_path = Path(config.get("dataset_path", "data/miracl_pairs.json"))

    corpus, queries, qrels = load_all(config)

    query_map: Dict[str, str] = {q["query_id"]: q["query"] for q in queries}
    doc_map: Dict[str, str] = {d["doc_id"]: d["text"] for d in corpus}

    pairs: List[Dict[str, Any]] = []
    skipped = 0
    for qrel in qrels:
        qid = qrel["query_id"]
        did = qrel["relevant_doc"]
        if qid not in query_map:
            skipped += 1
            continue
        if did not in doc_map:
            skipped += 1
            continue
        pairs.append(
            {
                "query_id": qid,
                "query": query_map[qid],
                "positive_doc_id": did,
                "positive_text": doc_map[did],
            }
        )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Training pairs       : {len(pairs)}")
    print(f"[INFO] Skipped (missing)    : {skipped}")
    print(f"[INFO] Saved pairs to {dataset_path}")
    return pairs
