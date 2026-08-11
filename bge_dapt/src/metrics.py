"""
Evaluation metrics for dense retrieval.

Implements Recall@K, MRR@K, and nDCG@K from scratch.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set


def _build_qrels_dict(qrels: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Convert qrels list to a nested dictionary for fast lookup.

    Args:
        qrels: List of {"query_id": str, "relevant_doc": str}.

    Returns:
        Dict mapping query_id -> {doc_id: relevance_score}.
    """
    qrels_dict: Dict[str, Dict[str, int]] = {}
    for entry in qrels:
        qid = entry["query_id"]
        did = entry["relevant_doc"]
        if qid not in qrels_dict:
            qrels_dict[qid] = {}
        qrels_dict[qid][did] = 1
    return qrels_dict


def recall_at_k(
    results: Dict[str, List[str]],
    qrels: List[Dict[str, Any]],
    k: int,
) -> float:
    """Compute mean Recall@K across all queries with relevance judgments."""
    qrels_dict = _build_qrels_dict(qrels)
    scores: List[float] = []

    for qid, relevant_docs in qrels_dict.items():
        retrieved_at_k = results.get(qid, [])[:k]
        relevant_set: Set[str] = set(relevant_docs.keys())
        num_relevant_retrieved = len(set(retrieved_at_k) & relevant_set)
        total_relevant = len(relevant_set)

        if total_relevant > 0:
            scores.append(num_relevant_retrieved / total_relevant)
        else:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(
    results: Dict[str, List[str]],
    qrels: List[Dict[str, Any]],
    k: int = 10,
) -> float:
    """Compute mean Reciprocal Rank at K (MRR@K)."""
    qrels_dict = _build_qrels_dict(qrels)
    reciprocal_ranks: List[float] = []

    for qid, relevant_docs in qrels_dict.items():
        relevant_set: Set[str] = set(relevant_docs.keys())
        rr = 0.0
        for rank, doc_id in enumerate(results.get(qid, [])[:k], start=1):
            if doc_id in relevant_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def ndcg_at_k(
    results: Dict[str, List[str]],
    qrels: List[Dict[str, Any]],
    k: int = 10,
) -> float:
    """Compute mean normalized Discounted Cumulative Gain at K (nDCG@K)."""
    qrels_dict = _build_qrels_dict(qrels)
    ndcg_scores: List[float] = []

    for qid, relevant_docs in qrels_dict.items():
        retrieved = results.get(qid, [])[:k]
        dcg = 0.0
        for i, doc_id in enumerate(retrieved):
            rel = relevant_docs.get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because rank is 1-indexed

        ideal_rels = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2 ** rel - 1) / math.log2(i + 2)

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def evaluate_all(
    results: Dict[str, List[str]],
    qrels: List[Dict[str, Any]],
    top_k: int = 10,
) -> Dict[str, float]:
    """Run all evaluation metrics.

    Args:
        results: Dict mapping query_id to ranked list of retrieved doc_ids.
        qrels: List of relevance judgments.
        top_k: Maximum K value for metrics.

    Returns:
        Dictionary of metric names to scores.
    """
    return {
        "Recall@1": recall_at_k(results, qrels, k=1),
        "Recall@5": recall_at_k(results, qrels, k=5),
        "Recall@10": recall_at_k(results, qrels, k=10),
        "MRR@10": mrr_at_k(results, qrels, k=top_k),
        "nDCG@10": ndcg_at_k(results, qrels, k=top_k),
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 40)
    print("  Evaluation Results")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print("=" * 40 + "\n")
