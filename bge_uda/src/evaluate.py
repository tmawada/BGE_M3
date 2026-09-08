"""Retrieval + metrics: Recall@K, MRR, nDCG@K (from scratch, binary relevance)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


def retrieve(index, qvecs: np.ndarray, qids: List[str],
             corpus_ids: List[str], top_k: int) -> Dict[str, List[str]]:
    """Top-K search: FAISS positions -> doc ids."""
    if qvecs.dtype != np.float32:  # FAISS hard requirement
        qvecs = qvecs.astype(np.float32)
    print(f"[INFO] Searching top-{top_k} for {len(qids)} queries...")
    _, indices = index.search(qvecs, top_k)  # scores ignored here; ranking only
    results = {}
    for i, qid in enumerate(tqdm(qids, desc="Retrieve")):
        docs = []
        for j in range(top_k):
            idx = int(indices[i][j])
            if 0 <= idx < len(corpus_ids):  # -1 means no result; guard overflow
                docs.append(corpus_ids[idx])
        results[qid] = docs
    return results


def _qrel_map(qrels: List[Dict[str, str]]) -> Dict[str, set]:
    """Collapse qrels list -> {qid: {relevant doc ids}} for O(1) lookup."""
    m: Dict[str, set] = {}
    for r in qrels:
        m.setdefault(str(r["query_id"]), set()).add(str(r["relevant_doc"]))
    return m


def recall_at_k(results: Dict[str, List[str]], qrels: List[Dict[str, str]], k: int) -> float:
    """Recall@K = |retrieved@K ∩ relevant| / |relevant|, macro-averaged."""
    m = _qrel_map(qrels)
    scores = []
    for qid, rel in m.items():
        hit = len(set(results.get(qid, [])[:k]) & rel)  # count relevant in top-K
        scores.append(hit / len(rel) if rel else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def mrr(results: Dict[str, List[str]], qrels: List[Dict[str, str]]) -> float:
    """MRR = mean(1 / rank of first relevant doc). Missing -> 0."""
    m = _qrel_map(qrels)
    scores = []
    for qid, rel in m.items():
        rr = 0.0
        for rank, doc in enumerate(results.get(qid, []), start=1):
            if doc in rel:  # first hit determines score
                rr = 1.0 / rank
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(results: Dict[str, List[str]], qrels: List[Dict[str, str]], k: int) -> float:
    """nDCG@K with binary rels: DCG/IDCG where gain = (2^rel-1)/log2(rank+1)."""
    m = _qrel_map(qrels)
    scores = []
    for qid, rel in m.items():
        retr = results.get(qid, [])[:k]
        # DCG: discounted gain of actual ranking
        dcg = sum((1.0 if d in rel else 0.0) / math.log2(i + 2) for i, d in enumerate(retr))
        # IDCG: best possible (all relevant docs ranked first)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_all(results: Dict[str, List[str]], qrels: List[Dict[str, str]],
                 qids: List[str] | None = None) -> Dict[str, float]:
    """Compute the standard 5-metric set in one call.

    If qids is given, score only those queries (required for held-out eval;
    otherwise unevaluated qrels count as 0 and deflate scores).
    """
    if qids is not None:
        keep = set(map(str, qids))
        qrels = [r for r in qrels if str(r["query_id"]) in keep]
    depth = max((len(v) for v in results.values()), default=0)
    if depth and depth < 10:
        print(f"[WARNING] Retrieved depth {depth} < 10: "
              f"Recall@10/nDCG@10 are capped. Set top_k >= 10 (current depth={depth}).")
    return {"Recall@1": recall_at_k(results, qrels, 1), "Recall@5": recall_at_k(results, qrels, 5),
            "Recall@10": recall_at_k(results, qrels, 10), "MRR": mrr(results, qrels),
            "nDCG@10": ndcg_at_k(results, qrels, 10)}


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation") -> None:
    """Pretty console table for metrics dict."""
    print(f"\n===== {title} =====")
    for k, v in metrics.items():  # 4 decimals = standard IR reporting
        print(f"  {k:<10}: {v:.4f}")
    print("=" * 30)


def save_results(results: Dict[str, List[str]], metrics: Dict[str, float], config: Dict[str, Any], name: str) -> Path:
    """Save retrieval ranking + metrics JSON to evaluation/<name>.json."""
    d = Path(config.get("evaluation_dir", "evaluation"))
    if not d.is_absolute():  # keep outputs inside repo
        d = Path(__file__).resolve().parent.parent / d
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps({"metrics": metrics, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved -> {path}")
    return path
