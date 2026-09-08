"""Embedding-similarity proof: informal ~= formal (held-out, single encoder).

Metrics (all frozen, no training):
  paired_cosine   mean/median/std of cos(inf_i, form_i) — register equivalence
  alignment       E||inf - form||^2 (Wang & Isola; lower = closer)
  uniformity      log E exp(-2||x-y||^2) over random cross pairs (lower = spread)
  random_cosine   mean cos(inf_i, form_j!=i) — discrimination control
  infonce_eval    frozen InfoNCE with 1 positive + n_random negatives
  match_R@1/R@5/MRR  rank true formal_i among all formals given inf_i
  doc-anchored    cos(form,gold), cos(inf,gold), gap = form_gold - inf_gold

Compare similarity.json across encoders (base vs dapt vs lora) for the gain.
NumPy only so it runs on CPU without torch.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _norm(mat: np.ndarray) -> np.ndarray:
    n = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    return mat / n


def _paired_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    cos = np.sum(a * b, axis=1)  # normalized => dot == cosine
    euc = np.linalg.norm(a - b, axis=1)
    return {
        "paired_cosine_mean": float(np.mean(cos)),
        "paired_cosine_median": float(np.median(cos)),
        "paired_cosine_std": float(np.std(cos)),
        "paired_cosine_p10": float(np.percentile(cos, 10)),
        "paired_cosine_p90": float(np.percentile(cos, 90)),
        "paired_euc_mean": float(np.mean(euc)),
        "alignment": float(np.mean(euc ** 2)),
        "frac_cos_gt_08": float(np.mean(cos > 0.8)),
        "frac_cos_gt_09": float(np.mean(cos > 0.9)),
    }


def _uniformity(vecs: np.ndarray, seed: int = 42, n_sample: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    n = len(vecs)
    if n < 2:
        return 0.0
    m = min(n_sample, n * 4)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    mask = i != j
    i, j = i[mask], j[mask]
    if len(i) == 0:
        return 0.0
    d2 = np.sum((vecs[i] - vecs[j]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-2.0 * d2))))


def _infonce_eval(inf: np.ndarray, form: np.ndarray, temp: float, n_random: int,
                  seed: int) -> Dict[str, float]:
    """Frozen InfoNCE: positive = true formal, negatives = random other formals."""
    rng = np.random.default_rng(seed)
    n = len(inf)
    losses, hits1, hits5, rrs = [], [], [], []
    sim_full = inf @ form.T  # N x N matching matrix (also reused below)
    for i in range(n):
        cand = [x for x in rng.choice(n, size=min(n_random + 1, n), replace=False).tolist()
                if x != i][:n_random]
        logits = np.array([sim_full[i, i] / temp] + [sim_full[i, j] / temp for j in cand])
        logits -= logits.max()
        exp = np.exp(logits)
        losses.append(float(-np.log(exp[0] / exp.sum())))
        rank = int(1 + np.sum(sim_full[i] > sim_full[i, i]))
        hits1.append(1.0 if rank == 1 else 0.0)
        hits5.append(1.0 if rank <= 5 else 0.0)
        rrs.append(1.0 / rank)
    return {
        "infonce_eval_loss": float(np.mean(losses)),
        "match_R@1": float(np.mean(hits1)),
        "match_R@5": float(np.mean(hits5)),
        "match_MRR": float(np.mean(rrs)),
        "random_cosine_mean": float((sim_full.sum() - np.trace(sim_full)) / max(1, n * (n - 1))),
    }


def evaluate_similarity(pairs: List[Dict[str, str]], corpus: List[Dict[str, str]],
                        qrels: List[Dict[str, str]], encoder, config: Dict[str, Any]) -> Dict[str, Any]:
    """Encode held-out pairs (+ gold docs) with ONE encoder; return summary."""
    sim_cfg = config.get("similarity", {})
    temp = float(sim_cfg.get("temperature", 0.05))
    n_random = int(sim_cfg.get("n_random", 10))
    seed = int(sim_cfg.get("seed", 42))
    bs = int(config.get("batch_size", 32))
    ml = int(config.get("max_length", 512))
    norm = bool(config.get("normalize_embeddings", True))

    form_txt = [p["formal"] for p in pairs]
    inf_txt = [p["informal"] for p in pairs]
    form = _norm(np.asarray(encoder.encode(form_txt, bs, ml, norm), dtype=np.float64))
    inf = _norm(np.asarray(encoder.encode(inf_txt, bs, ml, norm), dtype=np.float64))

    summary: Dict[str, float] = {}
    summary.update(_paired_stats(inf, form))
    summary["uniformity_all"] = _uniformity(np.concatenate([inf, form]), seed=seed)
    summary.update(_infonce_eval(inf, form, temp, n_random, seed))

    # Doc-anchored: does informal reach the same gold doc as formal?
    doc_of = {str(d["doc_id"]): str(d["passage"]) for d in corpus}
    gold: Dict[str, str] = {}
    for r in qrels:
        gold.setdefault(str(r["query_id"]), str(r["relevant_doc"]))
    f2g, i2g = [], []
    for p in pairs:
        gid = gold.get(str(p["query_id"]))
        if gid is None or gid not in doc_of:
            continue
        f2g.append(p["formal"])
        i2g.append(p["informal"])
    if f2g:
        g_txt = [doc_of[gold[str(p["query_id"])]] for p in pairs
                 if gold.get(str(p["query_id"])) in doc_of]
        g = _norm(np.asarray(encoder.encode(g_txt, bs, ml, norm), dtype=np.float64))
        # align: pairs order filtered to those with gold
        keep = [i for i, p in enumerate(pairs) if gold.get(str(p["query_id"])) in doc_of]
        fg = np.sum(form[keep] * g, axis=1)
        ig = np.sum(inf[keep] * g, axis=1)
        summary["doc_form_gold_cos"] = float(np.mean(fg))
        summary["doc_inf_gold_cos"] = float(np.mean(ig))
        summary["doc_gap_form_minus_inf"] = float(np.mean(fg - ig))
        summary["doc_n"] = float(len(keep))
    else:
        summary["doc_n"] = 0.0
    summary["n"] = float(len(pairs))
    return {"summary": summary}
