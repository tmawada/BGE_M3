"""Stage 2a: build UDA triplets (informal query -> formal doc).

Core UDA trick: train_pairs.csv has NO relevance labels, so we REUSE the
formal query's gold doc as the informal query's positive. Hard negatives are
mined from the baseline FAISS index (top-K excluding gold docs).
Output: data/training_pairs.json consumed by train_lora.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from .dataset import load_corpus, load_pairs, load_qrels


def _repo_root() -> Path:
    """Repo root = parent of src/ (for resolving relative paths)."""
    return Path(__file__).resolve().parent.parent


def mine_negatives_with_index(index, qvecs: np.ndarray, qids: List[str],
                              corpus_ids: List[str], gold: Dict[str, set],
                              num_negs: int, extra: int) -> Dict[str, List[str]]:
    """Same as above but with an already-loaded FAISS index."""
    if qvecs.dtype != np.float32:  # FAISS requires float32
        qvecs = qvecs.astype(np.float32)
    k = num_negs + extra  # over-fetch so we survive filtering out gold docs
    print(f"[INFO] Mining {num_negs} hard negs per query (search top-{k})...")
    _, indices = index.search(qvecs, k)  # scores unused; rank = hardness
    out: Dict[str, List[str]] = {}
    for i, qid in enumerate(tqdm(qids, desc="Mine negatives")):
        rel = gold.get(qid, set())
        negs = []
        for j in range(k):
            idx = int(indices[i][j])
            if 0 <= idx < len(corpus_ids) and corpus_ids[idx] not in rel:
                negs.append(corpus_ids[idx])  # first non-gold hits = hardest negs
                if len(negs) >= num_negs:
                    break
        out[qid] = negs
    return out


def prepare_training_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Join pairs + qrels + corpus + mined negs -> training_pairs.json."""
    from .encode import _out_dir  # reuse embeddings cache location
    from .index import load_index

    data_dir = Path(config.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = _repo_root() / data_dir
    out_path = data_dir / "training_pairs.json"
    if out_path.exists():  # cache: deterministic build, safe to reuse
        print(f"[INFO] Reusing {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    tr_cfg = config.get("training", {})
    num_negs = int(tr_cfg.get("num_hard_negatives", 7))
    extra = int(tr_cfg.get("search_k_extra", 50))
    seed = int(tr_cfg.get("seed", 42))
    random.seed(seed)

    corpus = load_corpus(config)  # for passage lookup by doc_id
    doc_of = {d["doc_id"]: d for d in corpus}
    gold: Dict[str, set] = {}  # qid -> {gold doc ids} from formal qrels
    for r in load_qrels(config):
        gold.setdefault(str(r["query_id"]), set()).add(str(r["relevant_doc"]))
    pairs = load_pairs(config)  # your 4072 informal/formal rows

    # Encode INFORMAL queries with current base model to mine their hard negs.
    from .model import load_model
    encoder = load_model(config)  # respects model_name (use DAPT ckpt here!)
    informal_texts = [p["informal"] for p in pairs]
    informal_qids = [p["query_id"] for p in pairs]
    print(f"[INFO] Encoding {len(informal_texts)} informal queries for mining...")
    qvecs = encoder.encode(informal_texts, config.get("batch_size", 32),
                           config.get("max_length", 512), config.get("normalize_embeddings", True))

    # Corpus ids come from the baseline encode step's cache file.
    emb_dir = _out_dir(config)
    corpus_ids = json.loads((emb_dir / "corpus_ids.json").read_text(encoding="utf-8"))
    index = load_index(config)
    neg_map = mine_negatives_with_index(index, qvecs, informal_qids, corpus_ids, gold, num_negs, extra)

    # Assemble triplets: informal query + formal's gold passage + hard negs.
    training: List[Dict[str, Any]] = []
    skipped = 0
    for p in pairs:
        qid = p["query_id"]
        if qid not in gold:  # no gold doc -> can't supervise this row
            skipped += 1
            continue
        pos_id = random.choice(sorted(gold[qid]))  # random gold if multi-relevant
        pos_doc = doc_of.get(pos_id)
        if pos_doc is None:
            skipped += 1
            continue
        neg_docs = [doc_of[n] for n in neg_map.get(qid, []) if n in doc_of]
        if not neg_docs:  # need at least 1 negative for contrastive loss
            skipped += 1
            continue
        training.append({"query_id": qid, "query": p["informal"],  # TRAIN ON INFORMAL
                         "formal": p["formal"],  # kept for debugging/analysis
                         "positive": {"doc_id": pos_doc["doc_id"], "passage": pos_doc["passage"]},
                         "negatives": [{"doc_id": n["doc_id"], "passage": n["passage"]} for n in neg_docs]})
    out_path.write_text(json.dumps(training, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Built {len(training)} triplets (skipped {skipped}) -> {out_path}")
    return training
