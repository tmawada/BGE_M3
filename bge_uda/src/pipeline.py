"""One CLI for the whole UDA workflow.

Usage (run from BGE_UDA/):
  python -m src.pipeline --stage baseline       # formal MIRACL eval
  python -m src.pipeline --stage dapt           # unsupervised MLM
  python -m src.pipeline --stage make_pairs     # informal triplets
  python -m src.pipeline --stage train          # LoRA contrastive
  python -m src.pipeline --stage eval_informal  # informal retrieval eval
  python -m src.pipeline --stage demo           # formal vs informal cosine
  python -m src.pipeline --stage all            # baseline->pairs->train->evals

After training: set use_adapter: true (and model_name to DAPT ckpt) in
config/config.yaml, then re-run eval_informal / demo to see the gain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config_loader import load_config
from .dataset import load_all, load_pairs
from .encode import encode_corpus, encode_queries
from .evaluate import evaluate_all, print_metrics, retrieve, save_results
from .index import build_index
from .model import load_model


def _emb_cache(config, name: str):
    """Helper: paths to cached embeddings for <name>."""
    from .encode import _out_dir
    d = _out_dir(config)
    return d / f"{name}.npy", d / f"{name}_ids.json"


def stage_baseline(config) -> None:
    """Formal baseline: encode corpus+queries, FAISS search, report metrics."""
    corpus, queries, qrels = load_all(config)  # MIRACL formal data
    encoder = load_model(config)  # BGEM3Encoder (use_adapter=false)
    cvecs = encode_corpus(corpus, encoder, config)  # (N_docs, 1024)
    qvecs, qids = encode_queries(queries, encoder, config, name="query")
    corpus_ids = json.loads((_emb_cache(config, "corpus")[1]).read_text(encoding="utf-8"))
    index = build_index(cvecs, config)  # IndexFlatIP == cosine
    results = retrieve(index, qvecs, qids, corpus_ids, int(config.get("top_k", 5)))
    metrics = evaluate_all(results, qrels)  # Recall/MRR/nDCG vs formal qrels
    print_metrics(metrics, "Baseline (formal queries)")
    save_results(results, metrics, config, "baseline_formal")


def stage_make_pairs(config) -> None:
    """Build training_pairs.json from train_pairs.csv (informal -> gold doc)."""
    from .make_pairs import prepare_training_data
    prepare_training_data(config)  # needs baseline embeddings+index to exist


def stage_train(config) -> None:
    """LoRA contrastive training on informal triplets."""
    from .train_lora import train_adapter
    train_adapter(config)  # saves adapters/bge-m3-lora/
    print("[NEXT] Set use_adapter: true in config, then run --stage eval_informal")


def stage_eval_informal(config) -> None:
    """Evaluate INFORMAL queries against formal qrels (same gold docs)."""
    import numpy as np
    from .index import load_index
    pairs = load_pairs(config)  # your 4072 rows
    # Informal queries reuse formal qids so we can score vs formal qrels.
    _, _, qrels = load_all(config)
    encoder = load_model(config)  # set use_adapter:true to test adapted model
    informal = [{"query_id": p["query_id"], "query": p["informal"]} for p in pairs]
    # Encode without cache (model may have changed between runs).
    qvecs = encoder.encode([q["query"] for q in informal], config.get("batch_size", 32),
                           config.get("max_length", 512), config.get("normalize_embeddings", True))
    qids = [q["query_id"] for q in informal]
    from .encode import _out_dir
    corpus_ids = json.loads((_out_dir(config) / "corpus_ids.json").read_text(encoding="utf-8"))
    index = load_index(config)
    results = retrieve(index, qvecs, qids, corpus_ids, int(config.get("top_k", 5)))
    metrics = evaluate_all(results, qrels)  # informal queries vs FORMAL gold docs
    print_metrics(metrics, "Informal queries (register robustness)")
    save_results(results, metrics, config, "informal")


def stage_demo(config) -> None:
    """Quick cosine check on 3 example formal/informal pairs."""
    from .inference import compare_pair
    encoder = load_model(config)
    pairs = load_pairs(config)[:3]  # first 3 rows as smoke test
    for p in pairs:
        compare_pair(encoder, p["formal"], p["informal"], config)


def main() -> None:
    """Parse --stage and dispatch (keeps CLI tiny and readable)."""
    ap = argparse.ArgumentParser(description="BGE-UDA pipeline")
    ap.add_argument("--stage", default="baseline",
                    choices=["baseline", "dapt", "make_pairs", "train", "eval_informal", "demo", "all"],
                    help="Which workflow stage to run")
    ap.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = ap.parse_args()
    config = load_config(args.config)
    # Dispatch table keeps main() flat and easy to extend.
    if args.stage == "baseline":
        stage_baseline(config)
    elif args.stage == "dapt":
        from .dapt import run_dapt_mlm
        run_dapt_mlm(config)
    elif args.stage == "make_pairs":
        stage_make_pairs(config)
    elif args.stage == "train":
        stage_train(config)
    elif args.stage == "eval_informal":
        stage_eval_informal(config)
    elif args.stage == "demo":
        stage_demo(config)
    elif args.stage == "all":
        stage_baseline(config)
        stage_make_pairs(config)
        stage_train(config)
        stage_eval_informal(config)


if __name__ == "__main__":
    main()
