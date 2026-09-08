"""One CLI for the whole UDA workflow.

Usage (run from BGE_UDA/):
  python -m src.pipeline --stage baseline       # formal MIRACL eval
  python -m src.pipeline --stage dapt           # unsupervised MLM (train split only)
  python -m src.pipeline --stage make_pairs     # informal triplets (train split only)
  python -m src.pipeline --stage train          # LoRA contrastive
  python -m src.pipeline --stage eval_informal  # informal retrieval eval (held-out only)
  python -m src.pipeline --stage eval_sim       # informal~=formal embedding proof (held-out)
  python -m src.pipeline --stage demo           # formal vs informal cosine
  python -m src.pipeline --stage all            # baseline->pairs->train->evals

After training: set use_adapter: true (and model_name to DAPT ckpt) in
config/config.yaml, then re-run eval_informal / eval_sim / demo to see the gain.
Each encoder uses its own corpus embeddings + FAISS index (see encode.encoder_tag);
never reuse a baseline index for adapter queries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config_loader import load_config
from .dataset import filter_qrels, load_all, load_pairs, split_pairs
from .encode import encode_corpus, encode_queries, encoder_tag
from .evaluate import evaluate_all, print_metrics, retrieve, save_results
from .index import build_index
from .model import load_model


def _free_cuda() -> None:
    """Best-effort CUDA cache release between encoder loads (3-way eval)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _emb_cache(config, name: str):
    """Helper: paths to cached embeddings for <name>."""
    from .encode import _out_dir
    d = _out_dir(config)
    return d / f"{name}.npy", d / f"{name}_ids.json"


def stage_baseline(config) -> None:
    """Formal baseline: encode corpus+queries with SAME encoder, report metrics."""
    corpus, queries, qrels = load_all(config)  # MIRACL formal data
    encoder = load_model(config)
    tag = encoder_tag(config)
    cvecs = encode_corpus(corpus, encoder, config)  # per-encoder cache
    qvecs, qids = encode_queries(queries, encoder, config, name="query")
    corpus_ids = json.loads((_emb_cache(config, f"corpus_{tag}")[1]).read_text(encoding="utf-8"))
    index = build_index(cvecs, config)  # per-encoder index; IndexFlatIP == cosine
    top_k = int(config.get("top_k", 10))
    results = retrieve(index, qvecs, qids, corpus_ids, top_k)
    metrics = evaluate_all(results, filter_qrels(qrels, qids), qids=qids)
    print_metrics(metrics, f"Baseline (formal queries) [{tag}] top_k={top_k}")
    save_results(results, metrics, config, "baseline_formal")
    _free_cuda()


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
    """Evaluate INFORMAL held-out queries (same encoder for corpus+queries)."""
    corpus, _, qrels = load_all(config)
    pairs = load_pairs(config)
    _, held = split_pairs(pairs, config)  # held-out only: no train-on-test
    if not held:
        raise ValueError("Held-out split is empty; check split.held_out_ratio.")
    # Informal queries reuse formal qids so we can score vs formal qrels.
    encoder = load_model(config)  # set use_adapter:true to test adapted model
    tag = encoder_tag(config)
    # CRITICAL FIX: re-encode corpus with THIS encoder; never reuse another
    # encoder's index (cross-space IP is meaningless).
    cvecs = encode_corpus(corpus, encoder, config)
    corpus_ids = json.loads((_emb_cache(config, f"corpus_{tag}")[1]).read_text(encoding="utf-8"))
    index = build_index(cvecs, config)
    informal = [{"query_id": p["query_id"], "query": p["informal"]} for p in held]
    # Encode without cache (model may have changed between runs).
    qvecs = encoder.encode([q["query"] for q in informal], config.get("batch_size", 32),
                           config.get("max_length", 512), config.get("normalize_embeddings", True))
    qids = [q["query_id"] for q in informal]
    top_k = int(config.get("top_k", 10))
    results = retrieve(index, qvecs, qids, corpus_ids, top_k)
    metrics = evaluate_all(results, filter_qrels(qrels, qids), qids=qids)
    print_metrics(metrics, f"Informal queries held-out n={len(held)} [{tag}] top_k={top_k}")
    save_results(results, metrics, config, "informal")
    _free_cuda()


def stage_eval_sim(config) -> None:
    """Embedding proof: informal ~= formal (held-out, per current encoder)."""
    from .eval_similarity import evaluate_similarity
    pairs = load_pairs(config)
    _, held = split_pairs(pairs, config)
    if not held:
        raise ValueError("Held-out split is empty; check split.held_out_ratio.")
    corpus, _, qrels = load_all(config)
    encoder = load_model(config)
    out = evaluate_similarity(held, corpus, qrels, encoder, config)
    tag = encoder_tag(config)
    print_metrics(out["summary"], f"Embedding similarity held-out n={len(held)} [{tag}]")
    from .evaluate import save_results as _save
    # Reuse evaluation/ dir convention for JSON output.
    eval_dir = Path(config.get("evaluation_dir", "evaluation"))
    if not eval_dir.is_absolute():
        eval_dir = Path(__file__).resolve().parent.parent / eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / "similarity.json"
    import json as _json
    path.write_text(_json.dumps({"tag": tag, **out}, ensure_ascii=False, indent=2,
                                default=float), encoding="utf-8")
    print(f"[INFO] Saved -> {path}")
    _free_cuda()


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
                    choices=["baseline", "dapt", "make_pairs", "train", "eval_informal",
                             "eval_sim", "demo", "all"],
                    help="Which workflow stage to run")
    ap.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = ap.parse_args()
    config = load_config(args.config)
    # Dispatch table keeps main() flat and easy to extend.
    if args.stage == "baseline":
        stage_baseline(config)
    elif args.stage == "dapt":
        from .dapt import run_dapt_mlm
        run_dapt_mlm(config, train_only=True)
    elif args.stage == "make_pairs":
        stage_make_pairs(config)
    elif args.stage == "train":
        stage_train(config)
    elif args.stage == "eval_informal":
        stage_eval_informal(config)
    elif args.stage == "eval_sim":
        stage_eval_sim(config)
    elif args.stage == "demo":
        stage_demo(config)
    elif args.stage == "all":
        # NOTE: DAPT excluded from default `all` (hours of MLM). Run it explicitly
        # via --stage dapt, flip model_name to its output, then run `all`.
        stage_baseline(config)
        stage_make_pairs(config)
        stage_train(config)
        stage_eval_informal(config)
        stage_eval_sim(config)


if __name__ == "__main__":
    main()
