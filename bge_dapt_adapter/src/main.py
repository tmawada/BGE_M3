"""
Main pipeline orchestrator for the BGE + DAPT + LoRA Adapter project.

Runs the three-stage pipeline:
    Stage 1 — DAPT MLM pretraining on the Indonesian review corpus
    Stage 2 — LoRA adapter training on the DAPT checkpoint (MIRACL, InfoNCE)
    Stage 3 — Dense retrieval evaluation (encode, index, retrieve, metrics)

Usage:
    # Run the full pipeline
    python -m src.main

    # Run individual steps
    python -m src.main --step load_data
    python -m src.main --step train_dapt
    python -m src.main --step prepare_mining
    python -m src.main --step prepare_data
    python -m src.main --step train_adapter
    python -m src.main --step encode_corpus
    python -m src.main --step encode_query
    python -m src.main --step build_index
    python -m src.main --step retrieve
    python -m src.main --step evaluate

    # Custom config file
    python -m src.main --config config/custom.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load pipeline configuration from a YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    print(f"[INFO] Loaded configuration from {config_path}")
    return config


def _section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Merge a nested section (e.g. 'dapt') into the top-level config."""
    merged = {key: value for key, value in config.items() if key not in ("dapt", "retrieval")}
    merged.update(config.get(name) or {})
    return merged


def _mining_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Config for hard-negative mining: vanilla model + isolated output dirs."""
    cfg = dict(config)
    cfg["use_adapter"] = False
    cfg["embeddings_dir"] = config.get("mining_embeddings_dir", "embeddings_mining")
    cfg["faiss_dir"] = config.get("mining_faiss_dir", "faiss_mining")
    return cfg


# ----------------------------------------------------------------------
#  Stage 1 — DAPT
# ----------------------------------------------------------------------
def step_load_data(config: Dict[str, Any]) -> None:
    """Step 1: Prepare the DAPT MLM dataset from the review corpus."""
    from .load_dapt_data import load_reviews

    print("\n" + "=" * 60)
    print("  STEP: Load DAPT Data")
    print("=" * 60)
    load_reviews(_section(config, "dapt"))


def step_train_dapt(config: Dict[str, Any]) -> Dict[str, Any]:
    """Step 2: Domain-Adaptive Pretraining (MLM)."""
    from .train_dapt import train_dapt

    print("\n" + "#" * 60)
    print("  STAGE 1: DAPT")
    print("#" * 60)
    return train_dapt(_section(config, "dapt"))


# ----------------------------------------------------------------------
#  Stage 2 — Adapter
# ----------------------------------------------------------------------
def step_prepare_mining(config: Dict[str, Any]) -> None:
    """Step 3: Encode corpus/queries and build a FAISS index with the
    vanilla model, used for hard-negative mining."""
    import numpy as np

    from .build_index import build_index
    from .encode_corpus import encode_corpus
    from .encode_query import encode_queries
    from .load_dataset import load_corpus, load_queries
    from .model_loader import load_model

    print("\n" + "=" * 60)
    print("  STEP: Prepare Mining Artifacts (vanilla model)")
    print("=" * 60)

    mining = _mining_config(config)

    corpus = load_corpus(mining)
    queries = load_queries(mining)
    encoder = load_model(mining)

    corpus_embeddings = encode_corpus(corpus, encoder, mining)
    query_embeddings = encode_queries(queries, encoder, mining)
    build_index(corpus_embeddings, mining)

    print(f"[INFO] Mining artifacts ready: {mining['embeddings_dir']}, {mining['faiss_dir']}")


def step_prepare_data(config: Dict[str, Any]) -> None:
    """Step 4: Build contrastive training data with hard negatives."""
    from .prepare_training_data import prepare_training_data

    print("\n" + "=" * 60)
    print("  STEP: Prepare Training Data")
    print("=" * 60)

    training_data = prepare_training_data(_mining_config(config))
    print(f"[INFO] Total training examples: {len(training_data)}")


def step_train_adapter(config: Dict[str, Any]) -> None:
    """Step 5: Train LoRA adapter on the DAPT checkpoint."""
    from .train_adapter import train_adapter

    print("\n" + "#" * 60)
    print("  STAGE 2: LoRA Adapter Training")
    print("#" * 60)

    output_dir = train_adapter(config)
    print(f"[INFO] Adapter saved to: {output_dir}")


# ----------------------------------------------------------------------
#  Stage 3 — Evaluation
# ----------------------------------------------------------------------
def step_encode_corpus(config: Dict[str, Any]) -> None:
    """Step 6: Encode corpus documents with the DAPT + adapter model."""
    from .encode_corpus import encode_corpus
    from .load_dataset import load_corpus
    from .model_loader import load_model

    print("\n" + "=" * 60)
    print("  STEP: Encode Corpus (DAPT + adapter)")
    print("=" * 60)

    corpus = load_corpus(config)
    encoder = load_model(config)
    encode_corpus(corpus, encoder, config)


def step_encode_query(config: Dict[str, Any]) -> None:
    """Step 7: Encode queries with the DAPT + adapter model."""
    from .encode_query import encode_queries
    from .load_dataset import load_queries
    from .model_loader import load_model

    print("\n" + "=" * 60)
    print("  STEP: Encode Queries (DAPT + adapter)")
    print("=" * 60)

    queries = load_queries(config)
    encoder = load_model(config)
    encode_queries(queries, encoder, config)


def step_build_index(config: Dict[str, Any]) -> None:
    """Step 8: Build FAISS index from adapter corpus embeddings."""
    import numpy as np

    from .build_index import build_index

    print("\n" + "=" * 60)
    print("  STEP: Build FAISS Index")
    print("=" * 60)

    embeddings_dir = Path(config.get("embeddings_dir", "embeddings"))
    corpus_embeddings = np.load(str(embeddings_dir / "corpus.npy"))
    build_index(corpus_embeddings, config)


def step_retrieve(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Step 9: Perform retrieval."""
    import numpy as np

    from .build_index import build_index
    from .retrieve import retrieve

    print("\n" + "=" * 60)
    print("  STEP: Retrieve")
    print("=" * 60)

    embeddings_dir = Path(config.get("embeddings_dir", "embeddings"))

    corpus_embeddings = np.load(str(embeddings_dir / "corpus.npy"))
    query_embeddings = np.load(str(embeddings_dir / "query.npy"))

    with open(embeddings_dir / "corpus_ids.json", "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)
    with open(embeddings_dir / "query_ids.json", "r", encoding="utf-8") as f:
        query_ids = json.load(f)

    index = build_index(corpus_embeddings, config)

    return retrieve(index, query_embeddings, query_ids, corpus_ids, config)


def step_evaluate(config: Dict[str, Any]) -> Dict[str, float]:
    """Step 10: Evaluate retrieval results."""
    from .load_dataset import load_qrels
    from .metrics import evaluate_all, print_metrics

    print("\n" + "=" * 60)
    print("  STEP: Evaluate")
    print("=" * 60)

    eval_dir = Path(config.get("evaluation_dir", "evaluation"))
    results_path = eval_dir / "retrieval_results.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Retrieval results not found at {results_path}. "
            "Run the retrieve step first."
        )

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    qrels = load_qrels(config)
    top_k = config.get("top_k", 10)

    metrics = evaluate_all(results, qrels, top_k)
    print_metrics(metrics)

    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "results.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved evaluation results to {metrics_path}")

    return metrics


# ----------------------------------------------------------------------
#  Full pipeline
# ----------------------------------------------------------------------
def run_full_pipeline(config: Dict[str, Any]) -> None:
    """Run the complete three-stage pipeline."""
    import numpy as np

    from .build_index import build_index
    from .encode_corpus import encode_corpus
    from .encode_query import encode_queries
    from .load_dataset import load_all
    from .metrics import evaluate_all, print_metrics
    from .model_loader import load_model
    from .retrieve import retrieve

    print("\n" + "#" * 60)
    print("  BGE + DAPT + LoRA Adapter Pipeline")
    print("#" * 60)

    step_load_data(config)
    step_train_dapt(config)

    mining = _mining_config(config)
    corpus, queries, _ = load_all(mining)
    encoder = load_model(mining)
    corpus_embeddings = encode_corpus(corpus, encoder, mining)
    query_embeddings = encode_queries(queries, encoder, mining)
    build_index(corpus_embeddings, mining)

    step_prepare_data(config)
    step_train_adapter(config)

    corpus, queries, qrels = load_all(config)
    encoder = load_model(config)
    corpus_embeddings = encode_corpus(corpus, encoder, config)
    query_embeddings = encode_queries(queries, encoder, config)
    index = build_index(corpus_embeddings, config)

    query_ids = [q["query_id"] for q in queries]
    corpus_ids = [doc["doc_id"] for doc in corpus]
    results = retrieve(index, query_embeddings, query_ids, corpus_ids, config)

    top_k = config.get("top_k", 10)
    metrics = evaluate_all(results, qrels, top_k)
    print_metrics(metrics)

    eval_dir = Path(config.get("evaluation_dir", "evaluation"))
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "results.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved evaluation results to {metrics_path}")

    print("\n" + "#" * 60)
    print("  Pipeline Complete!")
    print("#" * 60)


def main() -> None:
    """Entry point for the pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="BGE + DAPT + LoRA Adapter Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        choices=[
            "load_data",
            "train_dapt",
            "prepare_mining",
            "prepare_data",
            "train_adapter",
            "encode_corpus",
            "encode_query",
            "build_index",
            "retrieve",
            "evaluate",
        ],
        help="Run a specific pipeline step instead of the full pipeline.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    step_map = {
        "load_data": step_load_data,
        "train_dapt": step_train_dapt,
        "prepare_mining": step_prepare_mining,
        "prepare_data": step_prepare_data,
        "train_adapter": step_train_adapter,
        "encode_corpus": step_encode_corpus,
        "encode_query": step_encode_query,
        "build_index": step_build_index,
        "retrieve": step_retrieve,
        "evaluate": step_evaluate,
    }

    try:
        if args.step is None:
            run_full_pipeline(config)
        else:
            step_map[args.step](config)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
