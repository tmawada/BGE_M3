"""
Main pipeline orchestrator for BGE-M3 + LoRA Adapter retrieval.

Extends the baseline pipeline with adapter training steps:
  - prepare_data:   Build contrastive training pairs with hard negatives
  - train_adapter:  Train LoRA adapter on BGE-M3 with InfoNCE loss

After training, set `use_adapter: true` in config.yaml and re-run the
encode/retrieve/evaluate steps to measure the adapter's impact.

Usage:
    # Run full baseline pipeline (no adapter)
    python -m src.main

    # Prepare training data (requires baseline embeddings)
    python -m src.main --step prepare_data

    # Train the LoRA adapter
    python -m src.main --step train_adapter

    # Run full pipeline WITH adapter (set use_adapter: true first)
    python -m src.main

    # Run individual steps
    python -m src.main --step load_data
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
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[INFO] Loaded configuration from {config_path}")
    return config


def step_load_data(config: Dict[str, Any]) -> tuple:
    """Step 1-2: Load corpus, queries, and qrels."""
    from .load_dataset import load_all

    print("\n" + "=" * 60)
    print("  STEP: Load Dataset")
    print("=" * 60)
    return load_all(config)


def step_encode_corpus(config: Dict[str, Any]) -> None:
    """Step 3: Encode corpus documents."""
    from .encode_corpus import encode_corpus
    from .load_dataset import load_corpus
    from .model_loader import load_model

    print("\n" + "=" * 60)
    print("  STEP: Encode Corpus")
    print("=" * 60)

    corpus = load_corpus(config)
    encoder = load_model(config)
    encode_corpus(corpus, encoder, config)


def step_encode_query(config: Dict[str, Any]) -> None:
    """Step 4: Encode queries."""
    from .encode_query import encode_queries
    from .load_dataset import load_queries
    from .model_loader import load_model

    print("\n" + "=" * 60)
    print("  STEP: Encode Queries")
    print("=" * 60)

    queries = load_queries(config)
    encoder = load_model(config)
    encode_queries(queries, encoder, config)


def step_build_index(config: Dict[str, Any]) -> None:
    """Step 5: Build FAISS index."""
    import numpy as np

    from .build_index import build_index

    print("\n" + "=" * 60)
    print("  STEP: Build FAISS Index")
    print("=" * 60)

    embeddings_dir = Path(config.get("embeddings_dir", "embeddings"))
    corpus_embeddings = np.load(str(embeddings_dir / "corpus.npy"))
    build_index(corpus_embeddings, config)


def step_retrieve(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Step 6: Perform retrieval."""
    import json

    import numpy as np

    from .build_index import build_index
    from .retrieve import retrieve

    print("\n" + "=" * 60)
    print("  STEP: Retrieve")
    print("=" * 60)

    embeddings_dir = Path(config.get("embeddings_dir", "embeddings"))

    # Load embeddings and IDs
    corpus_embeddings = np.load(str(embeddings_dir / "corpus.npy"))
    query_embeddings = np.load(str(embeddings_dir / "query.npy"))

    with open(embeddings_dir / "corpus_ids.json", "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)
    with open(embeddings_dir / "query_ids.json", "r", encoding="utf-8") as f:
        query_ids = json.load(f)

    # Build or load index
    index = build_index(corpus_embeddings, config)

    # Retrieve
    return retrieve(index, query_embeddings, query_ids, corpus_ids, config)


def step_evaluate(config: Dict[str, Any]) -> Dict[str, float]:
    """Step 7: Evaluate retrieval results."""
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

    # Save metrics
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "results.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved evaluation results to {metrics_path}")

    return metrics


def step_prepare_data(config: Dict[str, Any]) -> None:
    """Prepare contrastive training data with hard negative mining."""
    from .prepare_training_data import prepare_training_data

    print("\n" + "=" * 60)
    print("  STEP: Prepare Training Data")
    print("=" * 60)

    training_data = prepare_training_data(config)
    print(f"[INFO] Total training examples: {len(training_data)}")


def step_train_adapter(config: Dict[str, Any]) -> None:
    """Train LoRA adapter on BGE-M3."""
    from .train_adapter import train_adapter

    print("\n" + "=" * 60)
    print("  STEP: Train LoRA Adapter")
    print("=" * 60)

    output_dir = train_adapter(config)
    print(f"[INFO] Adapter saved to: {output_dir}")
    print(
        "\n[TIP] To use the trained adapter for retrieval, update config.yaml:\n"
        "  use_adapter: true\n"
        f"  adapter_path: {output_dir}\n"
        "Then re-run: python -m src.main"
    )


def run_full_pipeline(config: Dict[str, Any]) -> None:
    """Run the complete pipeline from data loading to evaluation."""
    from .build_index import build_index
    from .encode_corpus import encode_corpus
    from .encode_query import encode_queries
    from .load_dataset import load_all
    from .metrics import evaluate_all, print_metrics
    from .model_loader import load_model
    from .retrieve import retrieve

    use_adapter = config.get("use_adapter", False)
    mode = "BGE-M3 + LoRA Adapter" if use_adapter else "BGE-M3 Baseline"

    print("\n" + "#" * 60)
    print(f"  {mode} Dense Retrieval Pipeline")
    print("#" * 60)

    # Step 1-2: Load data
    print("\n" + "=" * 60)
    print("  STEP 1-2: Load Dataset")
    print("=" * 60)
    corpus, queries, qrels = load_all(config)

    # Step 3: Load model and encode corpus
    print("\n" + "=" * 60)
    print("  STEP 3: Encode Corpus")
    print("=" * 60)
    encoder = load_model(config)
    corpus_embeddings = encode_corpus(corpus, encoder, config)

    # Step 4: Encode queries
    print("\n" + "=" * 60)
    print("  STEP 4: Encode Queries")
    print("=" * 60)
    query_embeddings = encode_queries(queries, encoder, config)

    # Step 5: Build FAISS index
    print("\n" + "=" * 60)
    print("  STEP 5: Build FAISS Index")
    print("=" * 60)
    index = build_index(corpus_embeddings, config)

    # Step 6: Retrieve
    print("\n" + "=" * 60)
    print("  STEP 6: Retrieve")
    print("=" * 60)
    query_ids = [q["query_id"] for q in queries]
    corpus_ids = [doc["doc_id"] for doc in corpus]
    results = retrieve(index, query_embeddings, query_ids, corpus_ids, config)

    # Step 7: Evaluate
    print("\n" + "=" * 60)
    print("  STEP 7: Evaluate")
    print("=" * 60)
    top_k = config.get("top_k", 10)
    metrics = evaluate_all(results, qrels, top_k)
    print_metrics(metrics)

    # Save metrics
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
        description="BGE-M3 + LoRA Adapter Dense Retrieval Pipeline"
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
        "prepare_data": step_prepare_data,
        "train_adapter": step_train_adapter,
        "encode_corpus": step_encode_corpus,
        "encode_query": step_encode_query,
        "build_index": step_build_index,
        "retrieve": step_retrieve,
        "evaluate": step_evaluate,
    }

    if args.step is None:
        run_full_pipeline(config)
    else:
        step_map[args.step](config)


if __name__ == "__main__":
    main()
