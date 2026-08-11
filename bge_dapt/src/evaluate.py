"""
Stage 2 evaluation and final report generation.

Runs dense retrieval over the MIRACL dev corpus using the retriever
checkpoint and writes a report consistent with the baseline experiment:

    outputs/evaluation.json
    outputs/evaluation.csv
    outputs/training_log.txt
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .build_index import build_index
from .encode_corpus import encode_corpus
from .encode_query import encode_queries
from .load_miracl import load_all
from .metrics import evaluate_all, print_metrics
from .model_loader import DenseEncoder
from .retrieve import retrieve


def evaluate_retrieval(
    config: Dict[str, Any],
    encoder: Optional[DenseEncoder] = None,
    model: Optional[torch.nn.Module] = None,
    tokenizer=None,
    use_cached_embeddings: Optional[bool] = None,
    tag: str = "",
) -> Dict[str, float]:
    """Encode, index, retrieve, and evaluate on the MIRACL dev set.

    Args:
        config: Configuration dictionary.
        encoder: Optional pre-built DenseEncoder (reused if provided).
        model: Optional in-memory backbone (evaluation during training).
        tokenizer: Optional tokenizer matching ``model``.
        use_cached_embeddings: Override for the config flag.
        tag: Optional suffix for logging (e.g. 'epoch_1').

    Returns:
        Dictionary of metric names to scores.
    """
    if use_cached_embeddings is None:
        use_cached_embeddings = bool(config.get("use_cached_embeddings", False))

    force = not use_cached_embeddings

    corpus, queries, qrels = load_all(config)

    if encoder is None:
        if model is not None:
            encoder = DenseEncoder(
                model=model,
                tokenizer=tokenizer,
                device=config.get("device", "cuda"),
                use_fp16=config.get("use_fp16", True),
            )
        else:
            encoder = DenseEncoder(
                model_path=config.get("retriever_model_dir", "models/bge-m3-dapt-retriever"),
                device=config.get("device", "cuda"),
                use_fp16=config.get("use_fp16", True),
            )

    batch_size = int(config.get("batch_size", 32))
    max_length = int(config.get("max_length", 256))
    top_k = int(config.get("top_k", 10))

    print("\n" + "=" * 60)
    print(f"  RETRIEVAL EVALUATION {tag and f'({tag})' or ''}".rstrip())
    print("=" * 60)

    corpus_embeddings = encode_corpus(corpus, encoder, config, force=force)
    index = build_index(corpus_embeddings, config, force=force)

    query_embeddings = encode_queries(queries, encoder, config, force=force)

    query_ids = [q["query_id"] for q in queries]
    corpus_ids = [d["doc_id"] for d in corpus]
    results = retrieve(index, query_embeddings, query_ids, corpus_ids, config)

    metrics = evaluate_all(results, qrels, top_k)
    print_metrics(metrics)
    return metrics


def _load_log(config: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    path = Path(config.get("outputs_dir", "outputs")) / name
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_report(config: Dict[str, Any], retrieval_metrics: Dict[str, float]) -> None:
    """Write the final evaluation report (json/csv/txt)."""
    outputs_dir = Path(config.get("outputs_dir", "outputs"))
    dapt_log = _load_log(config, "training_log_dapt.json")
    retrieval_log = _load_log(config, "training_log_retrieval.json")

    training_config = {
        "model_name": config.get("base_model_name", "BAAI/bge-m3"),
        "dapt_checkpoint_path": config.get("dapt_model_dir", "models/bge-m3-dapt"),
        "retriever_checkpoint_path": config.get("retriever_model_dir", "models/bge-m3-dapt-retriever"),
        "learning_rate": config.get("learning_rate", 2e-5),
        "batch_size": config.get("batch_size", 32),
        "optimizer": "adamw_torch",
        "scheduler": "linear",
        "max_length": config.get("max_length", 256),
        "fp16": config.get("fp16", True),
        "seed": config.get("seed", 42),
    }

    report = {
        "training_summary": {
            "dapt": dapt_log,
            "retrieval": retrieval_log,
        },
        "retrieval_metrics": retrieval_metrics,
        "training_config": training_config,
    }

    outputs_dir.mkdir(parents=True, exist_ok=True)

    # evaluation.json
    json_path = outputs_dir / "evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved evaluation report to {json_path}")

    # evaluation.csv
    csv_path = outputs_dir / "evaluation.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, value in retrieval_metrics.items():
            writer.writerow([name, f"{value:.6f}"])
    print(f"[INFO] Saved evaluation CSV to {csv_path}")

    # training_log.txt
    txt_path = outputs_dir / "training_log.txt"
    _write_training_log(txt_path, report)
    print(f"[INFO] Saved training log to {txt_path}")


def _write_training_log(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = ["=" * 60, "  BGE + DAPT — Training Log", "=" * 60]

    for stage_name, log in report["training_summary"].items():
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"STAGE: {stage_name.upper()}")
        lines.append("-" * 60)
        if not log:
            lines.append("(not found)")
            continue
        for key, value in log.items():
            if isinstance(value, (list, dict)):
                lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("RETRIEVAL METRICS")
    lines.append("-" * 60)
    for name, value in report["retrieval_metrics"].items():
        lines.append(f"  {name:<12}: {value:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("TRAINING CONFIGURATION")
    lines.append("-" * 60)
    for key, value in report["training_config"].items():
        lines.append(f"  {key:<24}: {value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(config: Dict[str, Any]) -> Dict[str, float]:
    """Full evaluation entry point: evaluate + write the final report."""
    start = time.perf_counter()
    metrics = evaluate_retrieval(config)
    save_report(config, metrics)
    print(f"[INFO] Total evaluation time: {time.perf_counter() - start:.2f} sec")
    return metrics
