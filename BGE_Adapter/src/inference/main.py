"""Standalone inference CLI for comparing two Indonesian queries."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict

from .embedding import embedding_statistics, generate_query_embeddings
from .model import load_model
from .similarity import cosine_similarity, dot_product, euclidean_distance
from .utils import (
    format_vector_preview,
    load_config,
    print_report,
    resolve_output_dir,
    save_json,
    save_numpy_vector,
    setup_logging,
)


def _prompt_for_query(label: str) -> str:
    """Prompt the user for a query if none was supplied via CLI."""
    return input(f"Enter {label}:\n").strip()


def run_inference(config: Dict[str, Any], query_1: str, query_2: str) -> Dict[str, Any]:
    """Run embedding inference and return the computed results."""
    logger = setup_logging()

    model_name = config.get("model_name", "BAAI/bge-m3")
    logger.info("Loading model: %s", model_name)
    model_start = time.perf_counter()
    encoder = load_model(config)
    model_time = time.perf_counter() - model_start

    device = getattr(encoder, "device", config.get("device", "cpu"))

    logger.info("Generating embeddings")
    query_1_vector, query_2_vector, embedding_time = generate_query_embeddings(
        encoder=encoder,
        query_1=query_1,
        query_2=query_2,
        config=config,
    )

    similarity_start = time.perf_counter()
    cosine = cosine_similarity(query_1_vector, query_2_vector)
    euclidean = euclidean_distance(query_1_vector, query_2_vector)
    dot = dot_product(query_1_vector, query_2_vector)
    similarity_time = time.perf_counter() - similarity_start

    query_1_stats = embedding_statistics(query_1_vector)
    query_2_stats = embedding_statistics(query_2_vector)
    inference_time = model_time + embedding_time + similarity_time

    payload: Dict[str, Any] = {
        "model": model_name,
        "query_1": query_1,
        "query_2": query_2,
        "query_1_vector": query_1_vector,
        "query_2_vector": query_2_vector,
        "query_1_preview": format_vector_preview(query_1_vector),
        "query_2_preview": format_vector_preview(query_2_vector),
        "query_1_stats": query_1_stats,
        "query_2_stats": query_2_stats,
        "embedding_dimension": int(query_1_stats["dimension"]),
        "cosine_similarity": cosine,
        "euclidean_distance": euclidean,
        "dot_product": dot,
        "device": str(device).upper(),
        "model_load_time": model_time,
        "embedding_time": embedding_time,
        "similarity_time": similarity_time,
        "inference_time": inference_time,
    }

    save_embedding = bool(config.get("save_embedding", True))
    if save_embedding:
        output_dir = resolve_output_dir(config)
        save_numpy_vector(output_dir / "embedding_query1.npy", query_1_vector)
        save_numpy_vector(output_dir / "embedding_query2.npy", query_2_vector)
        save_json(
            output_dir / "result.json",
            {
                "model": model_name,
                "query_1": query_1,
                "query_2": query_2,
                "embedding_dimension": int(query_1_stats["dimension"]),
                "cosine_similarity": cosine,
                "euclidean_distance": euclidean,
                "dot_product": dot,
                "device": str(device).upper(),
                "model_load_time_sec": round(model_time, 6),
                "embedding_time_sec": round(embedding_time, 6),
                "similarity_time_sec": round(similarity_time, 6),
                "inference_time_sec": round(inference_time, 6),
                "query_1_stats": query_1_stats,
                "query_2_stats": query_2_stats,
            },
        )
        logger.info("Saved inference outputs to %s", output_dir)

    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BGE-M3 inference utility for comparing two Indonesian queries"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--query1",
        type=str,
        default=None,
        help="First query text",
    )
    parser.add_argument(
        "--query2",
        type=str,
        default=None,
        help="Second query text",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    query_1 = args.query1 or _prompt_for_query("Query 1")
    query_2 = args.query2 or _prompt_for_query("Query 2")

    if not query_1 or not query_2:
        raise ValueError("Both Query 1 and Query 2 must be provided.")

    result = run_inference(config, query_1, query_2)
    print_report(result)


if __name__ == "__main__":
    main()