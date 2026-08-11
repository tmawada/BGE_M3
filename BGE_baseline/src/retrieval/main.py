"""CLI for interactive semantic retrieval using BGE-M3 and FAISS index."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from .faiss_loader import load_index_and_corpus
from .retriever import encode_query, search_index
from .formatter import format_and_print, save_results_json

from ..model_loader import load_model


LOGGER = logging.getLogger("src.retrieval")


def prompt_query() -> str:
    return input("Enter Query:\n").strip()


def run_retrieval(config: Dict[str, Any], query: str) -> Dict[str, Any]:
    LOGGER.info("Loading FAISS index and corpus metadata")
    index, corpus = load_index_and_corpus(config)
    LOGGER.info("Loading model")
    model = load_model(config)

    if not query:
        raise ValueError("Empty query provided")

    top_k = int(config.get("top_k", 5))

    # Encode query
    qvec, emb_time = encode_query(model, query, config)

    # Search
    start = time.perf_counter()
    ids, scores = search_index(index, qvec, top_k=top_k)
    search_time = time.perf_counter() - start

    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx < 0 or idx >= len(corpus):
            continue
        doc = corpus[idx]
        results.append(
            {
                "rank": rank,
                "document_id": doc.get("doc_id") or doc.get("id"),
                "score": float(score),
                "title": doc.get("title") or doc.get("document_title") or doc.get("name") or "",
                "text": doc.get("passage", doc.get("text", "")),
            }
        )

    total_time = emb_time + search_time

    timings = {"embedding_time": emb_time, "search_time": search_time, "total_time": total_time}

    return {
        "model": config.get("model_name"),
        "query": query,
        "results": results,
        "timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive retrieval using trained BGE-M3 and FAISS index")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--save", action="store_true", help="Save retrieval results to outputs/")
    args = parser.parse_args()

    import yaml
    from pathlib import Path

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if args.query:
        query = args.query
    else:
        query = prompt_query()

    if not query:
        print("Empty query. Exiting.")
        return

    try:
        out = run_retrieval(config, query)
    except Exception as e:
        LOGGER.error("Retrieval failed: %s", e)
        raise

    format_and_print(out["model"], out["query"], out["results"], config, out["timings"]) 

    if args.save:
        out_dir = Path(config.get("outputs_dir", "outputs"))
        save_results_json(out_dir / "retrieval_result.json", out)
        LOGGER.info("Saved retrieval result to %s", out_dir / "retrieval_result.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()