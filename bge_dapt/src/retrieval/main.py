"""CLI for interactive semantic retrieval using a trained BGE + DAPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ..model_loader import load_encoder
from .retriever import run_retrieval
from .utils import format_and_print, load_config, save_results_json, setup_logging


def prompt_query() -> str:
    """Prompt the user for a query."""
    return input("\nEnter Query (or 'exit'/'quit' to stop):\n").strip()


def main() -> None:
    """Interactive retrieval CLI supporting multiple queries per session."""
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Interactive retrieval using a trained BGE + DAPT checkpoint and FAISS index"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--save", action="store_true", help="Save retrieval results to outputs/")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("outputs_dir", "outputs"))

    logger.info("Loading model: %s", config.get("retriever_model_dir", "models/bge-m3-dapt-retriever"))
    encoder = load_encoder(config)

    def handle(query: str) -> None:
        if not query:
            return
        out = run_retrieval(config, query, encoder)
        format_and_print(out["model"], out["query"], out["results"], config, out["timings"])
        if args.save:
            save_results_json(output_dir / "retrieval_result.json", out)
            logger.info("Saved retrieval result to %s", output_dir / "retrieval_result.json")

    if args.query:
        handle(args.query)
        return

    print("Interactive retrieval session started.")
    while True:
        try:
            query = prompt_query()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Session ended.")
            break
        if query.lower() in ("exit", "quit", "q"):
            print("[INFO] Session ended.")
            break
        handle(query)


if __name__ == "__main__":
    main()
