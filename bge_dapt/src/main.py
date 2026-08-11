"""
Main pipeline orchestrator for the BGE + DAPT project.

Runs the two-stage pipeline (DAPT MLM pretraining, then MIRACL retrieval
fine-tuning) plus evaluation. Individual stages can be run via CLI arguments.

Usage:
    # Run full pipeline
    python -m src.main

    # Run individual steps
    python -m src.main --step load_data
    python -m src.main --step train_dapt
    python -m src.main --step prepare_retrieval
    python -m src.main --step train_retrieval
    python -m src.main --step evaluate

    # Custom config file
    python -m src.main --config config/custom.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

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
        config = yaml.safe_load(f) or {}

    print(f"[INFO] Loaded configuration from {config_path}")
    return config


def _section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Merge a nested section into the top-level configuration.

    General keys (model dirs, device, seed, output paths) stay accessible
    while the section-specific keys override any shared defaults.

    Args:
        config: Full configuration dictionary.
        name: Section name, e.g. 'dapt' or 'retrieval'.

    Returns:
        Merged configuration dictionary.
    """
    merged = {key: value for key, value in config.items() if key not in ("dapt", "retrieval")}
    merged.update(config.get(name) or {})
    return merged


def step_load_data(config: Dict[str, Any]) -> None:
    """Step 1: Prepare the DAPT MLM dataset from the review corpus."""
    from .load_dataset import load_reviews

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


def step_prepare_retrieval(config: Dict[str, Any]) -> None:
    """Step 3: Build retrieval training pairs from MIRACL."""
    from .prepare_retrieval_dataset import prepare_pairs

    print("\n" + "=" * 60)
    print("  STEP: Prepare Retrieval Data")
    print("=" * 60)
    prepare_pairs(_section(config, "retrieval"))


def step_train_retrieval(config: Dict[str, Any]) -> Dict[str, Any]:
    """Step 4: Retrieval fine-tuning (MIRACL, MNRL)."""
    from .train_retrieval import train_retrieval

    print("\n" + "#" * 60)
    print("  STAGE 2: Retrieval Fine-tuning")
    print("#" * 60)
    return train_retrieval(_section(config, "retrieval"))


def step_evaluate(config: Dict[str, Any]) -> Dict[str, float]:
    """Step 5: Evaluate the retriever and write the final report."""
    from .evaluate import run_evaluation

    print("\n" + "#" * 60)
    print("  EVALUATION")
    print("#" * 60)
    return run_evaluation(_section(config, "retrieval"))


def run_full_pipeline(config: Dict[str, Any]) -> None:
    """Run the complete two-stage pipeline."""
    print("\n" + "#" * 60)
    print("  BGE + DAPT Pipeline")
    print("#" * 60)

    step_load_data(config)
    step_train_dapt(config)
    step_prepare_retrieval(config)
    step_train_retrieval(config)
    step_evaluate(config)

    print("\n" + "#" * 60)
    print("  Pipeline Complete!")
    print("#" * 60)


def main() -> None:
    """Entry point for the pipeline CLI."""
    parser = argparse.ArgumentParser(description="BGE + DAPT Pipeline")
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
            "prepare_retrieval",
            "train_retrieval",
            "evaluate",
        ],
        help="Run a specific pipeline step instead of the full pipeline.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    try:
        if args.step is None:
            run_full_pipeline(config)
        elif args.step == "load_data":
            step_load_data(config)
        elif args.step == "train_dapt":
            step_train_dapt(config)
        elif args.step == "prepare_retrieval":
            step_prepare_retrieval(config)
        elif args.step == "train_retrieval":
            step_train_retrieval(config)
        elif args.step == "evaluate":
            step_evaluate(config)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
