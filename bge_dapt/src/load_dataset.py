"""
DAPT dataset loader.

Loads the Indonesian e-commerce review corpus and builds Masked Language
Modeling samples by merging the 'name' and 'review' fields into a single
text sample. The 'rating' column is ignored (per the DAPT specification).

Example merged sample::

    Xiaomi Redmi Note

    barangnya bagus bgt, recommended
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _merge_fields(row: Dict[str, str], columns: List[str]) -> str:
    """Merge selected CSV fields into a single text sample."""
    parts: List[str] = []
    for col in columns:
        value = str(row.get(col, "") or "").strip()
        if value:
            parts.append(value)
    return "\n\n".join(parts).strip()


def load_reviews(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load e-commerce reviews and save merged MLM samples to disk.

    Args:
        config: Configuration dictionary. The DAPT section should provide
            'data_path', 'dataset_path', 'text_columns', and optional
            'max_samples'.

    Returns:
        List of ``{"text": str}`` MLM samples.
    """
    data_path = Path(config.get("data_path", "data/master.csv"))
    dataset_path = Path(config.get("dataset_path", "data/dapt_dataset.json"))
    text_columns: List[str] = config.get("text_columns", ["name", "review"])
    max_samples: Optional[int] = config.get("max_samples")

    if not data_path.exists():
        raise FileNotFoundError(f"DAPT data file not found: {data_path}")

    print(f"[INFO] Reading review corpus from {data_path} ...")
    samples: List[Dict[str, str]] = []
    with open(data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = _merge_fields(row, text_columns)
            if not text:
                continue
            samples.append({"text": text})
            if max_samples is not None and len(samples) >= max_samples:
                break

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Dataset size        : {len(samples)} review rows")
    print(f"[INFO] MLM samples         : {len(samples)}")
    print(f"[INFO] Saved MLM dataset to {dataset_path}")
    return samples
