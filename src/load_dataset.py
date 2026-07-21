"""
Dataset loader for the Indonesian MIRACL dataset.

Downloads corpus, queries, and relevance judgments from HuggingFace
and saves them in the project's standard JSON format.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


def _save_json(data: List[Dict[str, Any]], filepath: Path) -> None:
    """Save a list of dicts as a JSON file.

    Args:
        data: List of dictionaries to serialize.
        filepath: Output file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(data)} records to {filepath}")


def load_corpus(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load the MIRACL corpus from HuggingFace and save to disk.

    Each document is stored as {"doc_id": str, "text": str} where
    text is the concatenation of the article title and passage text.

    Args:
        config: Configuration dictionary with 'corpus_name', 'language',
            and 'data_dir' keys.

    Returns:
        List of corpus documents.
    """
    base_dir = Path(config.get("data_dir", "data"))
    corpus_path = base_dir / "corpus.json"

    if corpus_path.exists():
        print(f"[INFO] Corpus already exists at {corpus_path}, loading from disk.")
        with open(corpus_path, "r", encoding="utf-8") as f:
            return json.load(f)

    from datasets import load_dataset as hf_load_dataset

    corpus_name = config.get("corpus_name", "miracl/miracl-corpus")
    language = config.get("language", "id")

    print(f"[INFO] Downloading corpus from '{corpus_name}' (language={language})...")
    dataset = hf_load_dataset(corpus_name, language, split="train")

    corpus: List[Dict[str, str]] = []
    for item in tqdm(dataset, desc="Processing corpus"):
        title = item.get("title", "")
        text = item.get("text", "")
        combined = f"{title}: {text}" if title else text

        corpus.append({
            "doc_id": item["docid"],
            "text": combined,
        })

    _save_json(corpus, corpus_path)
    return corpus


def load_queries(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load MIRACL queries from HuggingFace and save to disk.

    Args:
        config: Configuration dictionary with 'dataset_name', 'language',
            and 'data_dir' keys.

    Returns:
        List of query dictionaries {"query_id": str, "query": str}.
    """
    base_dir = Path(config.get("data_dir", "data"))
    queries_path = base_dir / "queries.json"

    if queries_path.exists():
        print(f"[INFO] Queries already exist at {queries_path}, loading from disk.")
        with open(queries_path, "r", encoding="utf-8") as f:
            return json.load(f)

    from datasets import load_dataset as hf_load_dataset

    dataset_name = config.get("dataset_name", "miracl/miracl")
    language = config.get("language", "id")

    print(f"[INFO] Downloading queries from '{dataset_name}' (language={language})...")
    dataset = hf_load_dataset(dataset_name, language, split="dev")

    queries: List[Dict[str, str]] = []
    seen_ids: set = set()

    for item in tqdm(dataset, desc="Processing queries"):
        qid = item["query_id"]
        if qid not in seen_ids:
            queries.append({
                "query_id": qid,
                "query": item["query"],
            })
            seen_ids.add(qid)

    _save_json(queries, queries_path)
    return queries


def load_qrels(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load relevance judgments from MIRACL and save to disk.

    Extracts positive passages from the dev split as binary relevance
    judgments (relevance=1).

    Args:
        config: Configuration dictionary with 'dataset_name', 'language',
            and 'data_dir' keys.

    Returns:
        List of qrel dictionaries {"query_id": str, "doc_id": str, "relevance": int}.
    """
    base_dir = Path(config.get("data_dir", "data"))
    qrels_path = base_dir / "qrels.json"

    if qrels_path.exists():
        print(f"[INFO] Qrels already exist at {qrels_path}, loading from disk.")
        with open(qrels_path, "r", encoding="utf-8") as f:
            return json.load(f)

    from datasets import load_dataset as hf_load_dataset

    dataset_name = config.get("dataset_name", "miracl/miracl")
    language = config.get("language", "id")

    print(f"[INFO] Downloading qrels from '{dataset_name}' (language={language})...")
    dataset = hf_load_dataset(dataset_name, language, split="dev")

    qrels: List[Dict[str, Any]] = []

    for item in tqdm(dataset, desc="Processing qrels"):
        qid = item["query_id"]
        positive_passages = item.get("positive_passages", [])
        for passage in positive_passages:
            qrels.append({
                "query_id": qid,
                "doc_id": passage["docid"],
                "relevance": 1,
            })

    _save_json(qrels, qrels_path)
    return qrels


def load_all(config: Dict[str, Any]) -> tuple:
    """Load corpus, queries, and qrels.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Tuple of (corpus, queries, qrels).
    """
    corpus = load_corpus(config)
    queries = load_queries(config)
    qrels = load_qrels(config)
    return corpus, queries, qrels
