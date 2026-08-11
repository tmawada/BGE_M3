"""
MIRACL Indonesian dataset loader for Stage 2 retrieval fine-tuning.

Normalizes MIRACL files into the project's standard structure:

- corpus:  {"doc_id", "title", "text"}  (title and text combined)
- queries: {"query_id", "query"}
- qrels:   {"query_id", "relevant_doc"} (only positive relevance)

Supported file formats: .json (array), .jsonl (line-delimited), and .tsv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return records


def _load_tsv_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row:
                rows.append(row)
    return rows


def _read_records(path: Path) -> List[Dict[str, Any]]:
    """Read a corpus/queries/qrels file, guessing its format by extension."""
    if not path.exists():
        raise FileNotFoundError(f"MIRACL data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".json":
        return _load_json_array(path)
    if suffix == ".tsv":
        return [{"row": row} for row in _load_tsv_rows(path)]
    raise ValueError(f"Unsupported data file format: {path.suffix}")


def load_corpus(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load the MIRACL corpus and combine title + text into one document.

    Args:
        config: Configuration dictionary with 'corpus_path'.

    Returns:
        List of ``{"doc_id": str, "title": str, "text": str}``.
    """
    path = Path(config.get("corpus_path", "data/corpus.json"))
    print(f"[INFO] Loading corpus from {path} ...")
    records = _read_records(path)

    corpus: List[Dict[str, str]] = []
    seen: set = set()
    for entry in records:
        if "row" in entry:
            row = entry["row"]
            if len(row) < 2:
                continue
            doc_id, text = str(row[0]), str(row[1])
            title = ""
        else:
            doc_id = str(entry.get("doc_id") or entry.get("docid") or entry.get("id") or entry.get("_id") or "")
            title = str(entry.get("title", "") or "")
            text = str(entry.get("text") or entry.get("passage") or entry.get("contents") or entry.get("body") or "")
        if not doc_id or not text:
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        combined = f"{title}\n\n{text}" if title else text
        corpus.append({"doc_id": doc_id, "title": title, "text": combined})

    print(f"[INFO] Loaded {len(corpus)} corpus documents.")
    return corpus


def load_queries(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load MIRACL queries.

    Args:
        config: Configuration dictionary with 'queries_path'.

    Returns:
        List of ``{"query_id": str, "query": str}``.
    """
    path = Path(config.get("queries_path", "data/queries.json"))
    print(f"[INFO] Loading queries from {path} ...")
    records = _read_records(path)

    queries: List[Dict[str, str]] = []
    seen: set = set()
    for entry in records:
        if "row" in entry:
            row = entry["row"]
            if len(row) < 2:
                continue
            query_id, query_text = str(row[0]), str(row[1])
        else:
            query_id = str(entry.get("query_id") or entry.get("qid") or entry.get("id") or "")
            query_text = str(entry.get("query", "") or "")
        if not query_id or not query_text:
            continue
        if query_id in seen:
            continue
        seen.add(query_id)
        queries.append({"query_id": query_id, "query": query_text})

    print(f"[INFO] Loaded {len(queries)} queries.")
    return queries


def load_qrels(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load MIRACL relevance judgments, keeping only positive relevance.

    Args:
        config: Configuration dictionary with 'qrels_path'.

    Returns:
        List of ``{"query_id": str, "relevant_doc": str}``.
    """
    path = Path(config.get("qrels_path", "data/qrels.tsv"))
    print(f"[INFO] Loading qrels from {path} ...")
    records = _read_records(path)

    qrels: List[Dict[str, str]] = []
    for entry in records:
        if "row" in entry:
            row = entry["row"]
            if len(row) < 4:
                continue
            query_id, doc_id, relevance = str(row[0]), str(row[2]), int(row[3])
        else:
            query_id = str(entry.get("query_id", "") or "")
            doc_id = str(entry.get("relevant_doc", "") or entry.get("doc_id", "") or entry.get("docid", "") or "")
            relevance = int(entry.get("relevance", 1) or 1)
        if not query_id or not doc_id:
            continue
        if relevance > 0:
            qrels.append({"query_id": query_id, "relevant_doc": doc_id})

    print(f"[INFO] Loaded {len(qrels)} positive relevance judgments.")
    return qrels


def load_all(config: Dict[str, Any]) -> tuple:
    """Load corpus, queries, and qrels.

    Returns:
        Tuple of (corpus, queries, qrels).
    """
    corpus = load_corpus(config)
    queries = load_queries(config)
    qrels = load_qrels(config)
    return corpus, queries, qrels
