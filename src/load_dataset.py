"""
Dataset loader for the MIRACL dataset.

Loads corpus, queries, and relevance judgments from local files:
- corpus.jsonl for the corpus
- query.tsv for queries
- qrels.tsv for relevance judgments

The loaded data is normalized into the project's standard JSON format.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List


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


def _resolve_data_path(config: Dict[str, Any], key: str, default_name: str) -> Path:
    """Resolve a data file path from config or the default data directory."""
    base_dir = Path(config.get("data_dir", "data"))
    configured = config.get(key)
    if configured:
        return Path(configured)
    return base_dir / default_name


def _load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    if not filepath.exists():
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    records: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {filepath}: {e}") from e
    return records


def _load_tsv(filepath: Path) -> List[List[str]]:
    """Load a TSV file into a list of rows."""
    if not filepath.exists():
        raise FileNotFoundError(f"TSV file not found: {filepath}")

    rows: List[List[str]] = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row:
                rows.append(row)
    return rows


def _extract_doc_id(entry: Dict[str, Any]) -> str:
    """Extract a document ID from a corpus entry using common field names."""
    for key in ("doc_id", "docid", "id", "_id"):
        if key in entry and entry[key] is not None:
            return str(entry[key])
    raise KeyError(f"Corpus record is missing a document id field: {entry}")


def _extract_passage(entry: Dict[str, Any]) -> str:
    """Extract passage text from a corpus entry using common field names."""
    for key in ("passage", "text", "contents", "body"):
        if key in entry and entry[key] is not None:
            return str(entry[key])
    raise KeyError(f"Corpus record is missing a text field: {entry}")


def _extract_query_row(row: List[str]) -> tuple[str, str]:
    """Parse a query TSV row as query_id and query text."""
    if len(row) < 2:
        raise ValueError(f"Invalid query TSV row (expected 2 columns): {row}")
    return str(row[0]), str(row[1])


def _extract_qrels_row(row: List[str]) -> tuple[str, str, int]:
    """Parse a qrels TSV row as query_id, doc_id, and relevance score."""
    if len(row) < 4:
        raise ValueError(f"Invalid qrels TSV row (expected 4 columns): {row}")
    query_id = str(row[0])
    doc_id = str(row[2])
    relevance = int(row[3])
    return query_id, doc_id, relevance


def load_corpus(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load the MIRACL corpus from a local JSONL file and save to disk.

    Each document is stored as ``{"doc_id": str, "title": str, "passage": str}``.

    Args:
        config: Configuration dictionary with 'dataset_name', 'language',
            'split', and 'data_dir' keys.

    Returns:
        List of corpus documents.
    """
    base_dir = Path(config.get("data_dir", "data"))
    corpus_path = base_dir / "corpus.json"

    if corpus_path.exists():
        print(f"[INFO] Corpus already exists at {corpus_path}, loading from disk.")
        with open(corpus_path, "r", encoding="utf-8") as f:
            return json.load(f)

    corpus_file = _resolve_data_path(config, "corpus_path", "corpus.jsonl")
    records = _load_jsonl(corpus_file)

    seen_ids: set = set()
    corpus: List[Dict[str, str]] = []

    for entry in records:
        docid = _extract_doc_id(entry)
        if docid in seen_ids:
            continue
        corpus.append({
            "doc_id": docid,
            "title": str(entry.get("title", "")),
            "passage": _extract_passage(entry),
        })
        seen_ids.add(docid)

    _save_json(corpus, corpus_path)
    return corpus


def load_queries(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load MIRACL queries from a local TSV file and save to disk.

    Args:
        config: Configuration dictionary with 'dataset_name', 'language',
            'split', and 'data_dir' keys.

    Returns:
        List of query dictionaries ``{"query_id": str, "query": str}``.
    """
    base_dir = Path(config.get("data_dir", "data"))
    queries_path = base_dir / "queries.json"

    if queries_path.exists():
        print(f"[INFO] Queries already exist at {queries_path}, loading from disk.")
        with open(queries_path, "r", encoding="utf-8") as f:
            return json.load(f)

    query_file = _resolve_data_path(config, "query_path", "query.tsv")
    rows = _load_tsv(query_file)

    queries: List[Dict[str, str]] = []
    seen_ids: set = set()

    for row in rows:
        qid, query_text = _extract_query_row(row)
        if qid not in seen_ids:
            queries.append({
                "query_id": qid,
                "query": query_text,
            })
            seen_ids.add(qid)

    _save_json(queries, queries_path)
    return queries


def load_qrels(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load relevance judgments from a local TSV file and save to disk.

    Expects qrels TSV rows in the form:
    query_id\tQ0\tdoc_id\trelevance

    Args:
        config: Configuration dictionary with 'dataset_name', 'language',
            'split', and 'data_dir' keys.

    Returns:
        List of qrel dictionaries ``{"query_id": str, "relevant_doc": str}``.
    """
    base_dir = Path(config.get("data_dir", "data"))
    qrels_path = base_dir / "qrels.json"

    if qrels_path.exists():
        print(f"[INFO] Qrels already exist at {qrels_path}, loading from disk.")
        with open(qrels_path, "r", encoding="utf-8") as f:
            return json.load(f)

    qrels_file = _resolve_data_path(config, "qrels_path", "qrels.tsv")
    rows = _load_tsv(qrels_file)

    qrels: List[Dict[str, Any]] = []

    for row in rows:
        qid, doc_id, relevance = _extract_qrels_row(row)
        if relevance > 0:
            qrels.append({
                "query_id": qid,
                "relevant_doc": doc_id,
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
