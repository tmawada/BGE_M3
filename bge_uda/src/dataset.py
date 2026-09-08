"""Load MIRACL files + your formal/informal pairs.

Expected files inside data_dir:
  corpus.jsonl          -> one JSON per line with doc id + passage
  query.tsv             -> qid <TAB> formal query (MIRACL formal)
  qrels.tsv             -> qid <TAB> Q0 <TAB> docid <TAB> relevance
  train_pairs.csv       -> query_id,formal,informal (your 4072 pairs)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _data_dir(config: Dict[str, Any]) -> Path:
    """Resolve data_dir relative to repo root."""
    # Allow data_dir: ../BGE_M3/data so you reuse existing files.
    p = Path(config.get("data_dir", "data"))
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        p = (repo_root / p).resolve()
    return p


def _get_id(entry: Dict[str, Any]) -> str:
    """Pick doc id from whichever field name the JSONL uses."""
    # MIRACL dumps vary: doc_id / docid / id / _id all appear in the wild.
    for key in ("doc_id", "docid", "id", "_id"):
        if entry.get(key) is not None:
            return str(entry[key])
    raise KeyError(f"No doc id field in record: {entry}")


def _get_text(entry: Dict[str, Any]) -> str:
    """Pick passage text from whichever field name the JSONL uses."""
    for key in ("passage", "text", "contents", "body"):
        if entry.get(key) is not None:
            return str(entry[key])
    raise KeyError(f"No passage text field in record: {entry}")


def load_corpus(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load corpus -> [{"doc_id","title","passage"}]. Uses cache corpus.json."""
    data_dir = _data_dir(config)
    cache = data_dir / "corpus.json"
    # Reuse cached parse to skip re-reading large JSONL every run.
    if cache.exists():
        print(f"[INFO] Loading cached corpus: {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))
    # Fallback filename comes from config so you can rename files freely.
    src = data_dir / config.get("corpus_jsonl", "corpus.jsonl")
    if not src.exists():
        raise FileNotFoundError(f"Corpus JSONL not found: {src}")
    out, seen = [], set()
    # JSONL = one JSON object per line; skip blanks.
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        doc_id = _get_id(entry)
        if doc_id in seen:  # dedupe keeps FAISS positions stable
            continue
        seen.add(doc_id)
        out.append({"doc_id": doc_id, "title": str(entry.get("title", "")), "passage": _get_text(entry)})
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {len(out)} docs -> {cache}")
    return out


def load_queries(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load formal MIRACL queries -> [{"query_id","query"}]. Uses cache."""
    data_dir = _data_dir(config)
    cache = data_dir / "queries.json"
    if cache.exists():
        print(f"[INFO] Loading cached queries: {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))
    src = data_dir / config.get("query_tsv", "query.tsv")
    if not src.exists():
        raise FileNotFoundError(f"Query TSV not found: {src}")
    out, seen = [], set()
    # TSV columns: qid \\t query_text
    with open(src, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            qid, text = str(row[0]), str(row[1])
            if qid in seen:
                continue
            seen.add(qid)
            out.append({"query_id": qid, "query": text})
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {len(out)} queries -> {cache}")
    return out


def load_qrels(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load relevance judgments -> [{"query_id","relevant_doc"}]. Keeps rel>0."""
    data_dir = _data_dir(config)
    cache = data_dir / "qrels.json"
    if cache.exists():
        print(f"[INFO] Loading cached qrels: {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))
    src = data_dir / config.get("qrels_tsv", "qrels.tsv")
    if not src.exists():
        raise FileNotFoundError(f"Qrels TSV not found: {src}")
    out = []
    # TSV columns: qid \\t Q0 \\t docid \\t relevance
    with open(src, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4:
                continue
            if int(row[3]) > 0:  # keep only judged-relevant pairs
                out.append({"query_id": str(row[0]), "relevant_doc": str(row[2])})
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {len(out)} qrels -> {cache}")
    return out


def load_pairs(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load your informal/formal pairs -> [{"query_id","formal","informal"}]."""
    # Path is fully configurable; defaults to data/train_pairs.csv.
    p = Path(config.get("train_pairs_csv", "data/train_pairs.csv"))
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        p = (repo_root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"train_pairs.csv not found: {p} (edit train_pairs_csv in config)")
    # pandas handles quoted commas like: "Dimana letak daerah ""Pardembanan""?"
    df = pd.read_csv(p, dtype=str).fillna("")
    # Normalize headers: allow Query_ID / Formal / Informal casing variants.
    lower = {c.lower().strip(): c for c in df.columns}
    qcol = lower.get("query_id", df.columns[0])
    fcol = lower.get("formal", df.columns[1])
    icol = lower.get("informal", df.columns[2])
    out = []
    for _, r in df.iterrows():
        informal = str(r[icol]).strip()
        if not informal:  # skip empty informal rows; nothing to adapt from
            continue
        out.append({"query_id": str(r[qcol]).strip(), "formal": str(r[fcol]).strip(), "informal": informal})
    print(f"[INFO] Loaded {len(out)} formal/informal pairs from {p}")
    return out


def load_all(config: Dict[str, Any]) -> Tuple[List, List, List]:
    """Convenience: load corpus + formal queries + qrels together."""
    return load_corpus(config), load_queries(config), load_qrels(config)
