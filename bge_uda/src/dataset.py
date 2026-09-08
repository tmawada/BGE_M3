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


def _split_path(config: Dict[str, Any]) -> Path:
    """Resolve persisted pair-split JSON (train vs held-out qids)."""
    p = Path(config.get("split", {}).get("split_file", "../data/pair_split.json"))
    if not p.is_absolute():
        p = (Path(__file__).resolve().parent.parent / p).resolve()
    return p


def split_pairs(pairs: List[Dict[str, str]], config: Dict[str, Any],
                force: bool = False) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Deterministic train/held-out split by query_id (persisted to disk).

    Fixes train-on-test leakage: mining + LoRA + DAPT must use train only,
    eval_informal/eval_sim must use held-out only.
    """
    import hashlib
    import json

    scfg = config.get("split", {})
    ratio = float(scfg.get("held_out_ratio", 0.15))
    seed = int(scfg.get("seed", 42))
    spath = _split_path(config)
    if not force and spath.exists():
        try:
            saved = json.loads(spath.read_text(encoding="utf-8"))
            train_ids = set(map(str, saved.get("train_ids", [])))
            held_ids = set(map(str, saved.get("held_out_ids", [])))
            by_id = {str(p["query_id"]): p for p in pairs}
            # Only reuse if split covers exactly the current pair ids.
            if train_ids | held_ids == set(by_id) and not (train_ids & held_ids):
                train = [by_id[i] for i in saved["train_ids"] if i in by_id]
                held = [by_id[i] for i in saved["held_out_ids"] if i in by_id]
                print(f"[INFO] Reusing split {spath}: {len(train)} train / {len(held)} held-out")
                return train, held
            print(f"[WARNING] Split {spath} stale (pair ids changed) -> re-split.")
        except Exception as e:
            print(f"[WARNING] Could not load split {spath} ({e}) -> re-split.")

    # Deterministic hash-based split: stable across runs/machines, no sklearn needed.
    train, held = [], []
    for p in pairs:
        qid = str(p["query_id"])
        h = int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (held if h < ratio else train).append(p)
    # Guard against degenerate splits on tiny inputs.
    if not train or not held:
        n_held = max(1, int(len(pairs) * ratio)) if len(pairs) > 1 else 0
        ordered = sorted(pairs, key=lambda p: str(p["query_id"]))
        held, train = ordered[:n_held], ordered[n_held:]
    try:
        spath.parent.mkdir(parents=True, exist_ok=True)
        spath.write_text(json.dumps({
            "seed": seed, "held_out_ratio": ratio,
            "train_ids": [str(p["query_id"]) for p in train],
            "held_out_ids": [str(p["query_id"]) for p in held],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote split {len(train)} train / {len(held)} held-out -> {spath}")
    except Exception as e:
        print(f"[WARNING] Could not persist split ({e}); using in-memory split.")
    return train, held


def filter_qrels(qrels: List[Dict[str, str]], qids: List[str]) -> List[Dict[str, str]]:
    """Keep only qrels for the given qids (avoids missing-result deflation)."""
    keep = set(map(str, qids))
    return [r for r in qrels if str(r["query_id"]) in keep]
