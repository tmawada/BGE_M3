"""Demo: compare formal vs informal embeddings (cosine / euclidean / dot)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _cos


def compare_pair(encoder, formal: str, informal: str, config: Dict[str, Any]) -> Dict[str, float]:
    """Encode [formal, informal] in ONE batch, return similarity scores."""
    # Single batch call = shared padding, faster than two separate calls.
    vecs = encoder.encode([formal, informal], config.get("batch_size", 32),
                          config.get("max_length", 512), config.get("normalize_embeddings", True))
    a, b = vecs[0].astype(np.float32), vecs[1].astype(np.float32)
    cos = float(_cos(a.reshape(1, -1), b.reshape(1, -1))[0][0])  # sklearn cosine
    euc = float(np.linalg.norm(a - b))  # L2 distance; lower = closer
    dot = float(np.dot(a, b))  # == cosine when vectors are normalized
    print(f"\nFormal  : {formal}\nInformal: {informal}")
    print(f"Cosine={cos:.4f} | Euclidean={euc:.4f} | Dot={dot:.4f} | dim={vecs.shape[1]}")
    print("Higher cosine = adaptation working (target: gain vs baseline).")
    return {"cosine": cos, "euclidean": euc, "dot": dot, "dim": int(vecs.shape[1])}
