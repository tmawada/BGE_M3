"""
Model loader for the BGE + DAPT pipeline.

Provides two loading strategies:

1. ``load_mlm_model`` — loads BGE-M3 as a masked language model
   (``XLMRobertaForMaskedLM``) for Stage 1 domain-adaptive pretraining.

2. ``DenseEncoder`` — wraps any BGE-M3 checkpoint (base / DAPT / retriever)
   as a dense (CLS-pooled) encoder with a ``.encode()`` interface identical
   to the baseline project, so the retrieval pipeline never changes when the
   checkpoint is swapped in ``config.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def resolve_device(device: str) -> str:
    """Return a usable device, falling back to CPU when CUDA is unavailable."""
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        return "cpu"
    return device


def load_mlm_model(config: Dict[str, Any]) -> tuple:
    """Load BGE-M3 as a masked language model for DAPT.

    The BGE-M3 checkpoint declares an ``XLMRobertaModel`` architecture, so a
    fresh MLM prediction head is attached by transformers (the encoder weights
    are fully reused). The tokenizer is reused as-is — no retraining, no
    vocabulary expansion.

    Args:
        config: Configuration dictionary with 'base_model_name'.

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_name: str = config.get("base_model_name", "BAAI/bge-m3")
    device: str = resolve_device(config.get("device", "cuda"))

    print(f"[INFO] Loading MLM backbone '{model_name}' on {device} ...")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device)
    model.train()
    return model, tokenizer


@torch.no_grad()
def encode_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
    normalize_embeddings: bool = True,
    use_fp16: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts into dense CLS-pooled embeddings.

    Mixed precision is applied via ``torch.autocast`` so the caller's model
    weights are never mutated in place.

    Args:
        model: A HuggingFace transformer model (``XLMRobertaModel``-like).
        tokenizer: Matching tokenizer.
        texts: List of text strings.
        batch_size: Sentences per encoding batch.
        max_length: Maximum token length per sentence.
        device: Inference device.
        normalize_embeddings: Whether to L2-normalize the output vectors.
        use_fp16: Whether to run the forward pass in fp16 (CUDA only).
        show_progress: Whether to print a progress bar.

    Returns:
        np.ndarray of shape (len(texts), hidden_size) in float32.
    """
    from tqdm import tqdm

    model.eval()
    use_autocast = bool(use_fp16 and device != "cpu")
    all_vectors: List[np.ndarray] = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding", unit="batch")

    for start in iterator:
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        if use_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(**encoded)
        else:
            outputs = model(**encoded)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        vectors = outputs[:, 0]  # CLS token embedding
        if normalize_embeddings:
            vectors = F.normalize(vectors, p=2, dim=-1)

        all_vectors.append(vectors.float().cpu().numpy())

    if not all_vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_vectors, axis=0)


class DenseEncoder:
    """Dense (CLS-pooled) encoder wrapper around a BGE-M3 checkpoint.

    Implements the same ``.encode()`` contract as the baseline project's
    encoder so that any checkpoint — base, DAPT, or retriever — can be used
    by only changing the model path in ``config.yaml``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        tokenizer=None,
        device: str = "cuda",
        use_fp16: bool = True,
    ) -> None:
        """Initialize the dense encoder.

        Pass ``model_path`` to load a checkpoint from disk, or pass an
        already-instantiated ``model``/``tokenizer`` (used during retrieval
        fine-tuning so evaluation shares the in-training weights).

        Args:
            model_path: HuggingFace name or local path to a checkpoint.
            model: Optional in-memory transformer model (replaces loading).
            tokenizer: Optional in-memory tokenizer.
            device: Device to run inference on ('cuda' or 'cpu').
            use_fp16: Whether to use mixed precision (fp16) inference.
        """
        from transformers import AutoModel, AutoTokenizer

        device = resolve_device(device)
        self.device = device

        if model is None:
            if model_path is None:
                raise ValueError("Either 'model_path' or 'model' must be provided.")
            print(f"[INFO] Loading dense encoder '{model_path}' ...")
            self.model = AutoModel.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = model
            self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("BAAI/bge-m3")

        self.use_fp16 = bool(use_fp16 and self.device != "cpu")
        self.model.to(self.device)
        print(f"[INFO] Dense encoder ready on {self.device} (fp16={self.use_fp16})")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of sentences into dense embeddings.

        Args:
            sentences: List of text strings to encode.
            batch_size: Number of sentences per encoding batch.
            max_length: Maximum token length per sentence.
            normalize_embeddings: Whether to L2-normalize the output vectors.
            show_progress: Whether to print a progress bar.

        Returns:
            np.ndarray of shape (len(sentences), embedding_dim).
        """
        return encode_texts(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=sentences,
            batch_size=batch_size,
            max_length=max_length,
            device=self.device,
            normalize_embeddings=normalize_embeddings,
            use_fp16=self.use_fp16,
            show_progress=show_progress,
        )


def load_encoder(config: Dict[str, Any]) -> DenseEncoder:
    """Load a dense encoder from the configured retriever checkpoint.

    Args:
        config: Configuration dictionary with 'retriever_model_dir',
            'device', and 'use_fp16' keys.

    Returns:
        A DenseEncoder instance.
    """
    model_path: str = config.get("retriever_model_dir", "models/bge-m3-dapt-retriever")
    device: str = config.get("device", "cuda")
    use_fp16: bool = config.get("use_fp16", True)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Retriever checkpoint not found at {model_path}. "
            "Run 'python -m src.main --step train_retrieval' first."
        )

    return DenseEncoder(
        model_path=model_path,
        device=device,
        use_fp16=use_fp16,
    )
