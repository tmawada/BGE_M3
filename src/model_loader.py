"""
Model loader module for BGE-M3 embedding generation.

This module abstracts the embedding model loading so that future experiments
(DAPT, Contrastive Learning, DANN) can swap the checkpoint via config.yaml
without code changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import torch


class Encoder(Protocol):
    """Protocol defining the interface any encoder must implement."""

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode a list of sentences into dense embeddings."""
        ...


class BGEM3Encoder:
    """Wrapper around FlagEmbedding's BGEM3FlagModel.

    Provides a consistent .encode() interface that returns dense embeddings
    as a numpy array.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_fp16: bool = True,
    ) -> None:
        """Initialize the BGE-M3 encoder.

        Args:
            model_name: HuggingFace model name or local path to checkpoint.
            device: Device to run inference on ('cuda' or 'cpu').
            use_fp16: Whether to use mixed precision (fp16) inference.
        """
        from FlagEmbedding import BGEM3FlagModel

        # Resolve device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU.")
            device = "cpu"
            use_fp16 = False

        self.device = device
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            device=device,
        )
        print(f"[INFO] Loaded model '{model_name}' on {device} (fp16={use_fp16})")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode sentences into dense embeddings.

        Args:
            sentences: List of text strings to encode.
            batch_size: Number of sentences per encoding batch.
            max_length: Maximum token length per sentence.
            normalize_embeddings: Whether to L2-normalize the output vectors.

        Returns:
            np.ndarray of shape (len(sentences), embedding_dim).
        """
        output = self.model.encode(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        embeddings: np.ndarray = output["dense_vecs"]

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        return embeddings


def load_model(config: Dict[str, Any]) -> Encoder:
    """Load an embedding model based on configuration.

    This factory function is the single entry point for model loading.
    Future experiments can extend this function to support additional
    model types (e.g., DAPT-adapted checkpoints) by checking config keys.

    Args:
        config: Configuration dictionary. Must contain 'model_name'.
            Optional keys: 'device', 'use_fp16'.

    Returns:
        An Encoder instance with an .encode() method.
    """
    model_name: str = config["model_name"]
    device: str = config.get("device", "cuda")
    use_fp16: bool = config.get("use_fp16", True)

    return BGEM3Encoder(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
    )
