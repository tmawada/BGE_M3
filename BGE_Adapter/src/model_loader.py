"""
Model loader module for BGE-M3 + LoRA Adapter embedding generation.

This module provides two encoder implementations:
  1. BGEM3Encoder        — vanilla BGE-M3 via FlagEmbedding (baseline)
  2. BGEM3AdapterEncoder — BGE-M3 with LoRA adapters via PEFT + Transformers

The factory function `load_model()` selects the appropriate encoder based on
the `use_adapter` flag in config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Protocol

import numpy as np
import torch

# PEFT weight filenames, in the order PeftModel.from_pretrained looks for them.
_ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def _verify_local_adapter(adapter_path: str) -> None:
    """Fail fast when `adapter_path` looks local but has no trained weights.

    PEFT silently falls back to downloading from the HuggingFace Hub when the
    weight file is missing, which surfaces as a confusing 401 / repo-not-found
    error for what is really an untrained adapter.

    Args:
        adapter_path: Path (or Hub repo id) passed to PeftModel.from_pretrained.

    Raises:
        FileNotFoundError: If the directory exists but holds no adapter weights,
            or if the path was clearly meant to be local but does not exist.
    """
    path = Path(adapter_path)

    # A bare "org/repo" style id that isn't on disk is a legitimate Hub ref,
    # but if its parent directory exists locally the path was meant to be local.
    if not path.exists():
        looks_local = (
            path.is_absolute()
            or adapter_path.startswith((".", "/", "\\"))
            or path.parent.is_dir()
        )
        if looks_local:
            raise FileNotFoundError(
                f"Adapter path '{adapter_path}' does not exist. "
                f"Train an adapter first: python -m src.train_adapter"
            )
        return

    if any((path / name).is_file() for name in _ADAPTER_WEIGHT_FILES):
        return

    found = sorted(p.name for p in path.iterdir()) or ["<empty>"]
    raise FileNotFoundError(
        f"No adapter weights in '{adapter_path}': expected one of "
        f"{', '.join(_ADAPTER_WEIGHT_FILES)}, found: {', '.join(found)}.\n"
        f"The adapter has not been trained yet. Either run\n"
        f"    python -m src.train_adapter\n"
        f"to produce the weights, or set 'use_adapter: false' in "
        f"config/config.yaml to use the baseline BGE-M3 model."
    )


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


class BGEM3AdapterEncoder:
    """BGE-M3 encoder with LoRA adapter weights loaded via PEFT.

    Loads the base XLM-RoBERTa backbone used by BGE-M3 from HuggingFace
    Transformers, then merges LoRA adapter weights on top. Produces dense
    embeddings using CLS-token pooling (matching BGE-M3's original behavior).
    """

    def __init__(
        self,
        model_name: str,
        adapter_path: str,
        device: str = "cuda",
        use_fp16: bool = True,
    ) -> None:
        """Initialize the adapter-enhanced BGE-M3 encoder.

        Args:
            model_name: HuggingFace model name for the base model.
            adapter_path: Path to saved PEFT adapter weights.
            device: Device to run inference on ('cuda' or 'cpu').
            use_fp16: Whether to use fp16 inference.
        """
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        # Resolve device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU.")
            device = "cpu"
            use_fp16 = False

        self.device = device
        self.use_fp16 = use_fp16

        print(f"[INFO] Loading base model '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load base transformer model
        dtype = torch.float16 if use_fp16 else torch.float32
        base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

        # Load LoRA adapter on top
        print(f"[INFO] Loading LoRA adapter from '{adapter_path}'...")
        _verify_local_adapter(adapter_path)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.to(device)
        self.model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"[INFO] Loaded adapter model on {device} (fp16={use_fp16})\n"
            f"       Total params: {total_params:,} | "
            f"Adapter params: {trainable_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode sentences using the adapter-enhanced model.

        Uses CLS-token pooling to match BGE-M3's dense embedding behavior.

        Args:
            sentences: List of text strings to encode.
            batch_size: Number of sentences per encoding batch.
            max_length: Maximum token length per sentence.
            normalize_embeddings: Whether to L2-normalize the output vectors.

        Returns:
            np.ndarray of shape (len(sentences), embedding_dim).
        """
        all_embeddings: List[np.ndarray] = []

        for start_idx in range(0, len(sentences), batch_size):
            batch = sentences[start_idx : start_idx + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            # CLS-token pooling (index 0)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().float().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        return embeddings


def load_model(config: Dict[str, Any]) -> Encoder:
    """Load an embedding model based on configuration.

    If `use_adapter` is True in the config, loads the base model with PEFT
    adapter weights from `adapter_path`. Otherwise, loads the vanilla BGE-M3
    model via FlagEmbedding.

    Args:
        config: Configuration dictionary. Must contain 'model_name'.
            Optional keys: 'device', 'use_fp16', 'use_adapter', 'adapter_path'.

    Returns:
        An Encoder instance with an .encode() method.
    """
    model_name: str = config["model_name"]
    device: str = config.get("device", "cuda")
    use_fp16: bool = config.get("use_fp16", True)
    use_adapter: bool = config.get("use_adapter", False)

    if use_adapter:
        adapter_path: str = config.get("adapter_path", "adapters/bge-m3-lora")
        return BGEM3AdapterEncoder(
            model_name=model_name,
            adapter_path=adapter_path,
            device=device,
            use_fp16=use_fp16,
        )

    return BGEM3Encoder(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
    )
