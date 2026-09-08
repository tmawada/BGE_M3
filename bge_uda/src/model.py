"""Model wrappers with one shared .encode() interface.

Two encoders:
  BGEM3Encoder      -> vanilla BAAI/bge-m3 via FlagEmbedding (baseline + DAPT ckpt)
  AdapterEncoder    -> base + LoRA weights via PEFT (after train stage)
Factory load_model() picks based on config use_adapter flag.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol

import numpy as np
import torch


class Encoder(Protocol):
    """Any encoder must implement .encode() returning (N, dim) numpy."""

    def encode(self, sentences: List[str], batch_size: int = 32,
               max_length: int = 512, normalize_embeddings: bool = True) -> np.ndarray:
        ...


def _resolve_device(device: str, use_fp16: bool) -> tuple[str, bool]:
    """Fall back to CPU if CUDA requested but unavailable."""
    # Prevents crash on CPU-only machines; also disables fp16 there.
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        return "cpu", False
    return device, use_fp16


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize so inner-product equals cosine similarity."""
    # max(norm, 1e-12) avoids divide-by-zero on empty strings.
    norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    return mat / norms


class BGEM3Encoder:
    """Vanilla BGE-M3 encoder (also loads your DAPT checkpoint)."""

    def __init__(self, model_name: str, device: str = "cuda", use_fp16: bool = True) -> None:
        """Load FlagEmbedding model once."""
        from FlagEmbedding import BGEM3FlagModel  # lazy import: heavy dep
        device, use_fp16 = _resolve_device(device, use_fp16)
        self.device = device
        # BGEM3FlagModel handles BGE-M3's special tokenizer + pooling.
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
        print(f"[INFO] Loaded BGE-M3 '{model_name}' on {device} (fp16={use_fp16})")

    def encode(self, sentences: List[str], batch_size: int = 32,
               max_length: int = 512, normalize_embeddings: bool = True) -> np.ndarray:
        """Encode texts -> dense vectors only (sparse/ColBERT disabled)."""
        # return_dense=True + others False = 1024-d single vector per text.
        out = self.model.encode(sentences, batch_size=batch_size, max_length=max_length,
                                return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vecs: np.ndarray = out["dense_vecs"]
        # Manual norm (don't rely on library flag) keeps FAISS IP == cosine.
        return _l2_normalize(vecs) if normalize_embeddings else vecs


class AdapterEncoder:
    """BGE-M3 + LoRA adapter encoder (CLS pooling to match BGE-M3 dense)."""

    def __init__(self, model_name: str, adapter_path: str,
                 device: str = "cuda", use_fp16: bool = True) -> None:
        """Load base transformer + PEFT adapter weights."""
        from peft import PeftModel  # lazy import: only needed after training
        from transformers import AutoModel, AutoTokenizer
        device, use_fp16 = _resolve_device(device, use_fp16)
        self.device = device
        print(f"[INFO] Loading base '{model_name}' + adapter '{adapter_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # fp16 halves VRAM on GPU; float32 on CPU.
        dtype = torch.float16 if use_fp16 else torch.float32
        base = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        # PeftModel wraps base with tiny trainable LoRA matrices.
        self.model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()
        print("[INFO] Adapter loaded and set to eval mode.")

    def encode(self, sentences: List[str], batch_size: int = 32,
               max_length: int = 512, normalize_embeddings: bool = True) -> np.ndarray:
        """Batched CLS-pooling encode with no grad (inference only)."""
        all_vecs: List[np.ndarray] = []
        # Manual batch loop avoids OOM on large corpora.
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():  # no grad = faster + less memory
                hidden = self.model(**toks).last_hidden_state
            cls = hidden[:, 0, :].cpu().float().numpy()  # [:,0,:] = CLS token
            all_vecs.append(cls)
        vecs = np.concatenate(all_vecs, axis=0)
        return _l2_normalize(vecs) if normalize_embeddings else vecs


def load_model(config: Dict[str, Any]) -> Encoder:
    """Factory: return AdapterEncoder if use_adapter else BGEM3Encoder."""
    # One-line switch in config.yaml changes the whole pipeline's model.
    if config.get("use_adapter", False):
        return AdapterEncoder(config["model_name"], config.get("adapter_path", "adapters/bge-m3-lora"),
                              config.get("device", "cuda"), config.get("use_fp16", True))
    return BGEM3Encoder(config["model_name"], config.get("device", "cuda"), config.get("use_fp16", True))
