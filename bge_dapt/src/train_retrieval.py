"""
Stage 2 — Retrieval Fine-tuning on MIRACL Indonesian.

Continues training the DAPT checkpoint (``models/bge-m3-dapt``) as a dense
retriever using MultipleNegativesRankingLoss (InfoNCE) with in-batch
negatives. The model is initialized ONLY from the DAPT checkpoint, never from
the original BGE-M3.

Evaluation runs after every epoch (FAISS retrieval on the MIRACL dev corpus)
and the best checkpoint is selected by Recall@10.

Outputs:
    outputs/checkpoint-best-retriever   best checkpoint (by Recall@10)
    models/bge-m3-dapt-retriever        final model
    outputs/training_log_retrieval.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .evaluate import evaluate_retrieval
from .model_loader import resolve_device


class RetrieverModel(nn.Module):
    """Dense retriever backbone producing L2-normalized CLS embeddings."""

    def __init__(self, model_path: str) -> None:
        """Initialize from an existing checkpoint (e.g. the DAPT model)."""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        vectors = outputs.last_hidden_state[:, 0]  # CLS token embedding
        return F.normalize(vectors, p=2, dim=-1)


class PairDataset(Dataset):
    """Dataset of (query, positive document) training pairs."""

    def __init__(self, pairs: List[Dict[str, Any]]) -> None:
        self.queries = [p["query"] for p in pairs]
        self.documents = [p["positive_text"] for p in pairs]

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return {"query": self.queries[index], "document": self.documents[index]}


def _collate(batch: List[Dict[str, str]], tokenizer, max_length: int) -> tuple:
    """Tokenize and pad a batch of query/document pairs."""
    queries = tokenizer(
        [b["query"] for b in batch],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    documents = tokenizer(
        [b["document"] for b in batch],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return queries, documents


def mnrl_loss(query_embeddings: torch.Tensor, document_embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    """MultipleNegativesRankingLoss (InfoNCE) with in-batch negatives.

    Args:
        query_embeddings: Normalized query embeddings, shape (B, D).
        document_embeddings: Normalized document embeddings, shape (B, D).
        temperature: Similarity scaling temperature.

    Returns:
        Scalar loss.
    """
    batch_size = query_embeddings.size(0)
    similarities = torch.matmul(query_embeddings, document_embeddings.t()) / temperature
    labels = torch.arange(batch_size, device=similarities.device)
    return F.cross_entropy(similarities, labels)


def _load_pairs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset_path = Path(config.get("dataset_path", "data/miracl_pairs.json"))
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Retrieval pairs not found at {dataset_path}. "
            "Run 'python -m src.main --step prepare_retrieval' first."
        )
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_retrieval(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the retrieval fine-tuning stage.

    Args:
        config: Configuration dictionary. Retrieval options are merged at
            the top level by the caller.

    Returns:
        Training summary dictionary (persisted as the retrieval training log).
    """
    start_time = time.perf_counter()

    dapt_model_dir = Path(config.get("dapt_model_dir", "models/bge-m3-dapt"))
    retriever_model_dir = Path(config.get("retriever_model_dir", "models/bge-m3-dapt-retriever"))
    outputs_dir = Path(config.get("outputs_dir", "outputs"))

    if not dapt_model_dir.exists():
        raise FileNotFoundError(
            f"DAPT checkpoint not found at {dapt_model_dir}. "
            "Run 'python -m src.main --step train_dapt' first."
        )

    device = resolve_device(config.get("device", "cuda"))
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 32))
    learning_rate = float(config.get("learning_rate", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.01))
    warmup_ratio = float(config.get("warmup_ratio", 0.06))
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    fp16 = bool(config.get("fp16", True) and device != "cpu")
    temperature = float(config.get("temperature", 0.02))
    max_length = int(config.get("max_length", 256))
    top_k = int(config.get("top_k", 10))
    evaluate_every_epoch = bool(config.get("evaluate_every_epoch", True))

    pairs = _load_pairs(config)
    if not pairs:
        raise ValueError("No retrieval training pairs found.")

    print("\n" + "=" * 60)
    print("  STAGE 2: Retrieval Fine-tuning (MIRACL, MNRL)")
    print("=" * 60)
    print(f"[INFO] DAPT init checkpoint : {dapt_model_dir}")
    print(f"[INFO] Training pairs       : {len(pairs)}")
    print(f"[INFO] Device               : {device}")
    print(f"[INFO] Learning rate        : {learning_rate}")
    print(f"[INFO] Batch size           : {batch_size}")
    print(f"[INFO] Epochs               : {epochs}")
    print(f"[INFO] Temperature          : {temperature}")
    print(f"[INFO] Max length           : {max_length}")

    model = RetrieverModel(str(dapt_model_dir))
    model.to(device)

    dataset = PairDataset(pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, model.tokenizer, max_length),
        drop_last=False,
    )

    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_autocast = bool(fp16 and device != "cpu")

    best_recall10 = -1.0
    train_losses: List[float] = []
    metrics_per_epoch: List[Dict[str, Any]] = []
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        running_loss = 0.0
        model.train()
        optimizer.zero_grad()

        for step, (queries, documents) in enumerate(dataloader, start=1):
            queries = {k: v.to(device) for k, v in queries.items()}
            documents = {k: v.to(device) for k, v in documents.items()}

            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    query_embeddings = model(**queries)
                    document_embeddings = model(**documents)
                    loss = mnrl_loss(query_embeddings, document_embeddings, temperature)
            else:
                query_embeddings = model(**queries)
                document_embeddings = model(**documents)
                loss = mnrl_loss(query_embeddings, document_embeddings, temperature)

            loss.backward()
            running_loss += loss.item()

            if step % grad_accum == 0 or step == steps_per_epoch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 0 or step == steps_per_epoch:
                print(f"  [Epoch {epoch}/{epochs}] step {step}/{steps_per_epoch} "
                      f"loss {running_loss / step:.4f} lr {scheduler.get_last_lr()[0]:.2e}")

        epoch_loss = running_loss / steps_per_epoch
        train_losses.append(epoch_loss)
        epoch_time = time.perf_counter() - epoch_start
        print(f"  [Epoch {epoch}/{epochs}] train loss {epoch_loss:.4f} "
              f"({epoch_time:.2f} sec)")

        entry: Dict[str, Any] = {"epoch": epoch, "train_loss": round(epoch_loss, 6)}

        if evaluate_every_epoch:
            eval_metrics = evaluate_retrieval(
                config,
                encoder=None,
                model=model,
                tokenizer=model.tokenizer,
                use_cached_embeddings=False,
                tag=f"epoch_{epoch}",
            )
            for name, value in eval_metrics.items():
                entry[name] = round(float(value), 6)
            recall10 = float(eval_metrics.get("Recall@10", 0.0))
        else:
            recall10 = -1.0

        metrics_per_epoch.append(entry)
        print(f"  [Epoch {epoch}/{epochs}] Recall@10={entry.get('Recall@10', 'n/a')}")

        # Save best checkpoint by Recall@10
        if recall10 > best_recall10:
            best_recall10 = recall10
            best_dir = outputs_dir / "checkpoint-best-retriever"
            _save_hf_model(model, best_dir)
            print(f"  [INFO] New best Recall@10={recall10:.4f} -> {best_dir}")

    # Save final model
    _save_hf_model(model, retriever_model_dir)
    print(f"[INFO] Saved final retriever -> {retriever_model_dir}")

    training_time = time.perf_counter() - start_time
    print(f"[INFO] Elapsed time: {training_time:.2f} sec")

    summary: Dict[str, Any] = {
        "stage": "retrieval",
        "dapt_checkpoint": str(dapt_model_dir),
        "training_pairs": len(pairs),
        "epochs": epochs,
        "train_loss_per_epoch": [round(v, 6) for v in train_losses],
        "metrics_per_epoch": metrics_per_epoch,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_length": max_length,
        "temperature": temperature,
        "loss": "MultipleNegativesRankingLoss",
        "optimizer": "adamw_torch",
        "scheduler": "linear",
        "fp16": fp16,
        "best_recall_at_10": round(best_recall10, 6),
        "training_time_sec": round(training_time, 2),
        "best_checkpoint": str(outputs_dir / "checkpoint-best-retriever"),
        "model_dir": str(retriever_model_dir),
    }

    log_path = outputs_dir / "training_log_retrieval.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved retrieval training log to {log_path}")

    return summary


def _save_hf_model(model: RetrieverModel, save_dir: Path) -> None:
    """Save the backbone + tokenizer in HuggingFace format."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(save_dir))
    model.tokenizer.save_pretrained(str(save_dir))
