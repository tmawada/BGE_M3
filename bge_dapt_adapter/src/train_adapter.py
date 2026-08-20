"""
LoRA adapter training for the DAPT-tuned BGE-M3 checkpoint.

Freezes the DAPT checkpoint (domain-adapted BGE-M3) and trains only LoRA
adapter parameters using InfoNCE contrastive loss on MIRACL
query-document pairs.

Training flow:
  1. Load DAPT checkpoint + attach LoRA adapters via PEFT
  2. Load training triplets (query, positive, hard-negatives)
  3. Pre-tokenize the dataset once and cache it to disk
  4. Train with InfoNCE loss (in-batch + hard negatives)
  5. Save adapter weights only
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ContrastiveDataset(Dataset):
    """Dataset for contrastive training with hard negatives."""

    def __init__(
        self,
        training_data: List[Dict[str, Any]],
        num_hard_negatives: int = 7,
    ) -> None:
        self.data = training_data
        self.num_hard_negatives = num_hard_negatives

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        example = self.data[idx]
        query = example["query"]
        positive = example["positive"]["passage"]

        negatives = example["negatives"]
        if len(negatives) > self.num_hard_negatives:
            negatives = random.sample(negatives, self.num_hard_negatives)

        neg_passages = [n["passage"] for n in negatives]

        return {
            "query": query,
            "positive": positive,
            "negatives": neg_passages,
        }


class TokenizedDataset(Dataset):
    """Dataset of pre-tokenized (query, positive, negatives) examples."""

    def __init__(self, examples: List[Dict[str, torch.Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def _tokenize_examples(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    num_hard_negatives: int,
    seed: int,
) -> List[Dict[str, torch.Tensor]]:
    """Tokenize all texts once, padding each sequence to ``max_length``.

    Returns a flat list of examples with fixed-size tensors for queries,
    positives, and the (variable-count) negatives per example.
    """
    rng = random.Random(seed)

    tokenized: List[Dict[str, torch.Tensor]] = []
    for example in examples:
        query = example["query"]
        positive = example["positive"]

        negatives = example["negatives"]
        if len(negatives) > num_hard_negatives:
            negatives = rng.sample(negatives, num_hard_negatives)

        neg_texts = [n["passage"] for n in negatives]

        q = tokenizer(
            [query], padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        p = tokenizer(
            [positive], padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        n = tokenizer(
            neg_texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        ) if neg_texts else {
            "input_ids": torch.zeros(0, max_length, dtype=torch.long),
            "attention_mask": torch.zeros(0, max_length, dtype=torch.long),
        }

        tokenized.append(
            {
                "query_input_ids": q["input_ids"][0],
                "query_attention_mask": q["attention_mask"][0],
                "positive_input_ids": p["input_ids"][0],
                "positive_attention_mask": p["attention_mask"][0],
                "negative_input_ids": n["input_ids"],
                "negative_attention_mask": n["attention_mask"],
            }
        )
    return tokenized


def _collate_tokenized(batch: List[Dict[str, torch.Tensor]]) -> tuple:
    """Stack a batch of pre-tokenized examples into batched tensors."""
    queries = {
        "input_ids": torch.stack([b["query_input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["query_attention_mask"] for b in batch]),
    }
    positives = {
        "input_ids": torch.stack([b["positive_input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["positive_attention_mask"] for b in batch]),
    }
    negatives = {
        "input_ids": torch.cat([b["negative_input_ids"] for b in batch], dim=0),
        "attention_mask": torch.cat([b["negative_attention_mask"] for b in batch], dim=0),
    }
    neg_counts = [b["negative_input_ids"].size(0) for b in batch]
    return queries, positives, negatives, neg_counts


def infonce_loss(
    query_embs: torch.Tensor,
    positive_embs: torch.Tensor,
    negative_embs: torch.Tensor,
    neg_counts: List[int],
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss.

    For each query, the positive is the matching document and negatives
    include both hard negatives AND in-batch positives from other queries.

    Args:
        query_embs: (batch_size, dim)
        positive_embs: (batch_size, dim)
        negative_embs: (total_negatives, dim)
        neg_counts: Number of negatives per query.
        temperature: Scaling temperature.

    Returns:
        Scalar loss.
    """
    batch_size = query_embs.size(0)

    # Similarity: query vs all positives (in-batch)
    pos_sim = torch.mm(query_embs, positive_embs.T) / temperature  # (B, B)

    # Similarity: query vs hard negatives
    if negative_embs.size(0) > 0:
        neg_sim = torch.mm(query_embs, negative_embs.T) / temperature  # (B, N_total)
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, B + N_total)
    else:
        logits = pos_sim

    # Labels: the diagonal of pos_sim is the correct match
    labels = torch.arange(batch_size, device=query_embs.device)

    return F.cross_entropy(logits, labels)


def train_adapter(config: Dict[str, Any]) -> str:
    """Train LoRA adapter on BGE-M3 with contrastive learning.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Path to saved adapter weights.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    training_config = config.get("training", {})
    adapter_config = config.get("adapter", {})

    # Hyperparameters
    model_name = config.get("dapt_model_dir") or config["model_name"]
    device = config.get("device", "cuda")
    use_fp16 = config.get("use_fp16", True)
    max_length = config.get("max_length", 128)
    epochs = training_config.get("epochs", 3)
    train_batch_size = training_config.get("train_batch_size", 8)
    learning_rate = training_config.get("learning_rate", 2e-5)
    warmup_ratio = training_config.get("warmup_ratio", 0.1)
    weight_decay = training_config.get("weight_decay", 0.01)
    temperature = training_config.get("temperature", 0.05)
    num_hard_negatives = training_config.get("num_hard_negatives", 15)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    log_every = training_config.get("log_every", 50)
    num_workers = training_config.get("num_workers", 4)
    output_dir = training_config.get("output_dir", "adapters/bge-m3-dapt-lora")
    seed = training_config.get("seed", 42)

    # LoRA config
    lora_r = adapter_config.get("lora_r", 16)
    lora_alpha = adapter_config.get("lora_alpha", 32)
    lora_dropout = adapter_config.get("lora_dropout", 0.1)
    target_modules = adapter_config.get("target_modules", ["q_proj", "v_proj"])

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device handling
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        device = "cpu"
        use_fp16 = False

    # Load training data
    from .prepare_training_data import prepare_training_data

    mining_config = {
        **config,
        "use_adapter": False,
        "embeddings_dir": config.get("mining_embeddings_dir", "embeddings_mining"),
        "faiss_dir": config.get("mining_faiss_dir", "faiss_mining"),
    }
    training_data = prepare_training_data(mining_config)

    if not training_data:
        raise ValueError("No training data available. Cannot train adapter.")

    print(f"\n[INFO] Training configuration:")
    print(f"  Base model:       {model_name}")
    print(f"  LoRA rank:        {lora_r}")
    print(f"  LoRA alpha:       {lora_alpha}")
    print(f"  Epochs:           {epochs}")
    print(f"  Batch size:       {train_batch_size}")
    print(f"  Learning rate:    {learning_rate}")
    print(f"  Temperature:      {temperature}")
    print(f"  Hard negatives:   {num_hard_negatives}")
    print(f"  Training samples: {len(training_data)}")

    # Load model and tokenizer from the DAPT checkpoint
    print(f"\n[INFO] Loading DAPT base model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # Attach LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    model = get_peft_model(base_model, peft_config)
    model = model.to(device)

    # Print parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Dataset and dataloader
    dataset = ContrastiveDataset(training_data, num_hard_negatives)

    cache_dir = Path(config.get("data_dir", "data"))
    cache_path = cache_dir / f"tokenized_pairs_max{max_length}.pt"
    source_path = cache_dir / "training_pairs.json"
    if cache_path.exists() and (not source_path.exists() or cache_path.stat().st_mtime >= source_path.stat().st_mtime):
        print(f"[INFO] Loading pre-tokenized dataset from {cache_path}")
        tokenized = torch.load(cache_path, map_location="cpu")
    else:
        print(f"[INFO] Pre-tokenizing {len(dataset)} training examples (max_length={max_length})...")
        tokenized = _tokenize_examples(
            dataset.data, tokenizer, max_length, num_hard_negatives, seed
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tokenized, cache_path)
        print(f"[INFO] Saved pre-tokenized dataset to {cache_path}")

    tokenized_dataset = TokenizedDataset(tokenized)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=_collate_tokenized,
        drop_last=True,
        num_workers=num_workers,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print(f"\n{'#' * 60}")
    print(f"  Starting LoRA Adapter Training")
    print(f"  Total steps: {total_steps} | Warmup: {warmup_steps}")
    print(f"{'#' * 60}\n")

    global_step = 0
    best_loss = float("inf")
    train_start = time.time()

    # Mixed precision scaler
    use_amp = use_fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            queries = {k: v.to(device) for k, v in batch[0].items()}
            positives = {k: v.to(device) for k, v in batch[1].items()}
            negatives = {k: v.to(device) for k, v in batch[2].items()}
            neg_counts = batch[3]

            with torch.amp.autocast("cuda", enabled=use_amp):
                query_embs = F.normalize(
                    model(**queries).last_hidden_state[:, 0, :], p=2, dim=1
                )
                positive_embs = F.normalize(
                    model(**positives).last_hidden_state[:, 0, :], p=2, dim=1
                )

                if negatives["input_ids"].size(0) > 0:
                    negative_embs = F.normalize(
                        model(**negatives).last_hidden_state[:, 0, :], p=2, dim=1
                    )
                else:
                    negative_embs = torch.zeros(
                        0, query_embs.size(1), device=device
                    )

                loss = infonce_loss(
                    query_embs, positive_embs, negative_embs,
                    neg_counts, temperature
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if log_every > 0 and global_step % log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                print(
                    f"  [Step {global_step}] "
                    f"loss={loss.item():.4f} | "
                    f"avg_loss={avg_loss:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  Epoch {epoch + 1} complete | Avg loss: {avg_epoch_loss:.4f}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            print(f"  [BEST] Saved adapter to {output_path}")

    elapsed = time.time() - train_start
    print(f"\n{'#' * 60}")
    print(f"  Training Complete!")
    print(f"  Total time: {elapsed:.1f}s | Best loss: {best_loss:.4f}")
    print(f"  Adapter saved to: {output_dir}")
    print(f"{'#' * 60}\n")

    # Save training summary
    summary = {
        "model_name": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "epochs": epochs,
        "batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "best_loss": best_loss,
        "total_steps": global_step,
        "training_samples": len(training_data),
        "training_time_seconds": elapsed,
        "trainable_params": trainable,
        "total_params": total,
    }

    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved training summary to {summary_path}")

    return output_dir
