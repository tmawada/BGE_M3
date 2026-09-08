"""Stage 2b: LoRA contrastive training (freeze base, train adapters only).

Loss = InfoNCE: pull informal query to its gold passage, push away from
hard negatives + in-batch positives. Saves ONLY adapter weights (tiny).
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


class TripletDataset(Dataset):
    """Wraps training_pairs.json rows for the DataLoader."""

    def __init__(self, rows: List[Dict[str, Any]], num_negs: int) -> None:
        """Store rows + cap negatives per query (random subsample)."""
        self.rows = rows
        self.num_negs = num_negs

    def __len__(self) -> int:
        """Number of training triplets."""
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Return {query, positive, negatives[]} for row i."""
        r = self.rows[i]
        negs = [n["passage"] for n in r["negatives"]]
        if len(negs) > self.num_negs:  # subsample keeps batch memory constant
            negs = random.sample(negs, self.num_negs)
        return {"query": r["query"], "positive": r["positive"]["passage"], "negatives": negs}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge rows -> {queries[], positives[], negatives[], neg_counts[]}."""
    queries = [b["query"] for b in batch]
    positives = [b["positive"] for b in batch]
    flat_negs, counts = [], []  # flatten because each query has variable negs
    for b in batch:
        flat_negs.extend(b["negatives"])
        counts.append(len(b["negatives"]))
    return {"queries": queries, "positives": positives, "negatives": flat_negs, "neg_counts": counts}


def encode_texts(model, tokenizer, texts: List[str], max_len: int, device: str) -> torch.Tensor:
    """Transformer forward + CLS pooling + L2 norm -> (B, dim)."""
    toks = tokenizer(texts, padding=True, truncation=True,
                     max_length=max_len, return_tensors="pt").to(device)
    hidden = model(**toks).last_hidden_state  # (B, L, dim)
    cls = hidden[:, 0, :]  # CLS token = sentence embedding (BGE-M3 style)
    return F.normalize(cls, p=2, dim=1)  # norm => dot product == cosine


def infonce(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, temp: float) -> torch.Tensor:
    """InfoNCE: softmax over [in-batch positives + hard negatives]."""
    # q@pos.T (B,B): diagonal = correct doc; off-diagonal = in-batch negatives (free!).
    pos_sim = (q @ pos.T) / temp
    if neg.numel() > 0:  # append hard negatives as extra columns
        neg_sim = (q @ neg.T) / temp  # (B, N_total)
        logits = torch.cat([pos_sim, neg_sim], dim=1)
    else:
        logits = pos_sim
    labels = torch.arange(q.size(0), device=q.device)  # correct = diagonal index
    return F.cross_entropy(logits, labels)


def train_adapter(config: Dict[str, Any]) -> str:
    """Full LoRA training loop. Returns adapter output dir."""
    from peft import LoraConfig, TaskType, get_peft_model  # heavy, import late
    from transformers import AutoModel, AutoTokenizer

    tr = config.get("training", {})
    ad = config.get("adapter", {})
    # All knobs from config.yaml — edit there, not here.
    base = config["model_name"]  # point to models/bge-m3-dapt for DAPT+LoRA
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] No CUDA, using CPU."); device = "cpu"
    max_len = int(config.get("max_length", 512))
    epochs, bs = int(tr.get("epochs", 3)), int(tr.get("train_batch_size", 16))
    lr, temp = float(tr.get("learning_rate", 2e-5)), float(tr.get("temperature", 0.05))
    num_negs = int(tr.get("num_hard_negatives", 7))
    out_dir = tr.get("output_dir", "adapters/bge-m3-lora")
    seed = int(tr.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Load triplet file built by make_pairs stage.
    data_dir = Path(config.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent.parent / data_dir
    rows = json.loads((Path(data_dir) / "training_pairs.json").read_text(encoding="utf-8"))
    print(f"[INFO] Training on {len(rows)} triplets | base={base}")

    # Load backbone + attach LoRA (only ~1% params trainable).
    tok = AutoTokenizer.from_pretrained(base)
    backbone = AutoModel.from_pretrained(base)
    peft_cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=int(ad.get("lora_r", 16)),
                          lora_alpha=int(ad.get("lora_alpha", 32)),
                          lora_dropout=float(ad.get("lora_dropout", 0.1)),
                          target_modules=list(ad.get("target_modules", ["query", "key", "value"])))
    model = get_peft_model(backbone, peft_cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable {n_train:,}/{n_all:,} ({100*n_train/n_all:.2f}%)")

    loader = DataLoader(TripletDataset(rows, num_negs), batch_size=bs, shuffle=True,
                        collate_fn=collate, drop_last=True)  # drop_last keeps loss shape stable
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=float(tr.get("weight_decay", 0.01)))
    total = len(loader) * epochs  # total steps for cosine schedule
    warmup = int(total * float(tr.get("warmup_ratio", 0.1)))

    def _lr(step: int) -> float:  # linear warmup then cosine decay
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)

    use_amp = bool(config.get("use_fp16", True)) and device == "cuda"  # mixed precision on GPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, t0 = float("inf"), time.time()
    for ep in range(epochs):
        model.train()
        ep_loss, n_steps = 0.0, 0
        for batch in tqdm(loader, desc=f"Epoch {ep+1}/{epochs}"):
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                q = encode_texts(model, tok, batch["queries"], max_len, device)
                p = encode_texts(model, tok, batch["positives"], max_len, device)
                n = encode_texts(model, tok, batch["negatives"], max_len, device) if batch["negatives"] \
                    else torch.zeros(0, q.size(1), device=device)
                loss = infonce(q, p, n, temp)  # one scalar for the batch
            scaler.scale(loss).backward()  # scaled backward avoids fp16 underflow
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("max_grad_norm", 1.0)))
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss += loss.item(); n_steps += 1
        avg = ep_loss / max(1, n_steps)
        print(f"[INFO] Epoch {ep+1} avg_loss={avg:.4f}")
        if avg < best:  # keep only best checkpoint (lowest loss)
            best = avg
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
            print(f"[INFO] New best -> saved to {out_dir}")
    print(f"[INFO] Done in {time.time()-t0:.1f}s. Best loss {best:.4f}.")
    return out_dir
