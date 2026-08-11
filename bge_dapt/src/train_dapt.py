"""
Stage 1 — Domain-Adaptive Pretraining (DAPT) via Masked Language Modeling.

Continues pretraining BGE-M3 on the informal Indonesian e-commerce review
corpus using MLM and the HuggingFace Trainer. No retrieval or contrastive
training happens in this stage.

Outputs:
    outputs/checkpoint-best      best checkpoint (by eval loss)
    outputs/checkpoint-last      latest checkpoint
    models/bge-m3-dapt           final model, loadable via AutoModel
    outputs/training_log_dapt.json
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from .load_dataset import load_reviews
from .model_loader import load_mlm_model, resolve_device


def _tokenize_dataset(
    samples: List[Dict[str, str]],
    tokenizer,
    max_length: int,
    eval_ratio: float,
    seed: int,
) -> tuple:
    """Tokenize MLM samples and split into train/eval datasets."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples)).tolist()

    num_eval = max(1, int(len(samples) * eval_ratio))
    train_indices = indices[num_eval:]
    eval_indices = indices[:num_eval]

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_ds = Dataset.from_list([samples[i] for i in train_indices])
    eval_ds = Dataset.from_list([samples[i] for i in eval_indices])

    train_ds = train_ds.map(
        tokenize, batched=True, remove_columns=["text"], desc="Tokenizing train"
    )
    eval_ds = eval_ds.map(
        tokenize, batched=True, remove_columns=["text"], desc="Tokenizing eval"
    )
    return train_ds, eval_ds


def train_dapt(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the DAPT MLM stage.

    Args:
        config: Configuration dictionary. DAPT options are merged at the
            top level by the caller.

    Returns:
        Training summary dictionary (persisted as the DAPT training log).
    """
    start_time = time.perf_counter()

    model_name: str = config.get("base_model_name", "BAAI/bge-m3")
    device: str = resolve_device(config.get("device", "cuda"))
    seed: int = int(config.get("seed", 42))

    mlm_probability: float = config.get("mlm_probability", 0.15)
    max_length: int = int(config.get("max_length", 256))
    epochs: int = int(config.get("epochs", 3))
    batch_size: int = int(config.get("batch_size", 8))
    learning_rate: float = float(config.get("learning_rate", 2e-5))
    weight_decay: float = float(config.get("weight_decay", 0.01))
    warmup_ratio: float = float(config.get("warmup_ratio", 0.06))
    grad_accum: int = int(config.get("gradient_accumulation_steps", 1))
    grad_ckpt: bool = bool(config.get("gradient_checkpointing", True))
    fp16: bool = bool(config.get("fp16", True) and device != "cpu")
    logging_steps: int = int(config.get("logging_steps", 50))
    eval_ratio: float = float(config.get("eval_ratio", 0.02))
    save_strategy: str = config.get("save_strategy", "epoch")
    save_total_limit: int = int(config.get("save_total_limit", 2))

    outputs_dir = Path(config.get("outputs_dir", "outputs"))
    dapt_model_dir = Path(config.get("dapt_model_dir", "models/bge-m3-dapt"))
    trainer_dir = outputs_dir / "dapt_trainer"

    # --- Data ---
    samples = load_reviews(config)
    if not samples:
        raise ValueError("No MLM samples found. Check the DAPT data file.")

    model, tokenizer = load_mlm_model(config)

    print("\n" + "=" * 60)
    print("  STAGE 1: Domain-Adaptive Pretraining (MLM)")
    print("=" * 60)
    print(f"[INFO] Dataset size      : {len(samples)}")
    print(f"[INFO] MLM samples       : {len(samples)}")
    print(f"[INFO] Base model        : {model_name}")
    print(f"[INFO] Device            : {device}")
    print(f"[INFO] MLM probability   : {mlm_probability}")
    print(f"[INFO] Max length        : {max_length}")
    print(f"[INFO] Learning rate     : {learning_rate}")
    print(f"[INFO] Batch size        : {batch_size}")
    print(f"[INFO] Epochs            : {epochs}")

    train_ds, eval_ds = _tokenize_dataset(samples, tokenizer, max_length, eval_ratio, seed)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    # 'eval_strategy' superseded 'evaluation_strategy' in transformers 4.41+
    from packaging.version import parse

    import transformers

    eval_kwargs: Dict[str, Any] = {}
    if parse(transformers.__version__) >= parse("4.41"):
        eval_kwargs["eval_strategy"] = save_strategy
    else:
        eval_kwargs["evaluation_strategy"] = save_strategy

    training_args = TrainingArguments(
        output_dir=str(trainer_dir),
        seed=seed,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=grad_ckpt,
        fp16=fp16,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        **eval_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    print("[INFO] Starting MLM training ...")
    trainer.train()

    train_losses = [item.get("loss") for item in trainer.state.log_history if item.get("loss")]
    eval_losses = [
        item.get("eval_loss") for item in trainer.state.log_history if item.get("eval_loss")
    ]

    # --- Save best and latest checkpoints ---
    best_dir = outputs_dir / "checkpoint-best"
    last_dir = outputs_dir / "checkpoint-last"
    shutil.rmtree(best_dir, ignore_errors=True)
    shutil.rmtree(last_dir, ignore_errors=True)

    if (trainer_dir / "checkpoint-best").exists():
        shutil.copytree(trainer_dir / "checkpoint-best", best_dir)
    if (trainer_dir / "checkpoint-last").exists():
        shutil.copytree(trainer_dir / "checkpoint-last", last_dir)

    print(f"[INFO] Saved best checkpoint  -> {best_dir}")
    print(f"[INFO] Saved last checkpoint  -> {last_dir}")

    # --- Save final model (loadable via AutoModel) ---
    dapt_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(dapt_model_dir))
    tokenizer.save_pretrained(str(dapt_model_dir))
    with open(dapt_model_dir / "training_args.json", "w", encoding="utf-8") as f:
        json.dump(training_args.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved final DAPT model -> {dapt_model_dir}")

    training_time = time.perf_counter() - start_time
    print(f"[INFO] Elapsed time: {training_time:.2f} sec")

    summary: Dict[str, Any] = {
        "stage": "dapt",
        "model_name": model_name,
        "dataset_size": len(samples),
        "mlm_samples": len(samples),
        "epochs": epochs,
        "train_loss_per_epoch": [round(v, 6) for v in train_losses],
        "eval_loss_per_epoch": [round(v, 6) for v in eval_losses],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_length": max_length,
        "mlm_probability": mlm_probability,
        "optimizer": "adamw_torch",
        "scheduler": "linear",
        "fp16": fp16,
        "training_time_sec": round(training_time, 2),
        "best_checkpoint": str(best_dir),
        "last_checkpoint": str(last_dir),
        "model_dir": str(dapt_model_dir),
    }

    log_path = outputs_dir / "training_log_dapt.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved DAPT training log to {log_path}")

    return summary
