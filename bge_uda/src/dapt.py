"""Stage 1 DAPT: unsupervised MLM on MIRACL corpus + your pairs.

Why: BGE-M3 saw formal text. MLM on informal spellings (gmn, ngobatin, sih,
dong) teaches the backbone the target register with NO labels.
Output: models/bge-m3-dapt/ used as base for Stage 2 LoRA training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .dataset import load_corpus, load_pairs


def make_dapt_corpus(config: Dict[str, Any]) -> Path:
    """Merge corpus passages + formal + informal texts -> dapt_corpus.txt."""
    dapt_cfg = config.get("dapt", {})
    out_path = Path(dapt_cfg.get("corpus_txt", "data/dapt_corpus.txt"))
    if not out_path.is_absolute():  # resolve inside repo
        out_path = Path(__file__).resolve().parent.parent / out_path
    if out_path.exists():  # reuse: file build is deterministic
        print(f"[INFO] Reusing DAPT corpus: {out_path}")
        return out_path
    lines: List[str] = []
    # 1) All MIRACL passages = domain knowledge (formal Indonesian).
    for doc in load_corpus(config):
        if doc["passage"].strip():
            lines.append(doc["passage"].strip())
    # 2) Both sides of your pairs = register coverage (formal + informal).
    for p in load_pairs(config):
        if p["formal"].strip():
            lines.append(p["formal"].strip())
        if p["informal"].strip():
            lines.append(p["informal"].strip())
    # One sentence per line = format HF line_by_line MLM expects.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {len(lines)} lines -> {out_path}")
    return out_path


def run_dapt_mlm(config: Dict[str, Any]) -> str:
    """Run HuggingFace masked-LM training, save to dapt.output_dir."""
    from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)
    from datasets import load_dataset  # HF datasets: streams txt line by line
    import torch

    dapt = config.get("dapt", {})
    base = config.get("model_name", "BAAI/bge-m3")
    out_dir = dapt.get("output_dir", "models/bge-m3-dapt")
    # All hyperparams editable in config.yaml, no code change needed.
    epochs = int(dapt.get("epochs", 3))
    batch = int(dapt.get("batch_size", 32))
    lr = float(dapt.get("learning_rate", 5e-5))
    mlm_p = float(dapt.get("mlm_probability", 0.15))
    max_len = int(dapt.get("max_length", 512))
    seed = int(dapt.get("seed", 42))

    txt_path = make_dapt_corpus(config)  # ensure corpus file exists first
    print(f"[INFO] DAPT base={base} -> {out_dir}")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForMaskedLM.from_pretrained(base)  # MLM head added on XLM-R
    # load_dataset("text") gives {"text": line}; filter blanks.
    ds = load_dataset("text", data_files=str(txt_path))["train"].filter(lambda x: len(x["text"].strip()) > 0)

    def _tok(batch):  # tokenize + truncate to max_len
        return tok(batch["text"], truncation=True, max_length=max_len)
    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    # DataCollator masks 15% tokens dynamically each epoch (BERT-style).
    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=mlm_p)
    args = TrainingArguments(output_dir=out_dir, overwrite_output_dir=True, num_train_epochs=epochs,
                             per_device_train_batch_size=batch, learning_rate=lr, seed=seed,
                             save_total_limit=1, logging_steps=50,
                             fp16=torch.cuda.is_available())  # fp16 only if GPU present
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()
    model.save_pretrained(out_dir)  # save backbone for Stage 2
    tok.save_pretrained(out_dir)
    print(f"[INFO] DAPT done -> {out_dir}. Set model_name: {out_dir} for next stage.")
    return out_dir
