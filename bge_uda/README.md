# BGE-UDA — Informal → Formal Embedding Adaptation

Unsupervised Embedding Adaptation (UDA) for Indonesian informal queries on top of `BAAI/bge-m3`.

## Idea in one line

Baseline BGE-M3 is trained on formal text, so `gmn ngobatin demam?` lands far from
`Bagaimana cara mengobati demam?`. We adapt it so **informal queries produce
formal-like embeddings** and retrieve the same documents — without labeling new qrels.

## Stages

| Stage | Script | Input | Output |
|---|---|---|---|
| 0 Baseline | `python -m src.pipeline --stage baseline` | MIRACL `corpus.jsonl/query.tsv/qrels.tsv` | `embeddings/`, `faiss/index.bin`, `evaluation/results.json` |
| 1 DAPT (unsupervised MLM) | `python -m src.pipeline --stage dapt` | corpus passages + `train_pairs.csv` formal+informal | `models/bge-m3-dapt/` |
| 2 Make pairs | `python -m src.pipeline --stage make_pairs` | `train_pairs.csv` + qrels + FAISS | `data/training_pairs.json` |
| 3 LoRA contrastive | `python -m src.pipeline --stage train` | `training_pairs.json` | `adapters/bge-m3-lora/` |
| 4 Eval informal | `python -m src.pipeline --stage eval_informal` | same index, `use_adapter: true` | cosine gain + Recall/MRR table |
| Demo | `python -m src.pipeline --stage demo` | two strings | cosine/euclidean/dot |

## Key UDA trick

`train_pairs.csv` gives `(query_id, formal, informal)` but no informal relevance labels.
We **reuse the formal query's relevant doc** as the informal query's positive:

```
informal_query -> SAME positive passage as formal query
```

Hard negatives are mined with the baseline FAISS index (top-K excluding positives).

## Config

Edit only `config/config.yaml` — model path, LoRA rank, LR, epochs, paths.
Set `use_adapter: true` after training to evaluate the adapted model.
Set `model_name: models/bge-m3-dapt` to evaluate the DAPT checkpoint.
