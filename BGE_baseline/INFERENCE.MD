# BGE-M3 Embedding Similarity Inference Tool

## Objective

Create a standalone inference utility that compares the semantic similarity between **two Indonesian queries** using a sentence embedding model.

The tool will initially use the pretrained **BAAI/bge-m3** model but must be designed so that any future checkpoint (e.g., DAPT-adapted model) can be loaded by changing only the model path.

---

# Goal

Input:

Two text queries.

Example

```text
Query 1:
Bagaimana cara mengobati demam?

Query 2:
gmn ngobatin demam?
```

Output:

* Embedding vector for Query 1
* Embedding vector for Query 2
* Cosine Similarity
* Euclidean Distance
* Vector Dimension
* Basic embedding statistics

---

# Project Structure

```text
embedding_similarity/

│

├── config/
│   └── config.yaml
│
├── src/
│   ├── model.py
│   ├── embedding.py
│   ├── similarity.py
│   ├── utils.py
│   └── main.py
│
├── outputs/
│
└── requirements.txt
```

---

# Configuration

Example

```yaml
model_name: BAAI/bge-m3

device: cuda

normalize_embeddings: true

save_embedding: true

embedding_precision: float32
```

---

# Input

The application should allow users to enter two text queries.

Example

```text
Enter Query 1:
Bagaimana cara mengobati demam?

Enter Query 2:
gmn ngobatin demam?
```

---

# Model Loading

Load the embedding model only once.

The model loader must be reusable so future checkpoints can be loaded without changing the inference code.

Example

```text
BAAI/bge-m3
```

Future example

```text
models/bge-m3-dapt
```

---

# Embedding Generation

Generate embeddings for

* Query 1
* Query 2

Return

```text
Embedding Dimension

Embedding Vector

L2 Norm
```

If normalization is enabled,

normalize embeddings before similarity calculation.

---

# Similarity Metrics

Implement

## 1. Cosine Similarity

Return

```text
0.0000 - 1.0000
```

Higher value indicates higher semantic similarity.

---

## 2. Euclidean Distance

Return

```text
Distance
```

Lower value indicates higher similarity.

---

## 3. Dot Product

Return

```text
Score
```

(Optional but recommended.)

---

# Output Example

```text
=====================================================

Embedding Similarity Analysis

=====================================================

Model

BAAI/bge-m3

-----------------------------------------------------

Query 1

Bagaimana cara mengobati demam?

-----------------------------------------------------

Query 2

gmn ngobatin demam?

-----------------------------------------------------

Embedding Dimension

1024

-----------------------------------------------------

Cosine Similarity

0.9148

-----------------------------------------------------

Euclidean Distance

0.4121

-----------------------------------------------------

Dot Product

0.9148

=====================================================
```

---

# Save Output

If enabled,

save

```text
outputs/

embedding_query1.npy

embedding_query2.npy

result.json
```

The JSON file should contain

```json
{
    "model": "...",
    "query_1": "...",
    "query_2": "...",
    "embedding_dimension": 1024,
    "cosine_similarity": 0.9148,
    "euclidean_distance": 0.4121,
    "dot_product": 0.9148
}
```

---

# Optional Debug Information

Display

* Device (CPU/GPU)
* Inference time
* Embedding generation time
* Similarity computation time

Example

```text
Device

CUDA

Embedding Time

0.034 sec

Similarity Time

0.0003 sec
```

---

# Coding Requirements

* Python 3.11+
* HuggingFace Transformers
* PyTorch
* NumPy
* scikit-learn
* Clean code
* Type hints
* Modular architecture
* Proper logging
* Exception handling
* Progress indicators where appropriate

---

# Future Compatibility

The tool should support comparing any embedding checkpoint.

Examples

```text
BAAI/bge-m3

↓

models/bge-m3-dapt

↓

models/bge-m3-dapt-contrastive

↓

models/final-register-invariant-bge
```

No code modifications should be required beyond updating the model path in `config.yaml`.

---

# Expected Use Cases

This tool will be used throughout the research to:

* Compare pretrained BGE-M3 against DAPT-adapted models.
* Observe whether formal and informal Indonesian queries become closer in embedding space.
* Validate the impact of future Contrastive Learning and DANN stages.
* Perform qualitative analysis before full retrieval evaluation.
* Debug embedding behavior during model development.
