# Add Retrieval Inference Module

## Objective

Extend the existing project by adding a standalone retrieval inference module.

The project currently contains:

```text
src/
└── inference/
    └── main.py
```

which is used to compare embeddings between two queries.

Create a new module:

```text
src/
└── retrieval/
    └── main.py
```

This module will perform semantic document retrieval using the embedding model and the pre-built FAISS index.

The purpose of this module is to manually test retrieval quality before and after future model adaptation (e.g., DAPT).

---

# Goal

Allow users to enter a free-text query.

Example

```text
Enter Query:

gmn ngobatin demam
```

The application should retrieve the most relevant documents from the indexed corpus and display them in ranked order.

---

# Project Structure

```text
src/

├── inference/
│   └── main.py
│
├── retrieval/
│   ├── main.py
│   ├── retriever.py
│   ├── faiss_loader.py
│   ├── search.py
│   └── formatter.py
```

---

# Input

Prompt the user to enter a query.

Example

```text
Enter Query:

gmn install docker
```

---

# Model Loading

Reuse the existing embedding model loader whenever possible.

The retrieval module must support loading any compatible embedding checkpoint.

Examples

```text
BAAI/bge-m3

models/bge-m3-dapt

models/final-model
```

The model should only be loaded once during application startup.

---

# Index Loading

Load the existing FAISS index.

The retrieval module must not rebuild embeddings during inference.

Expected resources include:

* FAISS index
* Document metadata
* Corpus mapping

Example

```text
data/

index/
    faiss.index

metadata/
    documents.json
```

---

# Retrieval Pipeline

The retrieval flow should follow this sequence:

```text
User Query

↓

Generate Query Embedding

↓

Normalize Embedding (if enabled)

↓

Search FAISS

↓

Retrieve Top-K Results

↓

Display Ranked Documents
```

---

# Configurable Parameters

Configuration should support:

```yaml
top_k: 5

normalize_embeddings: true

device: cuda

show_similarity_score: true

show_document_text: true

show_document_title: true
```

---

# Output

Example

```text
=======================================================

Semantic Retrieval Result

=======================================================

Model

BAAI/bge-m3

-------------------------------------------------------

Query

gmn install docker

-------------------------------------------------------

Top 5 Documents

-------------------------------------------------------

Rank 1

Similarity : 0.9234

Document ID : 28193

Title :

Cara Install Docker pada Windows

Content

Docker dapat diinstal dengan...

-------------------------------------------------------

Rank 2

Similarity : 0.8712

Document ID : 8123

Title :

Panduan Docker Desktop

Content

Docker Desktop merupakan...

-------------------------------------------------------

Rank 3

...
```

---

# Optional Metadata

Display additional retrieval information.

Example

```text
Embedding Time

0.028 sec

Search Time

0.004 sec

Total Time

0.032 sec

Retrieved Documents

5
```

---

# Save Results

If enabled, save retrieval results.

Example

```text
outputs/

retrieval_result.json
```

Example JSON

```json
{
    "model": "BAAI/bge-m3",
    "query": "gmn install docker",
    "top_k": 5,
    "results": [
        {
            "rank": 1,
            "document_id": "28193",
            "score": 0.9234,
            "title": "...",
            "text": "..."
        }
    ]
}
```

---

# Logging

Log the following events:

* Model loaded successfully
* FAISS index loaded
* Metadata loaded
* Query received
* Embedding generated
* Retrieval completed
* Results saved

---

# Error Handling

Handle the following scenarios gracefully:

* Missing FAISS index
* Missing metadata
* Missing model
* Empty query
* Invalid configuration
* GPU unavailable (fallback to CPU if configured)

---

# Coding Requirements

* Python 3.11+
* Modular architecture
* Type hints
* Clean code
* Reuse existing embedding utilities whenever possible
* Proper exception handling
* Structured logging

---

# Future Compatibility

The retrieval module should work with any embedding model that follows the same embedding interface.

Examples include:

```text
BAAI/bge-m3

BGE + DAPT

BGE + Contrastive Learning

BGE + DANN

Final Research Model
```

No retrieval code should require modification when switching models. Only the model path in the configuration should change.

---

# Purpose

This retrieval module will serve as the primary qualitative evaluation tool throughout the research.

It should allow rapid comparison of retrieval behavior between different embedding checkpoints, making it possible to observe whether model adaptation improves retrieval for informal Indonesian queries without changing the retrieval pipeline itself.
