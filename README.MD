# spar_measure

**Measure short documents against theoretical concepts, without training a model.**

SPAR (Semantic Projection with Active Retrieval) is the reference implementation for Yan, Mai, Wu, Chen, and Li (2024), "A Computational Framework for Understanding Firm Communication During Disasters," *Information Systems Research* 35(2):590-608, [doi.org/10.1287/isre.2022.0128](https://doi.org/10.1287/isre.2022.0128). The package is domain-agnostic: bring your own corpus and your own concepts.

---

## The idea in one example

You have this DataFrame of short texts:

| doc_id | text |
|---|---|
| 0 | We encourage new ways of thinking. |
| 1 | Safety first in every department. |
| 2 | Working together to weather the storm. |

You want numeric scores for theoretical concepts like `Creativity` and `Safety`:

| doc_id | Creativity | Safety |
|---|---|---|
| 0 | 0.71 | 0.18 |
| 1 | 0.22 | 0.83 |
| 2 | 0.49 | 0.31 |

You supply short generic seed sentences that define each concept ("We should adapt and innovate.", "Safety is our top priority."), embed the corpus and the seeds with a pretrained sentence model, then project each document vector onto each concept direction. One function call does the whole pipeline.

```python
from spar_measure import score

out = score(docs, scales, text_col="text", id_col="doc_id")
```

The full runnable code with the exact scales dict is in the [90-second tour](#90-second-tour) below.

---

## Why not just train a classifier?

A supervised classifier needs labels. For most theoretical constructs, labels do not exist, and hand-coding thousands of documents per construct is expensive. Zero-shot LLM prompting works but costs per document and drifts across model versions.

SPAR takes a third path. You never train and you never label. You:

1. Write two or three generic seed sentences per concept (roughly five minutes per concept).
2. Embed the corpus once with a frozen pretrained model (Sentence-BERT by default, or OpenAI Embeddings).
3. Use the seed embedding to retrieve the most similar sentences from your own corpus. This is the "active retrieval" step. You review those, keep the in-domain ones, and feed them back as refined queries.
4. Average the refined queries into a concept vector, then score each document by its dot product with that vector.

The result is a continuous score per concept per document, grounded in your domain, produced without labels and without any gradient updates. The paper reports the method on Russell 3000 Facebook posts during disasters; the same pipeline applies to any corpus of short texts.

---

## Install

```bash
pip install -U spar_measure
```

Optional extras:

```bash
pip install "spar_measure[vector]"   # ChromaDB persistence for big corpora
pip install "spar_measure[dev]"      # pytest + gradio_client for contributing
```

Python 3.10 or later. GPU is used automatically when available. To force CPU:

```bash
CUDA_VISIBLE_DEVICES="" python -m spar_measure gui
```

---

## 90-second tour

A minimal end-to-end run with three documents and three dimensions combined into two scales.

```python
import pandas as pd
from spar_measure import score

docs = pd.DataFrame({
    "doc_id": [0, 1, 2],
    "text": [
        "We encourage new ways of thinking.",
        "Safety first in every department.",
        "Working together to weather the storm.",
    ],
})

scales = {
    "dimensions": {
        "Creative": {"queries": ["We should adapt and innovate."]},
        "Safe":     {"queries": ["Safety is our top priority."]},
        "Danger":   {"queries": ["This is dangerous."]},
    },
    "scales": {
        "Creativity": {"pos_dims": ["Creative"], "neg_dims": []},
        "Safety":     {"pos_dims": ["Safe"],     "neg_dims": ["Danger"]},
    },
}

out = score(docs, scales, text_col="text", id_col="doc_id")
print(out)
```

Two data structures drive everything:

- `dimensions` maps a dimension name to a list of `queries` (the refined seed sentences).
- `scales` combines one or more dimensions into a signed construct. `Safety = Safe - Danger` is the textbook bipolar scale.

Swap the embedding backend to OpenAI with one extra keyword:

```python
out = score(
    docs, scales,
    text_col="text", id_col="doc_id",
    openai_api_key="sk-...",                              # switches backend
    openai_embedding_model="text-embedding-3-large",      # optional
)
```

Reuse an embedding matrix you have already computed:

```python
import numpy as np

embeddings = np.load("corpus_embeddings.npy")   # shape: (n_docs, dim)
out = score(
    docs, scales,
    text_col="text", id_col="doc_id",
    precomputed_embeddings=embeddings,
)
```

---

## Two paths: headless or GUI

| Path | When to use | How to launch |
|---|---|---|
| `score(...)` headless | Colab, batch scripts, reproducible pipelines, large corpora | `from spar_measure import score` |
| Gradio web UI | First-time use, exploring seeds, live active retrieval, teaching | `python -m spar_measure gui` |

The UI is the best place to discover good queries for a new construct. It walks through upload, embed, seed, retrieve, refine, define scales, and score in a browser at `http://localhost:7860/`. Once your scale definitions stabilize, export them and rerun the same scales headlessly on the full corpus.

The UI flow, tab by tab:

1. Upload a CSV with a text column and an ID column. Pick an embedding model. Embed.
2. For each dimension, write a generic seed sentence, retrieve the most similar lines from your corpus, mark the ones that fit, and use them as new queries. Repeat until the retrieved set looks in-domain.
3. Combine dimensions into signed scales (positive dimensions minus negative dimensions).
4. Score all documents. Download the result as CSV.

Screenshots live under `resources/imgs/`.

---

## The `score()` knobs

```python
score(
    docs,                               # pandas DataFrame
    scales,                             # dict: dimensions + scales
    text_col="text",                    # column holding the text
    id_col="doc_id",                    # column holding the ID (auto-added if None)
    embedding_model=None,               # e.g. "all-MiniLM-L6-v2" or "text-embedding-3-large"
    openai_api_key=None,                # supply to switch to OpenAI embeddings
    openai_embedding_model="text-embedding-3-small",
    whiten=False,                       # apply ZCA whitening to the score matrix
    single_subspace=False,              # joint-subspace projection (S Sᵀ)⁺ S X
    batch_size=8,                       # embedding batch size
    precomputed_embeddings=None,        # (n_docs, dim) np.ndarray to skip embedding
)
```

Returns a DataFrame of shape `(n_docs, 1 + n_scales)` with the ID column first and one column per scale. Full signature and Google-style docstring live in `src/spar_measure/api.py`.

---

## Headless CLI

For one-off scoring from the shell:

```bash
python -m spar_measure measure \
    --docs=corpus.csv \
    --scales=scales.json \
    --out=scored.csv \
    --id_col=doc_id
```

`scales.json` has the same shape as the `scales` dict above.

---

## Limitations

- SPAR represents each document with a single vector, so it struggles on long or multi-topic documents. Pre-split long documents into sentences or paragraphs and score those.
- Sentence embeddings encode meaning, not syntax. Constructs that hinge on syntactic features (negation polarity, modal voice, tense) are harder.
- Pretrained models carry a fixed world knowledge cutoff. New vocabulary and new entities will be weakly represented.
- Semantic projection is linear. Highly nonlinear constructs may be better served by supervised fine-tuning when labels are available.

---

## Origin and attribution

Originally built for:

> Yan, Bei, Feng Mai, Chaojiang Wu, Rui Chen, and Xiaolin Li (2024).
> "A Computational Framework for Understanding Firm Communication During Disasters."
> *Information Systems Research* 35(2):590-608.
> [doi.org/10.1287/isre.2022.0128](https://doi.org/10.1287/isre.2022.0128)

SPAR itself has no knowledge of firms, disasters, or any particular domain. It is a general-purpose measurement tool for short text against concept seeds. Bring your own corpus and your own scales.

Built on top of [Hugging Face Transformers](https://huggingface.co/transformers/), [Sentence-Transformers](https://www.sbert.net/), and [Gradio](https://gradio.app/).

## License

GPL-3.0.
