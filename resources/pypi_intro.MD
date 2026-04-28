## SPAR: Semantic Projection with Active Retrieval

SPAR scores short text on bipolar concepts you define as `positive_seeds - negative_seeds`.
No model training or fine-tuning required.

**Reference:** Yan, Bei, Feng Mai, Chaojiang Wu, Rui Chen, and Xiaolin Li (2024).
"A Computational Framework for Understanding Firm Communication During Disasters."
*Information Systems Research* 35(2): 590-608. https://doi.org/10.1287/isre.2022.0128

---

## Install

```bash
pip install -U spar-measure
```

Optional extras:

```bash
pip install "spar-measure[vector]"   # ChromaDB persistence for large corpora
pip install "spar-measure[dev]"      # pytest + gradio_client for contributing
```

Python 3.10 or later.

---

## GUI quickstart

Launch the browser-based app:

```bash
python -m spar_measure gui
# equivalently: spar gui   or   spar-measure gui
```

Open `http://localhost:7860/` in your browser. The GUI walks through five steps:
upload a CSV, embed, define dimension seeds, run active retrieval to refine seeds,
define scales (positive pole minus negative pole), and score. When you click
*Save Scales*, the GUI writes a `scales.json` file that the headless `score()` API
accepts directly.

Run headless in Google Colab:

[![GUI in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maifeng/SPAR_measure/blob/master/resources/example_colab.ipynb)

---

## Headless `score()` quickstart

Once seeds are stable (exported from the GUI or written by hand), call `score()`
directly without launching Gradio:

```python
import pandas as pd
from spar_measure import score

docs = pd.DataFrame({
    "doc_id": [0, 1, 2],
    "text": [
        "We encourage new ways of thinking.",
        "Quarterly results exceeded analyst expectations.",
        "We honor the founders' commitment to quality.",
    ],
})

scales = {
    "dimensions": {
        "Innovation": {"queries": ["We constantly experiment with new ideas.",
                                   "Innovation drives everything we do."]},
        "Tradition":  {"queries": ["We honor the practices that built this company.",
                                   "Our heritage and craft define who we are."]},
    },
    "scales": {
        "Innovation-Tradition": {"pos_dims": ["Innovation"], "neg_dims": ["Tradition"]},
    },
}

out = score(docs, scales, text_col="text", id_col="doc_id")
print(out)
```

Headless Colab notebook (no API key required, runs on CPU in ~60 seconds):

[![Headless API in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maifeng/SPAR_measure/blob/master/resources/example_colab_headless.ipynb)

---

## ChromaStore: persistent embeddings for large corpora

For 50k+ document corpora, install the `[vector]` extra and persist embeddings to disk:

```python
from spar_measure.vector_store import ChromaStore
from spar_measure import score

# Embed once.
store = ChromaStore("my_corpus", persist_dir="/data/chroma")
store.embed_and_store(docs_df, text_col="text")

# Load and score on subsequent runs (no re-embedding).
store = ChromaStore.load("/data/chroma", "my_corpus")
out = score(docs_df, scales, text_col="text", id_col="doc_id",
            precomputed_embeddings=store.get_all_embeddings())
```

---

## Citation

```bibtex
@article{yan2024spar,
  author  = {Yan, Bei and Mai, Feng and Wu, Chaojiang and Chen, Rui and Li, Xiaolin},
  title   = {A Computational Framework for Understanding Firm Communication During Disasters},
  journal = {Information Systems Research},
  volume  = {35},
  number  = {2},
  pages   = {590--608},
  year    = {2024},
  doi     = {10.1287/isre.2022.0128}
}
```

Source code and documentation: https://github.com/maifeng/SPAR_measure
