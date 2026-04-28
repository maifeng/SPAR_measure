# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SPAR · Headless Quickstart (no GUI required)
#
# Score short documents on bipolar concepts you define as
# `positive_seeds − negative_seeds`. This notebook runs end-to-end on
# the package's bundled 2,000-document sample corpus in about a minute,
# with no API key required.
#
# For the **interactive GUI version** (with active retrieval and live
# seed refinement), open the companion notebook:
# [example_colab.ipynb](https://colab.research.google.com/github/maifeng/SPAR_measure/blob/master/resources/example_colab.ipynb).
#
# Method reference: Yan, Bei, Feng Mai, Chaojiang Wu, Rui Chen, and
# Xiaolin Li (2024). "A Computational Framework for Understanding Firm
# Communication During Disasters." *Information Systems Research*
# 35(2):590-608. https://doi.org/10.1287/isre.2022.0128

# %% [markdown]
# ## 1. Install

# %%
!pip install -q -U spar-measure --upgrade-strategy only-if-needed

# %% [markdown]
# ## 2. Load the bundled sample corpus
#
# `spar_measure` ships 2,000 Russell-3000 Facebook posts from the ISR
# paper, plus precomputed sentence embeddings so you do not wait for
# embedding on first run.

# %%
from importlib.resources import files
import numpy as np
import pandas as pd

sample = files("spar_measure.sample_data")
docs = pd.read_csv(sample / "sample_text.csv")
embeddings = np.load(sample / "sample_emb.npy")
print(f"Corpus: {len(docs)} documents")
print(f"Embeddings: shape {embeddings.shape}")
docs.head(3)

# %% [markdown]
# ## 3. Define one bipolar concept: Innovation − Tradition
#
# A *dimension* is a concept defined by one or more seed sentences.
# A *scale* combines positive and negative dimensions into a signed
# bipolar construct. Below we define `Innovation` and `Tradition` as
# two single-pole dimensions, then combine them into the bipolar scale
# `Innovation-Tradition = pos(Innovation) − neg(Tradition)`.
#
# This is the same example shown in the package's README.

# %%
scales = {
    "dimensions": {
        "Innovation": {"queries": [
            "We constantly experiment with new ideas.",
            "We embrace change and disruption.",
            "Innovation drives everything we do.",
        ]},
        "Tradition": {"queries": [
            "We honor the practices that built this company.",
            "We trust time-tested ways of working.",
            "Our heritage and craft define who we are.",
        ]},
    },
    "scales": {
        "Innovation-Tradition": {
            "pos_dims": ["Innovation"],
            "neg_dims": ["Tradition"],
        },
    },
}

# %% [markdown]
# ## 4. Score the full corpus
#
# `precomputed_embeddings=` reuses the bundled embedding matrix. Drop
# that argument and `score()` embeds the corpus from scratch using the
# default `all-MiniLM-L6-v2` model (about 30 s on a Colab T4 for 2,000
# docs).

# %%
from spar_measure import score

out = score(
    docs,
    scales,
    text_col="text",
    id_col="doc_id",
    precomputed_embeddings=embeddings,
)
out.head(10)

# %% [markdown]
# ## 5. Sanity-check on three short, hand-picked documents
#
# Same scale, three short documents. The first leans Innovation, the
# third leans Tradition, the middle is neutral business language. These
# numbers (+0.27, +0.05, −0.17) are the figures shown in the README.

# %%
demo = pd.DataFrame({
    "doc_id": [0, 1, 2],
    "text": [
        "We encourage new ways of thinking.",
        "Quarterly results exceeded analyst expectations.",
        "We honor the founders' commitment to quality.",
    ],
})
demo_out = score(demo, scales, text_col="text", id_col="doc_id")
demo_out

# %% [markdown]
# ## 6. Inspect the score distribution

# %%
out["Innovation-Tradition"].describe()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(out["Innovation-Tradition"], bins=40, edgecolor="white",
        alpha=0.85, color="#9E1B32")
ax.axvline(0, color="black", lw=1, alpha=0.4)
ax.set_xlabel("Innovation − Tradition")
ax.set_ylabel("number of documents")
ax.set_title("SPAR scores on 2,000 Russell-3000 Facebook posts")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Bring your own construct
#
# SPAR is not limited to Innovation. Any construct that can be expressed
# as a few seed sentences works: ESG concern, customer focus, risk
# tolerance, political stance, you name it.

# %%
custom_scales = {
    "dimensions": {
        "People":      {"queries": ["We care about our employees and their well-being.",
                                    "The company invests in people, not just profits."]},
        "Performance": {"queries": ["Results are what matter here.",
                                    "We measure everything and hold people accountable."]},
    },
    "scales": {
        "People-Performance": {"pos_dims": ["People"], "neg_dims": ["Performance"]},
    },
}
out_custom = score(
    docs, custom_scales,
    text_col="text", id_col="doc_id",
    precomputed_embeddings=embeddings,
)
out_custom.head(5)

# %% [markdown]
# ## 8. Hand-off from the GUI: load a `scales.json` file
#
# If you discovered your seeds in the Gradio GUI (the companion
# notebook), clicking *Save Scales* there writes a single
# `scales.json` in exactly the format `score()` expects below. Drop
# that file alongside this notebook and load it with `json.load`.
# No reshaping required.
#
# To prove the round-trip is bit-exact, the cell below writes a
# `scales.json` with the SAME definitions used in section 3 above,
# loads it back from disk, and shows that `score()` returns identical
# values.

# %%
import json
from pathlib import Path

# In a real workflow, this file would have been written by the GUI's
# "Save Scales" button. We synthesise it here so the notebook is
# self-contained.
scales_json_path = Path("/tmp/scales.json")
scales_json_path.write_text(json.dumps(scales, indent=2))
print(scales_json_path.read_text()[:400] + "  ...")

# %%
# This is the line a real user would run in their notebook after
# downloading scales.json from the GUI's Files panel.
loaded_scales = json.loads(scales_json_path.read_text())

out_from_json = score(
    docs,
    loaded_scales,
    text_col="text",
    id_col="doc_id",
    precomputed_embeddings=embeddings,
)

assert np.allclose(
    out["Innovation-Tradition"].values,
    out_from_json["Innovation-Tradition"].values,
), "GUI handoff regression: JSON-loaded spec did not reproduce inline-spec scores"
print("Round-trip OK: scales.json reproduces the inline-spec scores exactly.")
out_from_json.head(5)

# %% [markdown]
# ## 9. Decorrelate scales with ZCA whitening
#
# When two scales share dimensions, their scores are correlated by
# construction. ZCA whitening rotates them to be orthogonal while
# preserving as much of the original geometry as possible. Useful when
# you plan to feed scores into downstream regressions.
#
# We illustrate with two scales that share `Innovation` as a positive
# pole.

# %%
multi = {
    "dimensions": {
        "Innovation":  {"queries": ["We innovate."]},
        "Tradition":   {"queries": ["We follow tradition."]},
        "Performance": {"queries": ["Results are what matter."]},
    },
    "scales": {
        "Innovation-Tradition":  {"pos_dims": ["Innovation"], "neg_dims": ["Tradition"]},
        "Innovation-Performance": {"pos_dims": ["Innovation"], "neg_dims": ["Performance"]},
    },
}

raw = score(docs, multi, text_col="text", id_col="doc_id",
            precomputed_embeddings=embeddings)
whitened = score(docs, multi, text_col="text", id_col="doc_id",
                 precomputed_embeddings=embeddings, whiten=True)

cols = ["Innovation-Tradition", "Innovation-Performance"]
print(f"raw      correlation: {raw[cols].corr().iloc[0, 1]:+.3f}")
print(f"whitened correlation: {whitened[cols].corr().iloc[0, 1]:+.3f}")

# %% [markdown]
# ## 10. Switch to OpenAI embeddings
#
# To use OpenAI's hosted embedding model instead of the local
# Sentence-BERT model, pass `openai_api_key=`. SPAR re-embeds the
# corpus through the API. Costs about \$0.02 per 1,000 short documents
# at current `text-embedding-3-small` prices.

# %%
# import os
# os.environ["OPENAI_API_KEY"] = "sk-..."
# out_openai = score(
#     docs, scales, text_col="text", id_col="doc_id",
#     openai_api_key=os.environ["OPENAI_API_KEY"],
#     openai_embedding_model="text-embedding-3-small",
# )

# %% [markdown]
# ## 11. Want to iterate seeds interactively?
#
# Open the GUI companion notebook for the full active-retrieval workflow
# (search the corpus for exemplar sentences, refine seeds, re-score):
#
# [![Open GUI in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maifeng/SPAR_measure/blob/master/resources/example_colab.ipynb)
#
# Or launch the GUI locally:
#
# ```bash
# python -m spar_measure gui
# ```

# %% [markdown]
# ## 12. Citation
#
# ```
# Yan, Bei, Feng Mai, Chaojiang Wu, Rui Chen, and Xiaolin Li. 2024.
# "A Computational Framework for Understanding Firm Communication
# During Disasters." Information Systems Research 35(2):590-608.
# https://doi.org/10.1287/isre.2022.0128
# ```

# %%
import spar_measure
print(spar_measure.__paper__)
