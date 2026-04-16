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
# # spar_measure: Quickstart
#
# Semantic Projection with Active Retrieval (SPAR): score short texts
# against theory-defined semantic scales using sentence embeddings and
# dot-product projection.
#
# Originally developed for:
# Yan, Bei, Feng Mai, Chaojiang Wu, Rong Chen, and Xiaolin Li (2024),
# "A Computational Framework for Understanding Firm Communication
# During Disasters," *Information Systems Research* 35(2):590-608.
# https://doi.org/10.1287/isre.2022.0128

# %% [markdown]
# ## 1. Install (run once in Colab)

# %%
# !pip install -q spar-measure    # uncomment and run in Colab

# %% [markdown]
# ## 2. Imports

# %%
import pandas as pd
from spar_measure import score

# %% [markdown]
# ## 3. Load sample corpus
#
# The package bundles 2,000 Facebook posts (from companies affected by
# natural disasters) with pre-computed sentence-transformer embeddings.
# `importlib.resources.files()` locates files packaged inside the installed
# library, so this works on any machine without downloading anything extra.

# %%
from importlib.resources import files

data_dir = files("spar_measure") / "sample_data"
docs = pd.read_csv(str(data_dir / "sample_text.csv"))
print(f"Corpus: {len(docs)} documents")
docs.head(3)

# %% [markdown]
# ## 4. Define dimensions and scales
#
# SPAR works with the Competing Values Framework (CVF). We define four
# poles as seed sentences, then combine them into two bipolar scales:
# External-Internal and Flexible-Stable.

# %%
scales = {
    "dimensions": {
        "Create": {
            "queries": [
                "We should adapt and innovate.",
                "Creativity and agility define how we operate.",
            ],
        },
        "Collaborate": {
            "queries": [
                "We value teamwork and mutual support.",
                "Our strength comes from working together.",
            ],
        },
        "Control": {
            "queries": [
                "Strict procedures keep things running smoothly.",
                "Consistency and compliance matter most here.",
            ],
        },
        "Compete": {
            "queries": [
                "We push hard to outperform the competition.",
                "Winning market share is the top priority.",
            ],
        },
    },
    "scales": {
        "External-Internal": {
            "pos_dims": ["Create", "Compete"],
            "neg_dims": ["Collaborate", "Control"],
        },
        "Flexible-Stable": {
            "pos_dims": ["Create", "Collaborate"],
            "neg_dims": ["Compete", "Control"],
        },
    },
}

# %% [markdown]
# ## 5. Score the corpus
#
# One function call. We use the pre-computed embeddings bundled with the
# package to skip the ~30s embedding step. In your own project, omit the
# `precomputed_embeddings` argument and the library will embed from scratch
# using the default `all-MiniLM-L6-v2` model.
#
# Each output column is a bipolar scale. Positive scores mean the text
# leans toward the positive poles (e.g., Create + Compete for
# External-Internal); negative scores lean toward the negative poles
# (Collaborate + Control).

# %%
import numpy as np

embeddings = np.load(str(data_dir / "sample_emb.npy"))
print(f"Pre-computed embeddings shape: {embeddings.shape}")

out = score(
    docs,
    scales,
    text_col="text",
    id_col="doc_id",
    precomputed_embeddings=embeddings,
)
out.head(10)

# %% [markdown]
# ## 6. Inspect the distribution

# %%
out[["External-Internal", "Flexible-Stable"]].describe()

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, ["External-Internal", "Flexible-Stable"]):
    ax.hist(out[col], bins=40, edgecolor="white", alpha=0.8, color="#9E1B32")
    ax.set_title(col)
    ax.set_xlabel("Score")
fig.tight_layout()
plt.show()

# %%
out.to_csv("spar_scores.csv", index=False)
print("Saved to spar_scores.csv")

# %% [markdown]
# ## 7. Score without pre-computed embeddings
#
# If you do not have pre-computed embeddings, `score()` will embed the
# corpus from scratch. This uses the default `all-MiniLM-L6-v2` model.
# Uncomment the cell below to try (takes ~30s on Colab T4).

# %%
# out_fresh = score(docs, scales, text_col="text", id_col="doc_id")

# %% [markdown]
# ## 8. ZCA whitening and subspace projection
#
# ZCA whitening decorrelates the scale scores so that External-Internal
# and Flexible-Stable are orthogonal. The single-subspace projection
# uses a joint pseudoinverse instead of independent dot products.

# %%
out_whitened = score(
    docs,
    scales,
    text_col="text",
    id_col="doc_id",
    precomputed_embeddings=embeddings,
    whiten=True,
)
print("Correlation (raw):", out[["External-Internal", "Flexible-Stable"]].corr().iloc[0, 1].round(3))
print("Correlation (ZCA):", out_whitened[["External-Internal", "Flexible-Stable"]].corr().iloc[0, 1].round(3))

# %% [markdown]
# ## 9. Define your own construct
#
# SPAR is not limited to CVF. Any construct that can be expressed as
# seed sentences works. Here is a simple "urgency" vs "reassurance" scale.

# %%
custom_scales = {
    "dimensions": {
        "Urgent": {
            "queries": [
                "This is an emergency, act now.",
                "Immediate action is required to address the crisis.",
            ],
        },
        "Reassuring": {
            "queries": [
                "Everything is under control, do not worry.",
                "We are confident the situation will improve soon.",
            ],
        },
    },
    "scales": {
        "Urgency-Reassurance": {
            "pos_dims": ["Urgent"],
            "neg_dims": ["Reassuring"],
        },
    },
}

out_custom = score(
    docs,
    custom_scales,
    text_col="text",
    id_col="doc_id",
    precomputed_embeddings=embeddings,
)
out_custom.head(10)

# %% [markdown]
# ## 10. The Gradio GUI (interactive seed iteration)
#
# For interactive seed-sentence iteration ("active retrieval"), launch
# the GUI locally:
#
# ```python
# from spar_measure import run_gui
# run_gui()                          # on your laptop
# run_gui(share=True)                # in Colab (creates a public tunnel)
# ```
#
# The GUI lets you search the corpus for exemplar sentences, refine your
# seed queries, and re-score in real time. This is the iterative loop
# that the ISR paper describes as "active retrieval."

# %% [markdown]
# ## 11. Citation
#
# If you use this package in research, please cite:
#
# ```
# Yan, Bei, Feng Mai, Chaojiang Wu, Rong Chen, and Xiaolin Li. 2024.
# "A Computational Framework for Understanding Firm Communication
# During Disasters."
# Information Systems Research 35(2):590-608.
# https://doi.org/10.1287/isre.2022.0128
# ```

# %%
import spar_measure
print(spar_measure.__paper__)

# %% [markdown]
# ## 12. Related packages
#
# This workshop covers three tools. Pick the one that fits your research question:
#
# | Package | Best for | Runtime |
# |---|---|---|
# | **`lmsyz_genai_ie_rfs`** | Structured extraction: culture type, causes, consequences, causal triples | Requires an LLM API key |
# | **`spar_measure`** (this notebook) | Scoring short texts on a custom semantic scale (e.g., CVF dimensions) | Local CPU/GPU, no API key |
# | **`lmsy_w2v_rfs`** | Historical, deterministic 5-dimension culture scores from word2vec | Local CPU, no API key |
