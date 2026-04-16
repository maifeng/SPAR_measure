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
#
# **Corpus**: 2,000 Glassdoor "pros" reviews about corporate culture,
# sampled from the RFS 2026 validation dataset. The same corpus is used
# across all three workshop notebooks for comparability.

# %% [markdown]
# ## 1. Install and download the corpus

# %%
# !pip install -q spar-measure    # uncomment and run in Colab

# %%
# Download the shared workshop corpus and pre-computed embeddings (uncomment in Colab):
# !wget -q https://raw.githubusercontent.com/maifeng/culture-llm-workshop/main/data/glassdoor_culture_2000.csv
# !wget -q https://raw.githubusercontent.com/maifeng/culture-llm-workshop/main/data/glassdoor_culture_2000_emb.npy

# %% [markdown]
# ## 2. Load the corpus

# %%
import os
import pandas as pd
import numpy as np
from spar_measure import score

CORPUS_PATH = "glassdoor_culture_2000.csv"
EMB_PATH = "glassdoor_culture_2000_emb.npy"
if not os.path.exists(CORPUS_PATH):
    CORPUS_PATH = "../../../data/glassdoor_culture_2000.csv"
    EMB_PATH = "../../../data/glassdoor_culture_2000_emb.npy"

docs = pd.read_csv(CORPUS_PATH)
print(f"Corpus: {len(docs)} reviews, {docs['firm_id'].nunique()} firms")
docs[["review_id", "text"]].head(3)

# %%
embeddings = np.load(EMB_PATH)
print(f"Pre-computed embeddings: {embeddings.shape}")
# These were computed with all-MiniLM-L6-v2, the same model SPAR uses
# by default. If you skip this step, score() embeds from scratch (~30s).

# %% [markdown]
# ## 3. Define dimensions and scales
#
# SPAR works with the Competing Values Framework (CVF). We define four
# poles as seed sentences, then combine them into two bipolar scales:
# External-Internal and Flexible-Stable.
#
# You can write any seed sentences that capture your theoretical construct.
# More seeds per dimension improve coverage but are not strictly necessary.

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
# ## 4. Score the corpus
#
# One function call. Each output column is a bipolar scale. Positive
# scores mean the text leans toward the positive poles (e.g.,
# Create + Compete for External-Internal); negative scores lean toward
# the negative poles (Collaborate + Control).

# %%
out = score(
    docs,
    scales,
    text_col="text",
    id_col="review_id",
    precomputed_embeddings=embeddings,
)
out.head(10)

# %%
out.to_csv("spar_glassdoor_scores.csv", index=False)
print("Saved to spar_glassdoor_scores.csv")

# %% [markdown]
# ## 5. Inspect the distribution

# %%
out[["External-Internal", "Flexible-Stable"]].describe()

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, ["External-Internal", "Flexible-Stable"]):
    ax.hist(out[col], bins=40, edgecolor="white", alpha=0.8, color="#9E1B32")
    ax.set_title(col)
    ax.set_xlabel("Score")
fig.suptitle("SPAR scores on 2,000 Glassdoor culture reviews", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Merge scores with metadata
#
# Since we scored the same corpus used by the other two notebooks, we
# can merge SPAR scores with the original metadata (culture rating,
# overall rating, year) and look for patterns.

# %%
merged = docs.merge(out, on="review_id")
print(f"Correlation: External-Internal vs rating_culture = "
      f"{merged['External-Internal'].corr(merged['rating_culture']):.3f}")
print(f"Correlation: Flexible-Stable vs rating_culture = "
      f"{merged['Flexible-Stable'].corr(merged['rating_culture']):.3f}")

# %% [markdown]
# ## 7. ZCA whitening
#
# ZCA whitening decorrelates the scale scores so that External-Internal
# and Flexible-Stable become orthogonal. This matters when your scales
# share dimensions or when you use the scores as regressors.

# %%
out_whitened = score(
    docs,
    scales,
    text_col="text",
    id_col="review_id",
    precomputed_embeddings=embeddings,
    whiten=True,
)
print("Correlation (raw):",
      out[["External-Internal", "Flexible-Stable"]].corr().iloc[0, 1].round(3))
print("Correlation (ZCA):",
      out_whitened[["External-Internal", "Flexible-Stable"]].corr().iloc[0, 1].round(3))

# %% [markdown]
# ## 8. Score without pre-computed embeddings
#
# If you do not have pre-computed embeddings, `score()` embeds the corpus
# from scratch using the default `all-MiniLM-L6-v2` model. Uncomment the
# cell below to try (takes ~30s on Colab T4 for 2,000 docs).

# %%
# out_fresh = score(docs, scales, text_col="text", id_col="review_id")

# %% [markdown]
# ## 9. Define your own construct
#
# SPAR is not limited to CVF. Any construct that can be expressed as
# seed sentences works. Here is a simple "people-focused" vs
# "performance-focused" scale, which maps onto two of the six culture
# types from the RFS 2026 paper.

# %%
custom_scales = {
    "dimensions": {
        "People": {
            "queries": [
                "We care about our employees and their well-being.",
                "The company invests in people, not just profits.",
            ],
        },
        "Performance": {
            "queries": [
                "Results are what matter here.",
                "We measure everything and hold people accountable.",
            ],
        },
    },
    "scales": {
        "People-Performance": {
            "pos_dims": ["People"],
            "neg_dims": ["Performance"],
        },
    },
}

out_custom = score(
    docs,
    custom_scales,
    text_col="text",
    id_col="review_id",
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
# This workshop covers three tools on the **same 2,000 Glassdoor reviews**.
# Pick the one that fits your research question:
#
# | Package | Best for | Runtime |
# |---|---|---|
# | **`lmsyz_genai_ie_rfs`** | Structured extraction: culture type, causes, consequences, causal triples | Requires an LLM API key |
# | **`spar_measure`** (this notebook) | Scoring short texts on a custom semantic scale (e.g., CVF dimensions) | Local CPU/GPU, no API key |
# | **`lmsy_w2v_rfs`** | Historical, deterministic 5-dimension culture scores from word2vec | Local CPU, no API key |
