# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration tests for :func:`spar_measure.score` on the bundled sample data.

Uses the precomputed ``sample_emb.npy`` so the tests never touch the network
or load a Hugging Face model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spar_measure import score


def test_score_with_precomputed_embeddings(
    sample_docs: pd.DataFrame,
    sample_embeddings: np.ndarray,
    cvf_scales: dict,
) -> None:
    """End-to-end headless scoring reproduces the documented output shape."""
    out = score(
        sample_docs,
        cvf_scales,
        text_col="text",
        id_col="doc_id",
        precomputed_embeddings=sample_embeddings,
    )
    assert len(out) == len(sample_docs)
    assert list(out.columns) == ["doc_id", "External-Internal", "Flexible-Stable"]
    assert out["doc_id"].iloc[0] == sample_docs["doc_id"].iloc[0]
    assert np.isfinite(out[["External-Internal", "Flexible-Stable"]].to_numpy()).all()


def test_score_without_id_col_adds_doc_id(
    sample_docs: pd.DataFrame,
    sample_embeddings: np.ndarray,
    cvf_scales: dict,
) -> None:
    """When ``id_col=None``, the output carries a 0-indexed ``doc_id``."""
    out = score(
        sample_docs.drop(columns=["doc_id"]),
        cvf_scales,
        text_col="text",
        precomputed_embeddings=sample_embeddings,
    )
    assert "doc_id" in out.columns
    assert out["doc_id"].tolist() == list(range(len(sample_docs)))


def test_score_with_whitening(
    sample_docs: pd.DataFrame,
    sample_embeddings: np.ndarray,
    cvf_scales: dict,
) -> None:
    """The ``whiten`` flag produces approximately zero-mean, unit-cov scores."""
    out = score(
        sample_docs,
        cvf_scales,
        text_col="text",
        id_col="doc_id",
        precomputed_embeddings=sample_embeddings,
        whiten=True,
    )
    cols = out[["External-Internal", "Flexible-Stable"]].to_numpy()
    # ZCA whitening centers and decorrelates.
    assert abs(cols.mean(axis=0)).max() < 1e-6
    cov = np.cov(cols, rowvar=False)
    np.testing.assert_allclose(cov, np.eye(2), atol=5e-2)


def test_score_single_subspace_shape(
    sample_docs: pd.DataFrame,
    sample_embeddings: np.ndarray,
    cvf_scales: dict,
) -> None:
    """Single-subspace projection returns the same (n_docs, n_scales) shape."""
    out = score(
        sample_docs,
        cvf_scales,
        text_col="text",
        id_col="doc_id",
        precomputed_embeddings=sample_embeddings,
        single_subspace=True,
    )
    assert out.shape == (len(sample_docs), 3)


def test_score_mismatched_precomputed_raises(
    sample_docs: pd.DataFrame,
    cvf_scales: dict,
) -> None:
    """Wrong-shape precomputed embeddings are rejected up front."""
    bad = np.zeros((len(sample_docs) - 1, 384), dtype=np.float32)
    import pytest
    with pytest.raises(ValueError, match="rows"):
        score(
            sample_docs,
            cvf_scales,
            text_col="text",
            id_col="doc_id",
            precomputed_embeddings=bad,
        )
