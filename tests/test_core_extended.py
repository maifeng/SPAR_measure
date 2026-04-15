# SPDX-License-Identifier: GPL-3.0-or-later
"""Extended unit tests for :mod:`spar_measure.core`.

All tests construct toy embedding matrices and run in <1s without loading
any Hugging Face model or making network calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spar_measure.core import (
    _iter_batches,
    build_dim_embedding,
    combine_scale,
    project_documents,
)


# ---------------------------------------------------------------------------
# combine_scale: multiple dims
# ---------------------------------------------------------------------------


def test_combine_scale_three_positives() -> None:
    """Three positive dimensions; result is their mean vector.

    The mean of ``[1,0,0]``, ``[0,1,0]``, ``[0,0,1]`` is ``[1/3, 1/3, 1/3]``.
    """
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 1.0])
    out = combine_scale([p1, p2, p3], [])
    np.testing.assert_allclose(out, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)


def test_combine_scale_three_negatives() -> None:
    """Three negative dimensions; result is ``-mean([v1, v2, v3])``."""
    n1 = np.array([2.0, 0.0])
    n2 = np.array([0.0, 2.0])
    n3 = np.array([1.0, 1.0])
    out = combine_scale([], [n1, n2, n3])
    expected = -np.array([1.0, 1.0])  # -mean([2,0],[0,2],[1,1]) = -[1,1]
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_combine_scale_multiple_pos_and_neg() -> None:
    """Two positive dims and two negative dims; result is ``mean(pos) - mean(neg)``."""
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.0, 1.0])
    n1 = np.array([1.0, 0.0])
    n2 = np.array([0.0, 1.0])
    # mean(pos) = [0.5, 0.5], mean(neg) = [0.5, 0.5], diff = [0, 0]
    out = combine_scale([p1, p2], [n1, n2])
    np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# project_documents: edge cases
# ---------------------------------------------------------------------------


def test_project_documents_single_row() -> None:
    """``project_documents`` with a 1-row input returns shape ``(1, n_scales)``."""
    x = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    scales = {
        "A": np.array([1.0, 0.0, 0.0]),
        "B": np.array([0.0, 1.0, 0.0]),
    }
    out = project_documents(x, scales, single_subspace=False)
    assert out.shape == (1, 2)
    assert list(out.columns) == ["A", "B"]


def test_project_documents_single_scale_single_subspace() -> None:
    """``single_subspace=True`` with exactly one scale does not crash.

    A single scale forms a 1-dimensional subspace; the pseudoinverse of a
    ``(1,1)`` matrix is its reciprocal, which is well-defined.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    scales = {"Solo": np.array([1.0, 0.0, 0.0, 0.0])}
    out = project_documents(X, scales, single_subspace=True)
    assert out.shape == (10, 1)
    assert np.isfinite(out.to_numpy()).all()


def test_project_documents_empty_scales_raises() -> None:
    """``project_documents`` with empty ``scale_embeddings`` raises ``ValueError``."""
    X = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        project_documents(X, {})


def test_scores_in_unit_range_after_dot_product() -> None:
    """Dot-product scores are in ``[-1, 1]`` for L2-normalized inputs and scales.

    When both document embeddings and scale vectors are unit-norm, the dot
    product equals cosine similarity, which is bounded in ``[-1, 1]``.
    ``project_documents`` L2-normalizes scale vectors internally.
    """
    rng = np.random.default_rng(7)
    n, dim = 50, 16
    X = rng.standard_normal((n, dim)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # Scales are also unit vectors.
    s = rng.standard_normal((3, dim))
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    scales = {"A": s[0], "B": s[1], "C": s[2]}
    out = project_documents(X, scales, single_subspace=False)
    vals = out.to_numpy()
    assert (vals >= -1.0 - 1e-9).all(), "Score below -1"
    assert (vals <= 1.0 + 1e-9).all(), "Score above +1"


# ---------------------------------------------------------------------------
# build_dim_embedding: patched embed_texts
# ---------------------------------------------------------------------------


def test_build_dim_embedding_mean_of_sentences() -> None:
    """``build_dim_embedding`` returns the mean of individual sentence embeddings.

    We patch ``embed_texts`` so that each query maps to a deterministic unit
    vector (row of identity matrix), then verify the output equals the mean.
    """
    dim = 4
    q1 = "sentence one"
    q2 = "sentence two"

    # embed_texts returns a (2, 4) array: first row e0, second row e1.
    fake_emb = np.eye(dim, dtype=np.float32)[:2]  # shape (2, 4)

    with patch("spar_measure.core.embed_texts", return_value=fake_emb):
        result = build_dim_embedding(
            [q1, q2],
            use_openai=False,
            sbert=MagicMock(),  # not called; embed_texts is patched
        )

    expected = fake_emb.mean(axis=0)
    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_build_dim_embedding_single_query() -> None:
    """``build_dim_embedding`` with a single query returns that embedding unchanged.

    Mean of a single row is the row itself.
    """
    dim = 4
    fake_emb = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)

    with patch("spar_measure.core.embed_texts", return_value=fake_emb):
        result = build_dim_embedding(
            ["just one query"],
            use_openai=False,
            sbert=MagicMock(),
        )

    np.testing.assert_allclose(result, fake_emb[0], atol=1e-7)


def test_build_dim_embedding_empty_queries_raises() -> None:
    """``build_dim_embedding`` with only whitespace queries raises ``ValueError``."""
    with pytest.raises(ValueError, match="query"):
        build_dim_embedding(
            ["   ", ""],
            use_openai=False,
            sbert=MagicMock(),
        )


# ---------------------------------------------------------------------------
# _iter_batches internals
# ---------------------------------------------------------------------------


def test_iter_batches_covers_all_elements() -> None:
    """``_iter_batches`` yields all elements, no gaps, no duplicates."""
    seq = list(range(10))
    batches = list(_iter_batches(seq, batch_size=3))
    flat = [x for b in batches for x in b]
    assert flat == seq
    # Last batch is shorter.
    assert len(batches[-1]) == 1  # 10 % 3 == 1


def test_iter_batches_single_element() -> None:
    """``_iter_batches`` on a 1-element list yields exactly one batch of one."""
    batches = list(_iter_batches(["a"], batch_size=1))
    assert batches == [["a"]]


def test_iter_batches_batch_larger_than_seq() -> None:
    """When ``batch_size`` exceeds sequence length, exactly one batch is yielded."""
    batches = list(_iter_batches([1, 2, 3], batch_size=100))
    assert len(batches) == 1
    assert batches[0] == [1, 2, 3]
