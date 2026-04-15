# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that batch_size does not affect projection results.

All tests use precomputed embeddings and patched dimension embeddings so no
Hugging Face model is loaded. The batching logic in :func:`core._iter_batches`
is exercised directly as well.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from spar_measure.core import _iter_batches, measure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, seed: int) -> np.ndarray:
    """Return a random unit vector in ``R^dim``.

    Args:
        dim: Dimension.
        seed: Random seed for reproducibility.

    Returns:
        ``(dim,)`` float32 unit vector.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_dim_stub(dim: int = 8) -> Any:
    """Return a stub for ``build_dim_embedding`` that ignores model loading.

    Args:
        dim: Embedding dimension.

    Returns:
        Callable with the same signature as ``core.build_dim_embedding``.
    """

    def _stub(queries: list[str], **kwargs: Any) -> np.ndarray:
        seed = hash(tuple(sorted(queries))) % (2**31)
        return _unit_vec(dim, seed)

    return _stub


# ---------------------------------------------------------------------------
# Projection result: batch_size=1 vs batch_size=100
# ---------------------------------------------------------------------------


def test_batch_size_1_vs_100_identical_scores() -> None:
    """``project_documents`` is batch-size-agnostic for precomputed embeddings.

    ``measure()`` does not batch the corpus projection step (projection is a
    single matrix multiply), but it does batch the dimension-query embedding
    step. Because we monkeypatch ``build_dim_embedding`` here, the test
    really validates that ``project_documents`` is deterministic regardless
    of any upstream batching, and that passing different ``batch_size`` values
    does not alter the output DataFrame.
    """
    dim = 8
    n = 20
    rng = np.random.default_rng(99)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    docs = pd.DataFrame(
        {"doc_id": list(range(n)), "text": [f"doc {i}" for i in range(n)]}
    )
    scales: dict[str, Any] = {
        "dimensions": {
            "Create": {"queries": ["We should adapt and innovate."]},
            "Control": {"queries": ["We should control and stabilize."]},
        },
        "scales": {
            "Flexibility": {"pos_dims": ["Create"], "neg_dims": ["Control"]},
        },
    }

    fake_bundle = MagicMock()
    with patch("spar_measure.core.load_sbert", return_value=fake_bundle):
        with patch("spar_measure.core.build_dim_embedding", _make_dim_stub(dim)):
            out_bs1 = measure(
                docs,
                scales,
                id_col="doc_id",
                precomputed_embeddings=X,
                batch_size=1,
            )
            out_bs100 = measure(
                docs,
                scales,
                id_col="doc_id",
                precomputed_embeddings=X,
                batch_size=100,
            )

    np.testing.assert_array_equal(
        out_bs1["Flexibility"].to_numpy(),
        out_bs100["Flexibility"].to_numpy(),
    )


def test_batch_size_single_batch_equals_multi_batch_shape() -> None:
    """Output shape is identical regardless of batch_size."""
    dim = 8
    n = 15
    rng = np.random.default_rng(12)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    docs = pd.DataFrame(
        {"doc_id": list(range(n)), "text": [f"doc {i}" for i in range(n)]}
    )
    scales: dict[str, Any] = {
        "dimensions": {
            "A": {"queries": ["sentence A"]},
            "B": {"queries": ["sentence B"]},
        },
        "scales": {
            "Scale1": {"pos_dims": ["A"], "neg_dims": ["B"]},
            "Scale2": {"pos_dims": ["B"], "neg_dims": []},
        },
    }

    fake_bundle = MagicMock()
    with patch("spar_measure.core.load_sbert", return_value=fake_bundle):
        with patch("spar_measure.core.build_dim_embedding", _make_dim_stub(dim)):
            out_small = measure(
                docs, scales, id_col="doc_id", precomputed_embeddings=X, batch_size=2
            )
            out_large = measure(
                docs, scales, id_col="doc_id", precomputed_embeddings=X, batch_size=1000
            )

    assert out_small.shape == out_large.shape


# ---------------------------------------------------------------------------
# _iter_batches internals (completeness)
# ---------------------------------------------------------------------------


def test_iter_batches_total_count() -> None:
    """All elements are yielded exactly once across all batches."""
    data = list(range(17))
    collected = []
    for batch in _iter_batches(data, batch_size=5):
        collected.extend(batch)
    assert collected == data


def test_iter_batches_last_batch_shorter() -> None:
    """The final batch is shorter than ``batch_size`` when len is not divisible."""
    batches = list(_iter_batches(list(range(7)), batch_size=3))
    assert len(batches) == 3
    assert len(batches[-1]) == 1


def test_iter_batches_empty_sequence() -> None:
    """Empty input yields no batches."""
    assert list(_iter_batches([], batch_size=4)) == []


def test_iter_batches_batch_size_equals_len() -> None:
    """When batch_size equals sequence length, exactly one batch is yielded."""
    data = [1, 2, 3, 4]
    batches = list(_iter_batches(data, batch_size=4))
    assert len(batches) == 1
    assert batches[0] == data
