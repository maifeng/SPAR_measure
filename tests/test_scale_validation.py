# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for scale/dimension dict validation in :func:`spar_measure.core.measure`.

All tests use precomputed (synthetic) embeddings plus monkeypatched
``load_sbert`` and ``build_dim_embedding`` so no Hugging Face model is loaded
and no network access is required.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from spar_measure.core import measure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docs(n: int = 5) -> pd.DataFrame:
    """Return a tiny corpus DataFrame with ``doc_id`` and ``text`` columns.

    Args:
        n: Number of rows.

    Returns:
        DataFrame with columns ``["doc_id", "text"]``.
    """
    return pd.DataFrame(
        {
            "doc_id": list(range(n)),
            "text": [f"Sentence number {i}." for i in range(n)],
        }
    )


def _make_precomputed(n: int = 5, dim: int = 8) -> np.ndarray:
    """Return L2-normalized synthetic embeddings.

    Args:
        n: Number of documents.
        dim: Embedding dimension.

    Returns:
        ``(n, dim)`` float32 numpy array, each row unit-norm.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _dim_stub(dim: int = 8) -> Any:
    """Return a ``build_dim_embedding`` monkeypatch that yields unit vectors.

    The returned callable accepts the same signature as ``build_dim_embedding``
    but ignores all model-loading arguments and returns a deterministic random
    unit vector seeded by the sorted query strings.

    Args:
        dim: Embedding dimension.

    Returns:
        Callable compatible with ``core.build_dim_embedding``.
    """

    def _stub(queries: list[str], **kwargs: Any) -> np.ndarray:
        seed = hash(tuple(sorted(queries))) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    return _stub


@contextmanager
def _no_model_load(dim: int = 8) -> Generator[None, None, None]:
    """Context manager that patches both ``load_sbert`` and ``build_dim_embedding``.

    This prevents any Hugging Face model loading and any network access.

    Args:
        dim: Embedding dimension to pass to the dim stub.

    Yields:
        Nothing; the patches are active for the duration of the ``with`` block.
    """
    fake_bundle = MagicMock()
    with patch("spar_measure.core.load_sbert", return_value=fake_bundle):
        with patch("spar_measure.core.build_dim_embedding", _dim_stub(dim)):
            yield


# ---------------------------------------------------------------------------
# Tests: empty / missing scale specs
# ---------------------------------------------------------------------------


def test_empty_scales_dict_raises() -> None:
    """An empty scales dict is rejected before any embedding step.

    The ``measure()`` function should raise ``ValueError`` with a message
    mentioning 'scales'. The error fires before ``load_sbert`` is reached
    because the scales dict is checked first — but we patch anyway for safety.
    """
    docs = _make_docs()
    with _no_model_load():
        with pytest.raises(ValueError, match="[Ss]cale"):
            measure(
                docs,
                {},
                id_col="doc_id",
                precomputed_embeddings=_make_precomputed(),
            )


def test_scales_with_only_dimensions_key_raises() -> None:
    """A dict with only a ``'dimensions'`` key and no ``'scales'`` key is rejected.

    When both ``'dimensions'`` and ``'scales'`` are present, ``measure()``
    treats them as the explicit form. A dict with only ``'dimensions'`` falls
    into the inline form, which iterates over the top-level keys: the single
    key ``'dimensions'`` is treated as a scale name; its value lacks
    ``pos_dims``/``neg_dims``, so ``combine_scale`` is called with two empty
    lists and raises.
    """
    docs = _make_docs()
    bad_scales: dict[str, Any] = {
        "dimensions": {
            "Create": {"queries": ["We should innovate."]},
        }
    }
    # Fails with ValueError from combine_scale (empty pos and neg).
    with _no_model_load():
        with pytest.raises((ValueError, KeyError)):
            measure(
                docs,
                bad_scales,
                id_col="doc_id",
                precomputed_embeddings=_make_precomputed(),
            )


def test_scale_referencing_unknown_dimension_raises() -> None:
    """A scale that references an undefined dimension raises an error.

    The error surfaces as a ``KeyError`` (from the dim_embs dict lookup)
    when the explicit form is used and a scale lists a dimension name that
    has no corresponding entry in ``dim_specs``.
    """
    docs = _make_docs()
    scales: dict[str, Any] = {
        "dimensions": {
            "Create": {"queries": ["We should innovate."]},
        },
        "scales": {
            "MyScale": {"pos_dims": ["Create", "Ghost"], "neg_dims": []},
        },
    }
    # KeyError surfaces from dim_embs["Ghost"] lookup after "Create" is embedded.
    with _no_model_load():
        with pytest.raises((ValueError, KeyError)):
            measure(
                docs,
                scales,
                id_col="doc_id",
                precomputed_embeddings=_make_precomputed(),
            )


# ---------------------------------------------------------------------------
# Tests: valid edge cases
# ---------------------------------------------------------------------------


def test_single_scale_pos_only_works() -> None:
    """One scale with one ``pos_dim`` and no ``neg_dims`` runs correctly.

    The output should have shape ``(n_docs, 2)`` (id column + 1 scale column).
    """
    docs = _make_docs()
    precomputed = _make_precomputed()
    scales: dict[str, Any] = {
        "dimensions": {
            "Creative": {"queries": ["We should adapt and innovate."]},
        },
        "scales": {
            "Creativity": {"pos_dims": ["Creative"], "neg_dims": []},
        },
    }
    with _no_model_load():
        out = measure(
            docs,
            scales,
            id_col="doc_id",
            precomputed_embeddings=precomputed,
        )
    assert out.shape == (len(docs), 2)
    assert "Creativity" in out.columns
    assert np.isfinite(out["Creativity"].to_numpy()).all()


def test_single_scale_neg_only_works() -> None:
    """One scale with only ``neg_dims`` (no ``pos_dims``) runs correctly.

    ``combine_scale([], [neg])`` returns the negated mean, which is valid.
    """
    docs = _make_docs()
    precomputed = _make_precomputed()
    scales: dict[str, Any] = {
        "dimensions": {
            "Danger": {"queries": ["This is dangerous and risky."]},
        },
        "scales": {
            "Safety": {"pos_dims": [], "neg_dims": ["Danger"]},
        },
    }
    with _no_model_load():
        out = measure(
            docs,
            scales,
            id_col="doc_id",
            precomputed_embeddings=precomputed,
        )
    assert "Safety" in out.columns
    assert out.shape[0] == len(docs)


def test_inline_scale_dict_limitation() -> None:
    """The inline form's mixed-dict limitation: dim-only sibling keys raise.

    In ``core.measure``'s inline branch, the entire top-level dict is treated
    as ``scale_specs``. A dim-definition entry (with ``'queries'`` but no
    ``'pos_dims'``/``'neg_dims'``) is therefore treated as a scale entry with
    empty pos and neg lists, causing ``combine_scale([], [])`` to raise
    ``ValueError``. This is a known limitation of the inline form.

    Users should use the explicit two-level form (``{"dimensions": ...,
    "scales": ...}``) to avoid this issue.
    """
    docs = _make_docs()
    scales_mixed: dict[str, Any] = {
        # Dim definition — will be treated as a scale entry with empty pos/neg.
        "Creative": {"queries": ["We should adapt and innovate."]},
        # Actual scale — references 'Creative' as a pos dim.
        "Creativity": {"pos_dims": ["Creative"], "neg_dims": []},
    }
    with _no_model_load():
        with pytest.raises(ValueError):
            # "Creative" entry → combine_scale([], []) → ValueError.
            measure(
                docs,
                scales_mixed,
                id_col="doc_id",
                precomputed_embeddings=_make_precomputed(),
            )


def test_explicit_two_level_form_works() -> None:
    """The explicit two-level form (``dimensions`` + ``scales`` keys) runs correctly.

    This is the canonical API for multi-scale measurement. Dimension definitions
    are nested under ``'dimensions'`` and scales are nested under ``'scales'``.
    """
    docs = _make_docs()
    precomputed = _make_precomputed()
    scales: dict[str, Any] = {
        "dimensions": {
            "Creative": {"queries": ["We should adapt and innovate."]},
            "Safe": {"queries": ["Safety is our top priority."]},
        },
        "scales": {
            "Creativity": {"pos_dims": ["Creative"], "neg_dims": ["Safe"]},
        },
    }
    with _no_model_load():
        out = measure(
            docs,
            scales,
            id_col="doc_id",
            precomputed_embeddings=precomputed,
        )
    assert "Creativity" in out.columns
    assert out.shape[0] == len(docs)


def test_inline_scale_missing_dim_raises() -> None:
    """Inline form referencing an undefined dimension raises ``ValueError``.

    The error message should mention the unknown dimension name.
    """
    docs = _make_docs()
    scales: dict[str, Any] = {
        "Creative": {"queries": ["We should adapt."]},
        # 'Creativity' scale references 'Ghost' which is not defined anywhere.
        "Creativity": {"pos_dims": ["Creative", "Ghost"], "neg_dims": []},
    }
    with _no_model_load():
        with pytest.raises(ValueError, match="Ghost"):
            measure(
                docs,
                scales,
                id_col="doc_id",
                precomputed_embeddings=_make_precomputed(),
            )


def test_dimension_with_no_queries_raises() -> None:
    """A dimension with empty ``queries`` and no ``exemplar_texts`` raises ``ValueError``.

    ``core.measure`` checks for empty queries before calling ``build_dim_embedding``
    and raises ``ValueError`` with a message mentioning 'queries or exemplar_texts'.
    We patch ``load_sbert`` only (to avoid the segfault), but NOT ``build_dim_embedding``,
    so the real validation logic in ``measure()`` is exercised.
    """
    docs = _make_docs()
    scales: dict[str, Any] = {
        "dimensions": {
            "Empty": {"queries": []},
        },
        "scales": {
            "EmptyScale": {"pos_dims": ["Empty"], "neg_dims": []},
        },
    }
    fake_bundle = MagicMock()
    with patch("spar_measure.core.load_sbert", return_value=fake_bundle):
        with pytest.raises(ValueError, match="queries or exemplar_texts"):
            measure(
                docs,
                scales,
                id_col="doc_id",
                precomputed_embeddings=_make_precomputed(),
            )


def test_id_col_none_auto_assigns_doc_id() -> None:
    """When ``id_col=None``, a 0-indexed ``doc_id`` column is inserted.

    This exercises the ``id_col is None`` branch of ``core.measure``.
    """
    n = 4
    docs = pd.DataFrame({"text": [f"doc {i}" for i in range(n)]})
    precomputed = _make_precomputed(n=n)
    scales: dict[str, Any] = {
        "dimensions": {
            "A": {"queries": ["sentence A"]},
        },
        "scales": {
            "ScaleA": {"pos_dims": ["A"], "neg_dims": []},
        },
    }
    with _no_model_load():
        out = measure(docs, scales, id_col=None, precomputed_embeddings=precomputed)
    assert "doc_id" in out.columns
    assert out["doc_id"].tolist() == list(range(n))
