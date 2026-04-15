# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :class:`spar_measure.vector_store.ChromaStore`.

Skipped automatically when ``chromadb`` is not installed.
All tests use precomputed embeddings (no model loading).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed; skip vector_store tests")

from spar_measure.vector_store import ChromaStore  # noqa: E402  (after importorskip)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


DIM = 16
N = 5


def _unit_embeddings(n: int = N, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Return ``(n, dim)`` L2-normalized float32 embeddings.

    Args:
        n: Number of documents.
        dim: Embedding dimension.
        seed: Random seed.

    Returns:
        ``(n, dim)`` float32 numpy array, each row unit-norm.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


@pytest.fixture()
def ephemeral_store() -> ChromaStore:
    """Create an in-memory ChromaStore with 5 precomputed documents.

    Returns:
        Populated :class:`ChromaStore` (ephemeral, no disk I/O).
    """
    store = ChromaStore(collection_name="test_col", persist_dir=None)
    emb = _unit_embeddings()
    ids = [f"doc_{i}" for i in range(N)]
    docs = [f"Document text {i}" for i in range(N)]
    store.add_precomputed(ids=ids, embeddings=emb, documents=docs)
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chromastore_count_after_add(ephemeral_store: ChromaStore) -> None:
    """``count()`` returns the number of added documents."""
    assert ephemeral_store.count() == N


def test_chromastore_get_all_embeddings_shape(ephemeral_store: ChromaStore) -> None:
    """``get_all_embeddings()`` returns ``(n_docs, dim)`` float32 array."""
    emb = ephemeral_store.get_all_embeddings()
    assert emb.shape == (N, DIM)
    assert emb.dtype == np.float32


def test_chromastore_get_all_ids(ephemeral_store: ChromaStore) -> None:
    """``get_all_ids()`` returns a list of length ``n_docs``."""
    ids = ephemeral_store.get_all_ids()
    assert len(ids) == N
    # All IDs should be the ones we inserted.
    assert set(ids) == {f"doc_{i}" for i in range(N)}


def test_chromastore_query_by_vector_shape(ephemeral_store: ChromaStore) -> None:
    """``query_by_vector`` returns the requested number of results."""
    q = _unit_embeddings(n=1, seed=99)[0]
    ids, embs = ephemeral_store.query_by_vector(q, n_results=3)
    assert len(ids) == 3
    assert embs.shape == (3, DIM)
    assert embs.dtype == np.float32


def test_chromastore_query_by_vector_ids_are_strings(ephemeral_store: ChromaStore) -> None:
    """All returned doc IDs are strings."""
    q = _unit_embeddings(n=1, seed=11)[0]
    ids, _ = ephemeral_store.query_by_vector(q, n_results=2)
    for doc_id in ids:
        assert isinstance(doc_id, str)


def test_chromastore_get_all_embeddings_roundtrip(ephemeral_store: ChromaStore) -> None:
    """Embeddings retrieved by ``get_all_embeddings()`` are approximately unit-norm.

    ChromaDB may apply cosine normalization internally, so we check that the
    stored and retrieved vectors are close to unit-norm (within float32 tolerance).
    """
    emb = ephemeral_store.get_all_embeddings()
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, np.ones(N), atol=1e-5)


def test_chromastore_empty_raises_on_get_all(tmp_path: Path) -> None:
    """``get_all_embeddings()`` raises ``ValueError`` on an empty store."""
    store = ChromaStore(collection_name="empty_col", persist_dir=None)
    with pytest.raises(ValueError, match="empty"):
        store.get_all_embeddings()


def test_chromastore_add_precomputed_id_mismatch_raises() -> None:
    """``add_precomputed`` raises ``ValueError`` when ids and embeddings lengths differ."""
    store = ChromaStore(collection_name="mismatch_col", persist_dir=None)
    emb = _unit_embeddings(n=3)
    with pytest.raises(ValueError, match="length"):
        store.add_precomputed(ids=["a", "b"], embeddings=emb)


def test_chromastore_persistent_roundtrip(tmp_path: Path) -> None:
    """Writing to ``tmp_path``, then reloading, preserves document count."""
    persist_dir = str(tmp_path / "chroma_db")
    # Create and populate.
    store = ChromaStore(collection_name="persist_col", persist_dir=persist_dir)
    emb = _unit_embeddings()
    ids = [f"doc_{i}" for i in range(N)]
    store.add_precomputed(ids=ids, embeddings=emb)
    assert store.count() == N

    # Reload from disk.
    store2 = ChromaStore.load(persist_dir, "persist_col")
    assert store2.count() == N

    # Embeddings are preserved.
    emb2 = store2.get_all_embeddings()
    assert emb2.shape == (N, DIM)


def test_chromastore_load_missing_dir_raises(tmp_path: Path) -> None:
    """``ChromaStore.load`` raises ``FileNotFoundError`` for a non-existent path."""
    missing = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        ChromaStore.load(missing, "any_col")


def test_chromastore_query_n_results_capped_at_count(ephemeral_store: ChromaStore) -> None:
    """Requesting more results than the collection size returns at most ``count()`` items."""
    q = _unit_embeddings(n=1, seed=77)[0]
    ids, embs = ephemeral_store.query_by_vector(q, n_results=100)
    # Collection has N=5 documents.
    assert len(ids) <= N
    assert embs.shape[0] <= N
