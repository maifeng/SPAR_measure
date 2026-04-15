# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional ChromaDB-backed vector store for large SPAR corpora.

Provides :class:`ChromaStore`, a thin wrapper around ChromaDB that persists
document embeddings to disk so they can be reloaded across runs without
re-embedding. Designed for corpora of 50k+ documents.

Typical workflow::

    from spar_measure.vector_store import ChromaStore
    from spar_measure import score

    # One-time: embed and persist.
    store = ChromaStore("my_corpus", persist_dir="/data/chroma")
    store.embed_and_store(docs_df, text_col="post", model_name="all-MiniLM-L6-v2")

    # Subsequent runs: load and score.
    store = ChromaStore.load("/data/chroma", "my_corpus")
    emb = store.get_all_embeddings()          # (n_docs, dim) float32
    ids = store.get_all_ids()                 # list[str]
    out = score(docs_df, scales, precomputed_embeddings=emb)

    # Active retrieval: find top-k similar docs.
    doc_ids, embs = store.query_by_vector(query_vec, n_results=100)

This module is optional: if ``chromadb`` is not installed, importing from it
raises ``ImportError`` with install instructions. Install with::

    pip install spar_measure[vector]

ChromaDB version requirement: >=0.5 (tested against 1.5.7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import chromadb  # type: ignore[import-untyped]

    HAS_CHROMADB: bool = True
except ImportError:
    HAS_CHROMADB = False

if TYPE_CHECKING:
    import pandas as pd


def _require_chromadb() -> None:
    """Raise ``ImportError`` if ``chromadb`` is not installed.

    Raises:
        ImportError: If ``chromadb`` is not importable.
    """
    if not HAS_CHROMADB:
        raise ImportError(
            "chromadb is required for ChromaStore. "
            "Install it with: pip install spar_measure[vector]"
        )


@dataclass
class ChromaStore:
    """Persistent vector store backed by ChromaDB for large SPAR corpora.

    Each instance wraps one ChromaDB collection. Embeddings are stored with
    document text and doc_id metadata so the store is self-contained.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        persist_dir: Local directory for persistence. If ``None``, an in-memory
            ``EphemeralClient`` is used (suitable for tests).
    """

    collection_name: str
    persist_dir: str | None = None
    _client: Any = field(default=None, init=False, repr=False)
    _collection: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the ChromaDB client and open (or create) the collection."""
        _require_chromadb()
        if self.persist_dir is not None:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            # No built-in embedding function: we always supply vectors ourselves.
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaStore: collection %r opened, %d documents.",
            self.collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Adding documents
    # ------------------------------------------------------------------

    def embed_and_store(
        self,
        docs_df: pd.DataFrame,
        text_col: str = "text",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        id_col: str | None = None,
    ) -> None:
        """Embed all documents and store them in the collection.

        Documents already present (same ID) are skipped via upsert semantics.

        Args:
            docs_df: Corpus DataFrame. Must contain ``text_col``.
            text_col: Name of the text column.
            model_name: Sentence-BERT model name from the
                ``sentence-transformers`` namespace.
            batch_size: Embedding batch size.
            id_col: Column to use as doc_id. If ``None``, uses the DataFrame
                index cast to string.

        Raises:
            ValueError: If ``text_col`` is not in ``docs_df.columns``.
            ImportError: If ``chromadb`` or ``transformers``/``torch`` are
                not installed.
        """
        if text_col not in docs_df.columns:
            raise ValueError(f"text_col {text_col!r} not in docs_df.columns")

        from .core import SbertBundle, embed_with_sbert, load_sbert

        bundle: SbertBundle = load_sbert(model_name)
        texts = docs_df[text_col].astype(str).tolist()

        if id_col is not None:
            if id_col not in docs_df.columns:
                raise ValueError(f"id_col {id_col!r} not in docs_df.columns")
            ids = [str(v) for v in docs_df[id_col].tolist()]
        else:
            ids = [str(i) for i in docs_df.index.tolist()]

        logger.info("ChromaStore.embed_and_store: embedding %d documents.", len(texts))
        embeddings = embed_with_sbert(texts, bundle, batch_size=batch_size)
        self.add_precomputed(ids=ids, embeddings=embeddings, documents=texts)

    def add_precomputed(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store precomputed embeddings without model loading.

        This is the primary method for tests and for users who have already
        embedded their corpus.

        Args:
            ids: List of unique string IDs for each document. Must have the
                same length as ``embeddings``.
            embeddings: ``(n_docs, dim)`` float32 numpy array.
            documents: Optional list of original text strings (stored as
                metadata in ChromaDB). If ``None``, empty strings are stored.
            metadatas: Optional list of metadata dicts, one per document. If
                ``None``, no extra metadata is stored.

        Raises:
            ValueError: If ``len(ids) != len(embeddings)``.
        """
        if len(ids) != len(embeddings):
            raise ValueError(
                f"ids length {len(ids)} != embeddings length {len(embeddings)}"
            )
        docs = documents if documents is not None else [""] * len(ids)

        # ChromaDB 0.6+ rejects empty dicts in metadatas.
        # Only pass the metadatas keyword when the caller provides non-empty dicts.
        upsert_kwargs: dict[str, Any] = {
            "ids": ids,
            "embeddings": embeddings.tolist(),
            "documents": docs,
        }
        if metadatas is not None:
            upsert_kwargs["metadatas"] = metadatas

        # ChromaDB upsert: overwrites existing docs with the same id.
        self._collection.upsert(**upsert_kwargs)
        logger.info("ChromaStore: upserted %d documents.", len(ids))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_by_vector(
        self,
        query_vector: np.ndarray,
        n_results: int = 10,
    ) -> tuple[list[str], np.ndarray]:
        """Query the collection with a precomputed embedding vector.

        Args:
            query_vector: ``(dim,)`` float32 query embedding.
            n_results: Number of nearest neighbours to return.

        Returns:
            A tuple ``(doc_ids, embeddings)`` where:
            - ``doc_ids`` is a list of ``n_results`` string IDs.
            - ``embeddings`` is an ``(n_results, dim)`` float32 array of the
              corresponding stored embeddings.
        """
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        results = self._collection.query(
            query_embeddings=q.tolist(),
            n_results=min(n_results, self.count()),
            include=["embeddings", "documents"],
        )
        ids: list[str] = results["ids"][0]
        raw_emb = results["embeddings"][0]
        emb_arr = np.array(raw_emb, dtype=np.float32)
        return ids, emb_arr

    def query(
        self,
        query_texts: list[str],
        n_results: int = 10,
        sbert: Any = None,
    ) -> tuple[list[str], np.ndarray]:
        """Embed query texts with SBERT and return nearest neighbours.

        Args:
            query_texts: List of query strings to embed.
            n_results: Number of nearest neighbours per query. Only the results
                for the first query are returned when multiple queries are given.
            sbert: Loaded :class:`spar_measure.core.SbertBundle`. If ``None``,
                the default SBERT model is loaded on the fly.

        Returns:
            A tuple ``(doc_ids, embeddings)`` for the first query text:
            - ``doc_ids``: list of nearest-neighbour string IDs.
            - ``embeddings``: ``(n_results, dim)`` float32 array.
        """
        from .core import SbertBundle, embed_with_sbert, load_sbert
        from .state import DEFAULT_SBERT_MODEL

        bundle: SbertBundle = sbert if sbert is not None else load_sbert(DEFAULT_SBERT_MODEL)
        q_emb = embed_with_sbert(query_texts, bundle)  # (n_queries, dim)
        return self.query_by_vector(q_emb[0], n_results=n_results)

    # ------------------------------------------------------------------
    # Bulk retrieval
    # ------------------------------------------------------------------

    def get_all_embeddings(self) -> np.ndarray:
        """Return all stored embeddings as a ``(n_docs, dim)`` float32 array.

        The row order matches the insertion order reported by ChromaDB's
        ``get()`` call (by ascending internal ID). This is consistent across
        calls within the same session; persistence across sessions is
        guaranteed only when a ``persist_dir`` is given.

        Returns:
            ``(n_docs, dim)`` float32 numpy array suitable for passing to
            ``score(precomputed_embeddings=...)``.

        Raises:
            ValueError: If the collection is empty.
        """
        if self.count() == 0:
            raise ValueError("ChromaStore is empty; add documents first.")
        result = self._collection.get(include=["embeddings"])
        return np.array(result["embeddings"], dtype=np.float32)

    def get_all_ids(self) -> list[str]:
        """Return all stored document IDs in insertion order.

        Returns:
            List of string IDs.
        """
        result = self._collection.get(include=[])
        return result["ids"]

    def count(self) -> int:
        """Return the number of documents currently stored.

        Returns:
            Non-negative integer count.
        """
        return self._collection.count()

    # ------------------------------------------------------------------
    # Persistence / factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, persist_dir: str | Path, collection_name: str) -> "ChromaStore":
        """Open an existing persistent store without re-creating it.

        Args:
            persist_dir: Directory that was previously passed to
                ``ChromaStore(persist_dir=...)``.
            collection_name: Name of the collection to open.

        Returns:
            A :class:`ChromaStore` connected to the existing collection.

        Raises:
            FileNotFoundError: If ``persist_dir`` does not exist.
            ImportError: If ``chromadb`` is not installed.
        """
        _require_chromadb()
        persist_path = Path(persist_dir)
        if not persist_path.exists():
            raise FileNotFoundError(
                f"persist_dir {persist_path!r} does not exist. "
                "Run ChromaStore(collection_name, persist_dir=...).embed_and_store(...) first."
            )
        return cls(collection_name=collection_name, persist_dir=str(persist_path))


__all__ = ["ChromaStore", "HAS_CHROMADB"]
