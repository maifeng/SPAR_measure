# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure business logic for SPAR: embedding, semantic search, projection, scoring.

None of this module imports Gradio. It is safe to call from Jupyter, Colab,
a pytest suite, or a batch script. The Gradio callbacks in :mod:`ui` are thin
adapters that forward to the functions here.

Key entry points:

- :func:`embed_texts`: embed a list of strings with Sentence Transformers or OpenAI.
- :func:`build_scale_embeddings`: combine dimension embeddings into scale vectors.
- :func:`project_documents`: score documents against scale vectors (dot product
  or single-subspace projection).
- :func:`measure`: end-to-end headless pipeline used by :mod:`api`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from transformers import AutoModel, AutoTokenizer

from . import util_funcs
from .state import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_SBERT_MODEL,
)
from .zca import ZCA

logger = logging.getLogger(__name__)

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
"""Torch device chosen at import time."""


# --- Sentence Transformers -------------------------------------------------


@dataclass
class SbertBundle:
    """Loaded Sentence-BERT tokenizer and model.

    Attributes:
        tokenizer: Hugging Face ``AutoTokenizer`` instance.
        model: Hugging Face ``AutoModel`` instance, in eval mode, on ``DEVICE``.
        name: The model name (e.g., ``"all-MiniLM-L6-v2"``).
    """

    tokenizer: Any
    model: Any
    name: str


def load_sbert(model_name: str = DEFAULT_SBERT_MODEL) -> SbertBundle:
    """Load a Sentence-BERT tokenizer/model pair from Hugging Face.

    Args:
        model_name: A model name under the ``sentence-transformers`` namespace,
            e.g. ``"all-MiniLM-L6-v2"`` or ``"all-mpnet-base-v2"``.

    Returns:
        Populated :class:`SbertBundle`.
    """
    logger.info("Loading Sentence Transformers model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")
    model = AutoModel.from_pretrained(f"sentence-transformers/{model_name}")
    model.eval()
    model.to(DEVICE)
    return SbertBundle(tokenizer=tokenizer, model=model, name=model_name)


def _iter_batches(seq: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    """Yield fixed-size slices of ``seq``.

    Args:
        seq: Sequence of strings.
        batch_size: Slice size. Must be >= 1.

    Yields:
        Successive batches of length ``batch_size`` (final batch may be shorter).
    """
    for i in range(0, len(seq), batch_size):
        yield list(seq[i : i + batch_size])


def embed_with_sbert(
    docs: Sequence[str],
    bundle: SbertBundle,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    progress_iter: Any = None,
) -> np.ndarray:
    """Mean-pool and L2-normalize Sentence-BERT embeddings for ``docs``.

    Args:
        docs: List of strings to embed.
        bundle: Loaded :class:`SbertBundle`.
        batch_size: Number of docs per forward pass.
        progress_iter: Optional callable wrapping a range iterator for progress
            reporting, e.g. ``tqdm`` or ``gr.Progress().tqdm``. If ``None``, no
            progress is shown.

    Returns:
        ``(len(docs), hidden_dim)`` float32 numpy array.

    Raises:
        AssertionError: If the output row count disagrees with ``len(docs)``.
    """
    docs = [str(x) for x in docs]
    sentence_embeddings: list[torch.Tensor] = []
    rng = range(0, len(docs), batch_size)
    iterator = progress_iter(rng) if progress_iter is not None else rng
    with torch.no_grad():
        bundle.model.eval()
        for i in iterator:
            batch = docs[i : i + batch_size]
            encoded = bundle.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(DEVICE)
            out = bundle.model(**encoded)
            pooled = util_funcs.mean_pooling(out, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            sentence_embeddings.append(pooled)
    arr = torch.cat(sentence_embeddings, dim=0).cpu().numpy()
    assert arr.shape[0] == len(docs), (
        f"Sentence Transformers row count mismatch: {arr.shape[0]} != {len(docs)}"
    )
    return arr.astype(np.float32)


# --- OpenAI Embeddings -----------------------------------------------------


def embed_with_openai(
    docs: Sequence[str],
    api_key: str,
    model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    progress_iter: Any = None,
) -> np.ndarray:
    """Embed ``docs`` with the OpenAI Embeddings API.

    Transient failures (rate limit, connection, timeout, 5xx) of
    :func:`openai.embeddings.create` are retried up to 6 times with exponential
    backoff (2-60s). Non-transient errors such as ``AuthenticationError`` (bad
    API key) and ``BadRequestError`` (malformed input) are re-raised
    immediately so the GUI shows the failure instead of hanging on retries.

    Args:
        docs: List of strings to embed.
        api_key: OpenAI API key.
        model: Embedding model name. Defaults to ``text-embedding-3-small``.
        batch_size: Number of docs per API call.
        progress_iter: Optional callable wrapping a list iterator for progress
            reporting.

    Returns:
        ``(len(docs), dim)`` float32 numpy array, L2-normalized.

    Raises:
        openai.OpenAIError: If the API fails after retries.
        AssertionError: If row count disagrees with ``len(docs)``.
    """
    client = OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
        ),
        after=util_funcs.print_error,
        reraise=True,
    )
    def _call(batch: list[str]) -> Any:
        return client.embeddings.create(input=batch, model=model)

    batches = list(_iter_batches(docs, batch_size))
    iterator = progress_iter(batches) if progress_iter is not None else batches
    out: list[list[float]] = []
    for batch in iterator:
        resp = _call(batch)
        out.extend(item.embedding for item in resp.data)
    arr = np.asarray(out, dtype=np.float32)
    assert arr.shape[0] == len(docs), (
        f"OpenAI embedding row count mismatch: expected {len(docs)}, got {arr.shape[0]}"
    )
    # L2 normalize so dot product equals cosine similarity downstream.
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


# --- Unified embed_texts ---------------------------------------------------


def embed_texts(
    docs: Sequence[str],
    *,
    use_openai: bool,
    sbert: SbertBundle | None = None,
    openai_api_key: str | None = None,
    openai_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    progress_iter: Any = None,
) -> np.ndarray:
    """Embed ``docs`` with either Sentence Transformers or OpenAI.

    Args:
        docs: List of strings to embed.
        use_openai: If ``True``, use the OpenAI API; otherwise use Sentence-BERT.
        sbert: Loaded :class:`SbertBundle`. Required when ``use_openai`` is ``False``.
        openai_api_key: OpenAI API key. Required when ``use_openai`` is ``True``.
        openai_model: OpenAI embedding model name.
        batch_size: Batch size for both backends.
        progress_iter: Optional progress wrapper. See :func:`embed_with_sbert`.

    Returns:
        ``(len(docs), dim)`` float32 numpy array.

    Raises:
        ValueError: If required arguments for the chosen backend are missing.
    """
    if use_openai:
        if not openai_api_key:
            raise ValueError("openai_api_key is required when use_openai=True")
        return embed_with_openai(
            docs,
            api_key=openai_api_key,
            model=openai_model,
            batch_size=batch_size,
            progress_iter=progress_iter,
        )
    if sbert is None:
        raise ValueError("sbert bundle is required when use_openai=False")
    return embed_with_sbert(
        docs, bundle=sbert, batch_size=batch_size, progress_iter=progress_iter
    )


# --- Scale assembly --------------------------------------------------------


def build_dim_embedding(
    queries: Sequence[str],
    *,
    use_openai: bool,
    sbert: SbertBundle | None = None,
    openai_api_key: str | None = None,
    openai_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Embed a list of seed queries and return their mean vector.

    Args:
        queries: List of seed/exemplar sentences for one dimension.
        use_openai: See :func:`embed_texts`.
        sbert: See :func:`embed_texts`.
        openai_api_key: See :func:`embed_texts`.
        openai_model: See :func:`embed_texts`.
        batch_size: Batch size.

    Returns:
        ``(dim,)`` mean embedding vector.

    Raises:
        ValueError: If ``queries`` is empty.
    """
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        raise ValueError("At least one non-empty query is required to define a dimension")
    mat = embed_texts(
        queries,
        use_openai=use_openai,
        sbert=sbert,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        batch_size=batch_size,
    )
    return mat.mean(axis=0)


def combine_scale(
    pos_dim_embeddings: Sequence[np.ndarray],
    neg_dim_embeddings: Sequence[np.ndarray],
) -> np.ndarray:
    """Combine positive and negative dimension embeddings into one scale vector.

    The scale is ``mean(pos_dims) - mean(neg_dims)``. If only positives are
    given the scale is ``mean(pos_dims)``; if only negatives, the scale is
    ``-mean(neg_dims)``.

    Args:
        pos_dim_embeddings: List of ``(dim,)`` vectors from the positive dimensions.
        neg_dim_embeddings: List of ``(dim,)`` vectors from the negative dimensions.

    Returns:
        ``(dim,)`` scale embedding.

    Raises:
        ValueError: If both lists are empty.
    """
    has_pos = len(pos_dim_embeddings) > 0
    has_neg = len(neg_dim_embeddings) > 0
    if has_pos and has_neg:
        return np.stack(pos_dim_embeddings).mean(axis=0) - np.stack(neg_dim_embeddings).mean(axis=0)
    if has_pos:
        return np.stack(pos_dim_embeddings).mean(axis=0)
    if has_neg:
        return -np.stack(neg_dim_embeddings).mean(axis=0)
    raise ValueError("At least one positive or negative dimension is required for a scale")


# --- Projection / scoring --------------------------------------------------


def project_documents(
    doc_embeddings: np.ndarray,
    scale_embeddings: dict[str, np.ndarray],
    *,
    single_subspace: bool = False,
    whiten: bool = False,
) -> pd.DataFrame:
    """Score documents against scales via dot product or single-subspace projection.

    With ``single_subspace=False`` (default), each scale is an independent
    direction and the score is a dot product. With ``single_subspace=True``,
    the scales are treated as a joint basis: the returned scores are the
    subspace projection coefficients ``(S Sᵀ)⁺ S X`` so that each score is
    orthogonalized against the others.

    Fixes M-05: the previous implementation used ``np.linalg.inv(S Sᵀ)`` which
    is numerically unstable when the scales are near-collinear. We use
    :func:`numpy.linalg.pinv` (SVD-based pseudoinverse) instead.

    Args:
        doc_embeddings: ``(n_docs, dim)`` L2-normalized document embeddings.
        scale_embeddings: Ordered map from scale name to ``(dim,)`` vector.
        single_subspace: If ``True``, use the joint-subspace projection.
        whiten: If ``True``, ZCA-whiten the score matrix (decorrelates columns).

    Returns:
        ``(n_docs, n_scales)`` DataFrame with scale names as columns.
    """
    if not scale_embeddings:
        raise ValueError("scale_embeddings is empty")

    names = list(scale_embeddings.keys())
    # Stack scales as rows, L2-normalize each (so dot product == cosine).
    S = np.stack([scale_embeddings[n] for n in names], axis=0).astype(np.float64)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)

    X = np.asarray(doc_embeddings, dtype=np.float64)
    if single_subspace:
        # Coefficients of X in the basis S: (S Sᵀ)⁺ S Xᵀ, shape (k, n_docs).
        coefs = np.linalg.pinv(S @ S.T) @ S @ X.T
        scores = coefs.T  # (n_docs, k)
    else:
        # Independent dot product per scale.
        scores = X @ S.T

    if whiten:
        trf = ZCA().fit(scores)
        scores = trf.transform(scores)

    return pd.DataFrame(scores, columns=names)


# --- Headless end-to-end pipeline ------------------------------------------


def measure(
    docs: pd.DataFrame,
    scales: dict[str, dict[str, Any]],
    *,
    text_col: str = "text",
    id_col: str | None = None,
    use_openai: bool = False,
    sbert_model: str = DEFAULT_SBERT_MODEL,
    openai_api_key: str | None = None,
    openai_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    whiten: bool = False,
    single_subspace: bool = False,
    precomputed_embeddings: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run the full SPAR pipeline end-to-end without Gradio.

    The ``scales`` mapping may take two shapes:

    1. **Inline dimensions + pos/neg assembly** (recommended for new users)::

        {
            "External-Internal": {
                "pos_dims": ["Create", "Compete"],
                "neg_dims": ["Control", "Collaborate"],
            },
            ...
        }
        # plus a top-level "dimensions" entry (see below)

    2. **Named dimensions first, then scales reference them**::

        {
            "dimensions": {
                "Create":      {"queries": ["We should adapt and innovate."]},
                "Collaborate": {"queries": ["We should empathize and collaborate."]},
                ...
            },
            "scales": {
                "External-Internal": {
                    "pos_dims": ["Create", "Compete"],
                    "neg_dims": ["Control", "Collaborate"],
                },
                ...
            },
        }

    Form 2 is required whenever dimensions are referenced by more than one
    scale. See :func:`api.score` for a thinner wrapper.

    Args:
        docs: Corpus DataFrame; must contain ``text_col`` (and ``id_col`` if provided).
        scales: Scale specification (see above).
        text_col: Name of the text column in ``docs``.
        id_col: Name of the ID column in ``docs``. If ``None``, the DataFrame
            index is used.
        use_openai: If ``True``, use OpenAI embeddings; otherwise Sentence-BERT.
        sbert_model: Sentence-BERT model name.
        openai_api_key: API key (required iff ``use_openai``).
        openai_model: OpenAI embedding model name.
        batch_size: Embedding batch size.
        whiten: If ``True``, ZCA-whiten the score matrix.
        single_subspace: If ``True``, use joint-subspace projection.
        precomputed_embeddings: Optional ``(n_docs, dim)`` array; skips embedding
            of the corpus if supplied.

    Returns:
        DataFrame with ``id_col`` (or index) and one column per scale.

    Raises:
        ValueError: If ``scales`` is empty or malformed.
    """
    if "dimensions" in scales and "scales" in scales:
        dim_specs: dict[str, dict[str, Any]] = scales["dimensions"]
        scale_specs: dict[str, dict[str, Any]] = scales["scales"]
    else:
        # Inline form: collect all unique dimension names referenced by scales.
        scale_specs = scales
        dim_specs = {}
        for spec in scale_specs.values():
            for key in ("pos_dims", "neg_dims"):
                for dim_name in spec.get(key, []):
                    if dim_name not in dim_specs:
                        # Try to find it as a sibling key in ``scales``.
                        if dim_name in scales and isinstance(scales[dim_name], dict) and "queries" in scales[dim_name]:
                            dim_specs[dim_name] = scales[dim_name]
                        else:
                            raise ValueError(
                                f"Dimension {dim_name!r} referenced by a scale but not defined. "
                                "Provide a top-level 'dimensions' entry or an inline sibling key."
                            )

    if not scale_specs:
        raise ValueError("No scales provided")

    # Load embedding backend once.
    sbert: SbertBundle | None = None
    if not use_openai:
        sbert = load_sbert(sbert_model)
    elif not openai_api_key:
        raise ValueError("openai_api_key is required when use_openai=True")

    # Embed corpus (or accept precomputed).
    if precomputed_embeddings is not None:
        if precomputed_embeddings.shape[0] != len(docs):
            raise ValueError(
                f"precomputed_embeddings has {precomputed_embeddings.shape[0]} rows "
                f"but docs has {len(docs)}"
            )
        doc_emb = precomputed_embeddings.astype(np.float32)
    else:
        if text_col not in docs.columns:
            raise ValueError(f"text_col {text_col!r} not in docs.columns")
        doc_emb = embed_texts(
            docs[text_col].tolist(),
            use_openai=use_openai,
            sbert=sbert,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            batch_size=batch_size,
        )

    # Embed dimensions.
    dim_embs: dict[str, np.ndarray] = {}
    for dim_name, spec in dim_specs.items():
        queries: list[str] = list(spec.get("queries", []))
        exemplars: list[str] = list(spec.get("exemplar_texts", []))
        all_queries = queries + exemplars
        if not all_queries:
            raise ValueError(f"Dimension {dim_name!r} has no queries or exemplar_texts")
        dim_embs[dim_name] = build_dim_embedding(
            all_queries,
            use_openai=use_openai,
            sbert=sbert,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            batch_size=batch_size,
        )

    # Combine into scales.
    scale_embs: dict[str, np.ndarray] = {}
    for scale_name, spec in scale_specs.items():
        pos = [dim_embs[d] for d in spec.get("pos_dims", [])]
        neg = [dim_embs[d] for d in spec.get("neg_dims", [])]
        scale_embs[scale_name] = combine_scale(pos, neg)

    # Project.
    scores = project_documents(
        doc_emb,
        scale_embs,
        single_subspace=single_subspace,
        whiten=whiten,
    )

    # Attach id column.
    if id_col is not None:
        if id_col not in docs.columns:
            raise ValueError(f"id_col {id_col!r} not in docs.columns")
        scores.insert(0, id_col, docs[id_col].to_numpy())
    else:
        scores.insert(0, "doc_id", np.arange(len(docs)))

    return scores


__all__ = [
    "DEVICE",
    "SbertBundle",
    "load_sbert",
    "embed_with_sbert",
    "embed_with_openai",
    "embed_texts",
    "build_dim_embedding",
    "combine_scale",
    "project_documents",
    "measure",
]
