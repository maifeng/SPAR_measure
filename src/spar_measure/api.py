# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless Python API for SPAR measurement.

Designed so a Colab notebook (or any batch script) can run SPAR without
touching the Gradio UI or browser tunnel. This is the workshop-critical
deliverable from ``plan/17_spar_refactor.md``.

Example:

    >>> import pandas as pd
    >>> from spar_measure import score
    >>> docs = pd.DataFrame({
    ...     "id": [0, 1, 2],
    ...     "text": [
    ...         "We encourage new ways of thinking.",
    ...         "Safety first in every department.",
    ...         "Our team is working together to weather the storm.",
    ...     ],
    ... })
    >>> scales = {
    ...     "dimensions": {
    ...         "Creative": {"queries": ["We should adapt and innovate."]},
    ...         "Safe":     {"queries": ["Safety is our top priority."]},
    ...         "Danger":   {"queries": ["This is dangerous."]},
    ...     },
    ...     "scales": {
    ...         "Creativity": {"pos_dims": ["Creative"], "neg_dims": []},
    ...         "Safety":     {"pos_dims": ["Safe"],     "neg_dims": ["Danger"]},
    ...     },
    ... }
    >>> out = score(docs, scales, text_col="text", id_col="id")  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import measure
from .state import DEFAULT_EMBED_BATCH_SIZE, DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_SBERT_MODEL


def score(
    docs: pd.DataFrame,
    scales: dict[str, dict[str, Any]],
    *,
    embedding_model: str | None = None,
    openai_api_key: str | None = None,
    openai_embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    whiten: bool = False,
    single_subspace: bool = False,
    text_col: str = "text",
    id_col: str | None = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    precomputed_embeddings: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score documents against SPAR scales with zero UI.

    This is a thin wrapper around :func:`spar_measure.core.measure` with a
    friendlier keyword interface for notebook use. The ``embedding_model``
    argument accepts both Sentence-BERT model names (e.g.
    ``"all-MiniLM-L6-v2"``) and OpenAI model names (e.g.
    ``"text-embedding-3-small"``); the backend is auto-selected based on
    whether ``openai_api_key`` is supplied.

    Args:
        docs: Corpus DataFrame. Must contain ``text_col`` and, if given,
            ``id_col``.
        scales: Dict describing dimensions and scales. Two accepted forms:

            * **Explicit**::

                {
                    "dimensions": {
                        "Creative": {"queries": ["We should adapt and innovate."]},
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

            * **Inline** (dimensions alongside scales at the top level).
              See :func:`spar_measure.core.measure` for details.

        embedding_model: Sentence-BERT or OpenAI model name. If ``None``,
            defaults to ``"all-MiniLM-L6-v2"``. If ``openai_api_key`` is set,
            this is interpreted as the OpenAI embedding model name instead
            (overrides ``openai_embedding_model``).
        openai_api_key: Optional OpenAI API key. Supplying this switches the
            embedding backend to the OpenAI Embedding API.
        openai_embedding_model: Name of the OpenAI model (ignored unless
            ``openai_api_key`` is supplied and ``embedding_model`` is ``None``).
        whiten: If ``True``, apply ZCA whitening to the score matrix to
            decorrelate columns.
        single_subspace: If ``True``, score documents using a joint-subspace
            projection ``(S Sᵀ)⁺ S X`` rather than independent dot products.
        text_col: Name of the text column in ``docs``.
        id_col: Name of the ID column in ``docs``. If ``None``, a 0-indexed
            ``doc_id`` column is added.
        batch_size: Embedding batch size.
        precomputed_embeddings: Optional ``(n_docs, dim)`` array. If supplied,
            the corpus is not re-embedded.

    Returns:
        DataFrame of shape ``(n_docs, 1 + n_scales)`` with the ID column first
        and one column per scale.

    Raises:
        ValueError: If ``scales`` is empty or malformed, if required columns
            are missing, or if backend-specific arguments are missing.

    Example:
        >>> from spar_measure import score
        >>> out = score(docs, scales, text_col="text", id_col="doc_id")  # doctest: +SKIP
    """
    use_openai = openai_api_key is not None
    if embedding_model is None:
        if use_openai:
            model_name = openai_embedding_model
            sbert_name = DEFAULT_SBERT_MODEL  # not used
            openai_name = model_name
        else:
            model_name = DEFAULT_SBERT_MODEL
            sbert_name = model_name
            openai_name = openai_embedding_model
    else:
        if use_openai:
            sbert_name = DEFAULT_SBERT_MODEL
            openai_name = embedding_model
        else:
            sbert_name = embedding_model
            openai_name = openai_embedding_model

    return measure(
        docs,
        scales,
        text_col=text_col,
        id_col=id_col,
        use_openai=use_openai,
        sbert_model=sbert_name,
        openai_api_key=openai_api_key,
        openai_model=openai_name,
        batch_size=batch_size,
        whiten=whiten,
        single_subspace=single_subspace,
        precomputed_embeddings=precomputed_embeddings,
    )


__all__ = ["score", "measure"]
