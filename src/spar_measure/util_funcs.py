# SPDX-License-Identifier: GPL-3.0-or-later
"""Embedding utility functions: cosine similarity, dot score, semantic search.

Several functions below are copied from the ``sentence-transformers`` project
(https://github.com/UKPLab/sentence-transformers), Apache 2.0. Original
license at https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE.

The unused IO helpers (``print_time_now``, ``time_now``, ``line_counter``,
``file_to_list``, ``list_to_file``, ``normalize_embeddings``,
``pytorch_cos_sim``) that previously lived here were removed during the
0.3.0 refactor (fixes M-05 per digest/05_SPAR_code_review.md).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """Compute cosine similarity ``cos_sim(a[i], b[j])`` for all ``i, j``.

    Args:
        a: ``(n, dim)`` tensor, list, or ``np.ndarray``.
        b: ``(m, dim)`` tensor, list, or ``np.ndarray``.

    Returns:
        ``(n, m)`` tensor of pairwise cosine similarities.
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: Tensor, b: Tensor) -> Tensor:
    """Compute dot product ``a[i] . b[j]`` for all ``i, j``.

    Assumes inputs are already L2-normalized if cosine-similarity semantics
    are desired (saves a normalization pass).

    Args:
        a: ``(n, dim)`` tensor, list, or ``np.ndarray``.
        b: ``(m, dim)`` tensor, list, or ``np.ndarray``.

    Returns:
        ``(n, m)`` tensor of pairwise dot products.
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return torch.mm(a, b.transpose(0, 1))


def semantic_search(
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500_000,
    top_k: int = 10,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> list[list[dict[str, Any]]]:
    """Perform cosine-similarity search between queries and a corpus.

    Chunked to keep memory bounded; works for corpora up to ~1M entries.

    Args:
        query_embeddings: ``(n_queries, dim)`` tensor or ndarray.
        corpus_embeddings: ``(n_corpus, dim)`` tensor or ndarray.
        query_chunk_size: Number of queries processed per step.
        corpus_chunk_size: Number of corpus entries scanned per step.
        top_k: How many top matches to return per query.
        score_function: Pairwise scoring function. Default: :func:`cos_sim`.

    Returns:
        A list with one entry per query; each entry is a list of
        ``{"corpus_id": int, "score": float}`` dicts sorted by score desc.
    """
    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)
    if query_embeddings.dim() == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list: list[list[dict[str, Any]]] = [
        [] for _ in range(len(query_embeddings))
    ]

    for q_start in range(0, len(query_embeddings), query_chunk_size):
        for c_start in range(0, len(corpus_embeddings), corpus_chunk_size):
            scores = score_function(
                query_embeddings[q_start : q_start + query_chunk_size],
                corpus_embeddings[c_start : c_start + corpus_chunk_size],
            )
            top_values, top_idx = torch.topk(
                scores,
                min(top_k, len(scores[0])),
                dim=1,
                largest=True,
                sorted=False,
            )
            top_values = top_values.cpu().tolist()
            top_idx = top_idx.cpu().tolist()
            for q_i in range(len(scores)):
                for sub_corpus_id, score in zip(top_idx[q_i], top_values[q_i]):
                    corpus_id = c_start + sub_corpus_id
                    query_id = q_start + q_i
                    queries_result_list[query_id].append(
                        {"corpus_id": corpus_id, "score": score}
                    )

    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(
            queries_result_list[idx], key=lambda x: x["score"], reverse=True
        )[:top_k]
    return queries_result_list


def mean_pooling(model_output: Any, attention_mask: Tensor) -> Tensor:
    """Mean-pool token embeddings using an attention mask.

    Args:
        model_output: Output of a ``transformers`` model; ``model_output[0]`` is
            the token-level embeddings ``(batch, seq_len, dim)``.
        attention_mask: ``(batch, seq_len)`` attention mask.

    Returns:
        ``(batch, dim)`` mean-pooled embeddings.
    """
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def print_error(retry_state: Any) -> None:
    """Tenacity ``after`` callback: log the exception of a failed retry attempt.

    Args:
        retry_state: A ``tenacity.RetryCallState`` instance.
    """
    if retry_state.outcome and retry_state.outcome.failed:
        logger.warning(
            "Retry %d failed: %s",
            retry_state.attempt_number,
            retry_state.outcome.exception(),
        )


__all__ = [
    "cos_sim",
    "dot_score",
    "semantic_search",
    "mean_pooling",
    "print_error",
]
