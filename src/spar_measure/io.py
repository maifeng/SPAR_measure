# SPDX-License-Identifier: GPL-3.0-or-later
"""File I/O helpers for SPAR: CSV reading with encoding fallback and NPY validation.

Fixes H-02 (CSV encoding was hardcoded to ``ISO-8859-1``) and M-11 (no input
validation on uploaded ``.npy``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_csv_smart(path: str | Path, encoding: str | None = None) -> pd.DataFrame:
    """Read a CSV with a UTF-8 first, latin-1 fallback strategy.

    The previous implementation hardcoded ``ISO-8859-1`` which silently
    mojibakes CJK, emoji, and smart quotes. This helper tries UTF-8 first
    (the dominant corpus encoding in 2026) and falls back to latin-1 on
    ``UnicodeDecodeError``, which is guaranteed to succeed for any byte
    sequence.

    Args:
        path: Path to the CSV file.
        encoding: Explicit encoding override. If ``None``, auto-detect.

    Returns:
        Parsed pandas DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pd.errors.ParserError: If the CSV is malformed.
    """
    path = Path(path)
    if encoding is not None:
        return pd.read_csv(path, encoding=encoding)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 decode failed for %s; falling back to latin-1. "
            "Non-ASCII characters may appear incorrectly. "
            "Consider resaving the source as UTF-8.",
            path,
        )
        return pd.read_csv(path, encoding="latin-1")


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from a ``.npy`` file and validate their shape and dtype.

    Args:
        path: Path to the ``.npy`` file.

    Returns:
        The loaded array, cast to ``float32`` if needed.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the array is not 2D, has NaN/Inf, or has the wrong dtype.
    """
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(
            f"Embedding matrix must be 2D (n_docs, dim); got shape {arr.shape}"
        )
    if arr.dtype not in (np.float32, np.float64):
        raise ValueError(
            f"Embedding matrix dtype must be float32 or float64; got {arr.dtype}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("Embedding matrix contains NaN or Inf values")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


__all__ = ["read_csv_smart", "load_embeddings"]
