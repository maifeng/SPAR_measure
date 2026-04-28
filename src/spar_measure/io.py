# SPDX-License-Identifier: GPL-3.0-or-later
"""File I/O helpers for SPAR: CSV reading with encoding fallback and NPY validation.

Fixes H-02 (CSV encoding was hardcoded to ``ISO-8859-1``) and M-11 (no input
validation on uploaded ``.npy``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCALES_JSON_NAME = "scales.json"


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


def gui_state_to_scales_spec(
    dim_queries: dict[str, list[str]],
    scale_definitions: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Convert in-GUI state to the canonical ``score()``-compatible scales spec.

    The Gradio UI tracks dimensions and scales separately for editor
    ergonomics: dimensions as ``{name: [query, ...]}`` and scales as
    ``{name: {"Positive": [...], "Negative": [...]}}``. The headless
    :func:`spar_measure.score` API and the CLI expect a single nested
    dict with ``dimensions`` and ``scales`` keys at the top level, where
    each scale uses ``pos_dims`` / ``neg_dims``. This helper bridges the
    two so that a JSON file written by the GUI can be loaded directly into
    ``score()`` without any manual reshaping.

    Args:
        dim_queries: Mapping from dimension name to list of seed query
            strings (the shape held in ``MeasurementState["dim_queries"]``).
        scale_definitions: Mapping from scale name to a
            ``{"Positive": [...], "Negative": [...]}`` dict (the shape
            held in ``MeasurementState["scale_definitions"]``). Pass
            ``None`` or ``{}`` for a partial export taken before scales
            are defined; in that case the returned dict has only the
            ``"dimensions"`` key.

    Returns:
        A dict of the form expected by :func:`spar_measure.score`::

            {
                "dimensions": {dim_name: {"queries": [...]}, ...},
                "scales":     {scale_name: {"pos_dims": [...],
                                            "neg_dims": [...]}, ...}
            }

        If ``scale_definitions`` is empty or ``None``, the ``"scales"``
        key is omitted (the file is then a save-point rather than a
        runnable spec).

    Raises:
        ValueError: If any scale definition references a dimension name
            that is not present in ``dim_queries``. Catching this here
            (rather than letting :func:`spar_measure.core.measure` raise a
            confusing ``KeyError`` deep in the projection pipeline) gives
            workshop attendees a clear, actionable error that names the
            offending scale and dimension.
    """
    spec: dict[str, dict[str, Any]] = {
        "dimensions": {
            name: {"queries": list(qs)} for name, qs in dim_queries.items()
        }
    }
    if scale_definitions:
        known_dims = set(dim_queries.keys())
        scales_out: dict[str, dict[str, list[str]]] = {}
        for name, definition in scale_definitions.items():
            pos = list(definition.get("Positive", []))
            neg = list(definition.get("Negative", []))
            unknown = [d for d in (*pos, *neg) if d not in known_dims]
            if unknown:
                raise ValueError(
                    f"Scale {name!r} references unknown dimension(s) "
                    f"{unknown}. Defined dimensions: {sorted(known_dims) or '(none)'}. "
                    f"Either rename/define the missing dimension(s) or remove "
                    f"them from the scale before saving."
                )
            scales_out[name] = {"pos_dims": pos, "neg_dims": neg}
        spec["scales"] = scales_out
    return spec


def write_scales_spec(
    spec: dict[str, dict[str, Any]],
    path: str | Path,
) -> Path:
    """Write a scales spec to disk as pretty-printed JSON.

    Args:
        spec: The scales spec to write (typically from
            :func:`gui_state_to_scales_spec`).
        path: Destination path. Parent directory is created if missing.

    Returns:
        The resolved destination path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    return out_path


__all__ = [
    "read_csv_smart",
    "load_embeddings",
    "gui_state_to_scales_spec",
    "write_scales_spec",
    "SCALES_JSON_NAME",
]
