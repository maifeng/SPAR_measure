# SPDX-License-Identifier: GPL-3.0-or-later
"""SPAR: Semantic Projection with Active Retrieval.

Reference implementation for Yan, Mai, Wu, Chen, and Li (2024),
"A Computational Framework for Understanding Firm Communication During
Disasters," *Information Systems Research* 35(2): 590-608,
https://doi.org/10.1287/isre.2022.0128.

Top-level API:

- :func:`score` (alias: :func:`measure`): headless scoring of a document
  DataFrame against a scale spec. No Gradio, no browser tunnel.
- :class:`Measurement`: Gradio callback adapter (see :mod:`ui`).
- :func:`run_gui`: launch the Gradio UI.

Example:

    >>> from spar_measure import score
    >>> out = score(docs_df, scales_dict, text_col="text", id_col="doc_id")  # doctest: +SKIP
"""

from __future__ import annotations

__version__ = "0.3.0a1"
__paper__ = "Yan, Mai, Wu, Chen & Li (2024), ISR 35(2):590-608"

from .api import measure, score
from .core import project_documents
from .state import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_SBERT_MODEL,
    MAX_DIMENSIONS,
    MeasurementState,
    OPENAI_EMBEDDING_MODELS,
)
from .ui import CVFDemo, Measurement, PathManager, run_gui

# Backwards-compat alias for the original misspelled class name.
Meaurement = Measurement

__all__ = [
    "__version__",
    "__paper__",
    "score",
    "measure",
    "project_documents",
    "MeasurementState",
    "Measurement",
    "Meaurement",
    "PathManager",
    "CVFDemo",
    "run_gui",
    "MAX_DIMENSIONS",
    "DEFAULT_EMBED_BATCH_SIZE",
    "DEFAULT_SBERT_MODEL",
    "DEFAULT_OPENAI_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODELS",
]
