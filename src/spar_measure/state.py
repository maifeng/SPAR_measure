# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed state container for SPAR measurement sessions.

Replaces the 16-key untyped ``gr.State({})`` dict that previously flowed
through every Gradio callback. Fields default to ``None`` until each tab of
the workflow populates them; the :meth:`MeasurementState.assert_ready_for`
helper validates preconditions between tabs.

Fixes M-02 per ``digest/05_SPAR_code_review.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# Module-level constants. Fixes M-03 (magic constant 10 duplicated 6 places)
# and M-04 (hardcoded batch size 8) per 05_SPAR_code_review.md.
MAX_DIMENSIONS: int = 10
"""Maximum number of dimensions (and scales) the Gradio UI renders.

Each dimension adds one row to Tab 2 and each scale adds one row to Tab 3.
Raising this requires no code changes beyond bumping the constant, but the UI
becomes unwieldy past ~12.
"""

DEFAULT_EMBED_BATCH_SIZE: int = 8
"""Default batch size for Sentence Transformers and OpenAI embedding calls.

The original value 8 was tuned for small GPUs (e.g., Colab T4). Larger GPUs
can safely raise this to 32-128 for Sentence Transformers; OpenAI's batch
embedding endpoint accepts up to 2048 inputs per request.
"""

DEFAULT_SBERT_MODEL: str = "all-MiniLM-L6-v2"
"""Default Sentence-BERT model loaded from the ``sentence-transformers`` namespace.

384-dim, 22M parameters, ships with the bundled ``sample_emb.npy``.
"""

DEFAULT_OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
"""Default OpenAI embedding model.

Fixes H-03: ``text-embedding-ada-002`` was superseded in Jan 2024 by the
``text-embedding-3-*`` family. ``3-small`` is cheaper and scores better on
MTEB.
"""

OPENAI_EMBEDDING_MODELS: tuple[str, ...] = (
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
)
"""Supported OpenAI embedding model names, newest first."""


@dataclass
class MeasurementState:
    """Typed session state for SPAR measurement.

    Attributes:
        input_df: Loaded corpus as a pandas DataFrame. Set by Tab 1.
        col_names: Column names extracted from ``input_df``.
        embeddings: ``(n_docs, dim)`` float32 array of document embeddings.
        doc_col_name: Column in ``input_df`` that holds the text to embed.
        doc_id_col_name: Column in ``input_df`` that holds the document ID.
        model: Loaded ``transformers`` model. ``None`` if using OpenAI.
        tokenizer: Loaded ``transformers`` tokenizer. ``None`` if using OpenAI.
        model_name: Human-readable name of the active embedding model.
        use_openai: ``True`` if using the OpenAI Embedding API.
        openai_api_key: OpenAI API key (not persisted to disk).
        openai_embedding_model: Name of the OpenAI embedding model.
        dim_embeddings: Map from dimension name to its averaged embedding.
        dim_queries: Map from dimension name to the list of seed queries.
        scale_embeddings: Map from scale name to its embedding vector.
        scale_definitions: Map from scale name to its pos/neg dimension lists.
        n_dims: Currently active number of dimensions in the UI.
        n_scales: Currently active number of scales in the UI.
        batch_size: Batch size used by :func:`embed_texts`.
    """

    input_df: pd.DataFrame | None = None
    col_names: list[str] | None = None
    embeddings: np.ndarray | None = None
    doc_col_name: str | None = None
    doc_id_col_name: str | None = None
    model: Any = None
    tokenizer: Any = None
    model_name: str | None = None
    use_openai: bool = False
    openai_api_key: str | None = None
    openai_embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    dim_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    dim_queries: dict[str, list[str]] = field(default_factory=dict)
    scale_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    scale_definitions: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    n_dims: int = 4
    n_scales: int = 2
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE

    # --- dict-compatibility shim ------------------------------------------------
    # The Gradio callbacks in ``ui.py`` were originally written against a plain
    # dict. We expose item-access so the incremental refactor does not need to
    # touch every call site in one commit.

    _KEY_ALIASES: dict[str, str] = field(
        default_factory=lambda: {"use_openAI": "use_openai"},
        init=False,
        repr=False,
    )

    def _resolve(self, key: str) -> str:
        return self._KEY_ALIASES.get(key, key)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, self._resolve(key))

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, self._resolve(key), value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, self._resolve(key)) and getattr(self, self._resolve(key)) is not None

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style ``get`` with default fallback."""
        try:
            value = getattr(self, self._resolve(key))
        except AttributeError:
            return default
        return value if value is not None else default

    # --- validation helpers ----------------------------------------------------

    def assert_ready_for(self, stage: str) -> None:
        """Validate that the state has the fields a given stage needs.

        Args:
            stage: One of ``"embed"``, ``"search"``, ``"measure"``.

        Raises:
            ValueError: if a required field is missing.
        """
        if stage == "embed":
            if self.input_df is None:
                raise ValueError("No corpus loaded; upload a CSV first.")
            if self.doc_col_name is None:
                raise ValueError("Document text column not selected.")
            if not self.use_openai and self.model is None:
                raise ValueError("Sentence Transformers model not loaded.")
            if self.use_openai and not self.openai_api_key:
                raise ValueError("OpenAI API key not set.")
        elif stage == "search":
            if self.embeddings is None:
                raise ValueError("Corpus embeddings not computed.")
        elif stage == "measure":
            if self.embeddings is None:
                raise ValueError("Corpus embeddings not computed.")
            if not self.scale_embeddings:
                raise ValueError("No scales defined.")
        else:
            raise ValueError(f"Unknown stage: {stage!r}")


__all__ = [
    "MeasurementState",
    "MAX_DIMENSIONS",
    "DEFAULT_EMBED_BATCH_SIZE",
    "DEFAULT_SBERT_MODEL",
    "DEFAULT_OPENAI_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODELS",
]
