# SPDX-License-Identifier: GPL-3.0-or-later
"""Extended tests for :class:`spar_measure.state.MeasurementState`.

Covers all ``assert_ready_for`` branches and the dict-access shim edge cases
not covered by :mod:`tests.test_state`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spar_measure.state import MeasurementState


# ---------------------------------------------------------------------------
# assert_ready_for("embed") — full branch coverage
# ---------------------------------------------------------------------------


def test_assert_ready_embed_corpus_present_but_no_col() -> None:
    """``assert_ready_for('embed')`` raises when ``doc_col_name`` is unset.

    The user has uploaded a corpus but has not selected the text column yet.
    """
    s = MeasurementState(
        input_df=pd.DataFrame({"text": ["hello"]}),
        doc_col_name=None,
        use_openai=False,
    )
    with pytest.raises(ValueError, match="column"):
        s.assert_ready_for("embed")


def test_assert_ready_embed_missing_model_sbert() -> None:
    """``assert_ready_for('embed')`` raises when SBERT model is not loaded.

    The corpus and column are ready, ``use_openai=False``, but ``model`` is
    ``None`` (model not loaded yet).
    """
    s = MeasurementState(
        input_df=pd.DataFrame({"text": ["hello"]}),
        doc_col_name="text",
        use_openai=False,
        model=None,
    )
    with pytest.raises(ValueError, match="[Mm]odel"):
        s.assert_ready_for("embed")


def test_assert_ready_embed_openai_missing_api_key() -> None:
    """``assert_ready_for('embed')`` raises when OpenAI mode has no API key.

    The corpus and column are ready, ``use_openai=True``, but ``openai_api_key``
    is ``None``.
    """
    s = MeasurementState(
        input_df=pd.DataFrame({"text": ["hello"]}),
        doc_col_name="text",
        use_openai=True,
        openai_api_key=None,
    )
    with pytest.raises(ValueError, match="[Aa][Pp][Ii]|key"):
        s.assert_ready_for("embed")


def test_assert_ready_embed_openai_with_api_key_passes() -> None:
    """``assert_ready_for('embed')`` does not raise when OpenAI key is set."""
    s = MeasurementState(
        input_df=pd.DataFrame({"text": ["hello"]}),
        doc_col_name="text",
        use_openai=True,
        openai_api_key="sk-fake-key",
    )
    # Should not raise.
    s.assert_ready_for("embed")


# ---------------------------------------------------------------------------
# assert_ready_for("search")
# ---------------------------------------------------------------------------


def test_assert_ready_search_missing_embeddings() -> None:
    """``assert_ready_for('search')`` raises when embeddings are not computed."""
    s = MeasurementState(embeddings=None)
    with pytest.raises(ValueError, match="[Ee]mbedding"):
        s.assert_ready_for("search")


def test_assert_ready_search_with_embeddings_passes() -> None:
    """``assert_ready_for('search')`` does not raise when embeddings are present."""
    s = MeasurementState(embeddings=np.zeros((5, 4)))
    s.assert_ready_for("search")  # should not raise


# ---------------------------------------------------------------------------
# assert_ready_for("measure")
# ---------------------------------------------------------------------------


def test_assert_ready_measure_missing_embeddings() -> None:
    """``assert_ready_for('measure')`` raises when embeddings are absent.

    Even if scales are defined, missing embeddings must be caught.
    """
    s = MeasurementState(
        embeddings=None,
        scale_embeddings={"A": np.zeros(4)},
    )
    with pytest.raises(ValueError, match="[Ee]mbedding"):
        s.assert_ready_for("measure")


def test_assert_ready_measure_with_embeddings_and_scales_passes() -> None:
    """``assert_ready_for('measure')`` passes when both embeddings and scales are set."""
    s = MeasurementState(
        embeddings=np.zeros((5, 4)),
        scale_embeddings={"A": np.zeros(4)},
    )
    s.assert_ready_for("measure")  # should not raise


# ---------------------------------------------------------------------------
# assert_ready_for: unknown stage
# ---------------------------------------------------------------------------


def test_assert_ready_unknown_stage_raises() -> None:
    """An unrecognized stage name raises ``ValueError`` mentioning 'Unknown'."""
    s = MeasurementState()
    with pytest.raises(ValueError, match="[Uu]nknown"):
        s.assert_ready_for("frobnicate")


# ---------------------------------------------------------------------------
# __contains__ and .get() on None fields
# ---------------------------------------------------------------------------


def test_contains_returns_false_for_none_field() -> None:
    """``'embeddings' not in state`` evaluates to ``True`` when ``embeddings=None``."""
    s = MeasurementState()
    assert "embeddings" not in s


def test_contains_returns_true_for_set_field() -> None:
    """``'embeddings' in state`` evaluates to ``True`` when embeddings are set."""
    s = MeasurementState(embeddings=np.zeros((3, 4)))
    assert "embeddings" in s


def test_get_returns_default_for_none_field() -> None:
    """``state.get('embeddings', 'sentinel')`` returns the default when field is ``None``."""
    s = MeasurementState()
    result = s.get("embeddings", "sentinel")
    assert result == "sentinel"


def test_get_returns_value_when_set() -> None:
    """``state.get('doc_col_name', None)`` returns the actual value when set."""
    s = MeasurementState(doc_col_name="post_text")
    assert s.get("doc_col_name") == "post_text"


def test_get_returns_default_for_unknown_key() -> None:
    """``state.get('nonexistent_key', 42)`` returns the default for unknown keys."""
    s = MeasurementState()
    result = s.get("nonexistent_key", 42)
    assert result == 42
