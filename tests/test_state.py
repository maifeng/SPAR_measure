# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :class:`spar_measure.state.MeasurementState`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spar_measure.state import MAX_DIMENSIONS, MeasurementState


def test_dict_compatibility_set_get() -> None:
    """The dict-access shim lets existing callbacks use MeasurementState."""
    s = MeasurementState()
    s["doc_col_name"] = "text"
    assert s["doc_col_name"] == "text"
    assert s.doc_col_name == "text"


def test_dict_compatibility_use_openai_alias() -> None:
    """The old ``use_openAI`` key transparently routes to ``use_openai``."""
    s = MeasurementState()
    s["use_openAI"] = True
    assert s.use_openai is True
    assert s["use_openAI"] is True


def test_assert_ready_for_embed_missing_df() -> None:
    """``assert_ready_for('embed')`` flags missing corpus."""
    s = MeasurementState()
    with pytest.raises(ValueError, match="corpus"):
        s.assert_ready_for("embed")


def test_assert_ready_for_measure_missing_scales() -> None:
    """``assert_ready_for('measure')`` flags missing scales even with embeddings."""
    s = MeasurementState(embeddings=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="scales"):
        s.assert_ready_for("measure")


def test_max_dimensions_constant() -> None:
    """MAX_DIMENSIONS exposes the UI row cap."""
    assert isinstance(MAX_DIMENSIONS, int)
    assert MAX_DIMENSIONS >= 4
