# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`spar_measure.io` CSV and NPY helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spar_measure.io import load_embeddings, read_csv_smart


def test_read_csv_smart_utf8(tmp_path: Path) -> None:
    """Reads UTF-8 CSVs cleanly (the common case)."""
    p = tmp_path / "utf8.csv"
    p.write_text("name,text\nfoo,We should innovate.\n", encoding="utf-8")
    df = read_csv_smart(p)
    assert df.iloc[0]["text"] == "We should innovate."


def test_read_csv_smart_latin1_fallback(tmp_path: Path) -> None:
    """Falls back to latin-1 when UTF-8 decode fails."""
    p = tmp_path / "latin.csv"
    # 0xE9 is 'é' in latin-1 but an invalid UTF-8 start byte.
    content = b"name,text\nfoo,caf\xe9\n"
    p.write_bytes(content)
    df = read_csv_smart(p)
    assert "caf" in df.iloc[0]["text"]


def test_load_embeddings_validates_ndim(tmp_path: Path) -> None:
    """1D arrays are rejected."""
    p = tmp_path / "bad.npy"
    np.save(p, np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError, match="2D"):
        load_embeddings(p)


def test_load_embeddings_validates_finite(tmp_path: Path) -> None:
    """Arrays with NaN are rejected."""
    p = tmp_path / "nan.npy"
    arr = np.zeros((3, 4), dtype=np.float32)
    arr[0, 0] = np.nan
    np.save(p, arr)
    with pytest.raises(ValueError, match="NaN"):
        load_embeddings(p)


def test_load_embeddings_casts_to_float32(tmp_path: Path) -> None:
    """float64 input is returned as float32."""
    p = tmp_path / "ok.npy"
    np.save(p, np.random.default_rng(0).standard_normal((3, 4)))
    arr = load_embeddings(p)
    assert arr.dtype == np.float32
    assert arr.shape == (3, 4)
