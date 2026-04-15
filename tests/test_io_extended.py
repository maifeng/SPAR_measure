# SPDX-License-Identifier: GPL-3.0-or-later
"""Extended tests for :mod:`spar_measure.io`.

Covers multi-row latin-1 files, explicit encoding overrides, wrong-dtype
numpy arrays, and 3D array rejection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spar_measure.io import load_embeddings, read_csv_smart


# ---------------------------------------------------------------------------
# read_csv_smart
# ---------------------------------------------------------------------------


def test_read_csv_smart_latin1_multirow(tmp_path: Path) -> None:
    """Multi-row CSV with French diacritics parses without error.

    Words like ``café``, ``résumé``, and ``naïve`` contain latin-1 bytes
    that are invalid UTF-8 start sequences. The function must fall back
    to latin-1 and return all rows intact.
    """
    rows = [
        b"id,text",
        b"1,caf\xe9",      # é
        b"2,r\xe9sum\xe9", # résumé
        b"3,na\xefve",     # naïve
    ]
    content = b"\n".join(rows) + b"\n"
    p = tmp_path / "french.csv"
    p.write_bytes(content)
    df = read_csv_smart(p)
    assert len(df) == 3
    assert df["id"].tolist() == [1, 2, 3]
    # Verify that diacritics survived (latin-1 decode gives the right chars).
    assert "caf" in df.loc[0, "text"]
    assert "sum" in df.loc[1, "text"]


def test_read_csv_smart_explicit_utf8_raises_on_latin1(tmp_path: Path) -> None:
    """Passing ``encoding='utf-8'`` bypasses the fallback and raises on a latin-1 file.

    This confirms the ``encoding`` override is respected rather than ignored.
    """
    p = tmp_path / "bad_utf8.csv"
    p.write_bytes(b"name,text\nfoo,caf\xe9\n")
    with pytest.raises(UnicodeDecodeError):
        read_csv_smart(p, encoding="utf-8")


def test_read_csv_smart_explicit_latin1_works(tmp_path: Path) -> None:
    """Passing ``encoding='latin-1'`` explicitly reads a latin-1 file without warnings."""
    p = tmp_path / "ok_latin1.csv"
    p.write_bytes(b"name,text\nfoo,caf\xe9\n")
    df = read_csv_smart(p, encoding="latin-1")
    assert df.iloc[0]["name"] == "foo"


# ---------------------------------------------------------------------------
# load_embeddings: dtype and shape validation
# ---------------------------------------------------------------------------


def test_load_embeddings_wrong_dtype_raises(tmp_path: Path) -> None:
    """An int32 embedding matrix raises ``ValueError`` mentioning dtype."""
    p = tmp_path / "int.npy"
    np.save(p, np.zeros((4, 8), dtype=np.int32))
    with pytest.raises(ValueError, match="dtype"):
        load_embeddings(p)


def test_load_embeddings_3d_raises(tmp_path: Path) -> None:
    """A 3D array raises ``ValueError`` mentioning '2D'."""
    p = tmp_path / "three_d.npy"
    np.save(p, np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="2D"):
        load_embeddings(p)


def test_load_embeddings_float64_casts_to_float32(tmp_path: Path) -> None:
    """float64 input is silently cast to float32."""
    arr = np.random.default_rng(5).standard_normal((6, 12))  # float64
    p = tmp_path / "f64.npy"
    np.save(p, arr)
    out = load_embeddings(p)
    assert out.dtype == np.float32
    assert out.shape == (6, 12)
    np.testing.assert_allclose(out, arr.astype(np.float32), atol=1e-5)


def test_load_embeddings_inf_raises(tmp_path: Path) -> None:
    """Arrays with Inf values are rejected."""
    arr = np.ones((3, 4), dtype=np.float32)
    arr[1, 2] = np.inf
    p = tmp_path / "inf.npy"
    np.save(p, arr)
    with pytest.raises(ValueError, match="NaN|Inf"):
        load_embeddings(p)
