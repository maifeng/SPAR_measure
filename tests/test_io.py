# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`spar_measure.io` CSV and NPY helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spar_measure.io import (
    SCALES_JSON_NAME,
    gui_state_to_scales_spec,
    load_embeddings,
    read_csv_smart,
    write_scales_spec,
)


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


# --- gui_state_to_scales_spec / write_scales_spec ---------------------------


def test_gui_state_to_scales_spec_full_roundtrip() -> None:
    """Full GUI state (dims + scales) converts to a score()-loadable spec."""
    dim_queries = {
        "Innovation": ["we innovate", "we adapt"],
        "Tradition":  ["we follow tradition", "we trust the past"],
    }
    scale_definitions = {
        "Innovation-Tradition": {
            "Positive": ["Innovation"],
            "Negative": ["Tradition"],
        },
    }
    spec = gui_state_to_scales_spec(dim_queries, scale_definitions)
    assert spec == {
        "dimensions": {
            "Innovation": {"queries": ["we innovate", "we adapt"]},
            "Tradition":  {"queries": ["we follow tradition", "we trust the past"]},
        },
        "scales": {
            "Innovation-Tradition": {
                "pos_dims": ["Innovation"],
                "neg_dims": ["Tradition"],
            },
        },
    }


def test_gui_state_to_scales_spec_partial_drops_scales_key() -> None:
    """Partial save (no scales yet) omits the ``scales`` key entirely."""
    spec = gui_state_to_scales_spec({"D": ["q"]}, scale_definitions=None)
    assert spec == {"dimensions": {"D": {"queries": ["q"]}}}
    assert "scales" not in spec

    spec_empty = gui_state_to_scales_spec({"D": ["q"]}, scale_definitions={})
    assert "scales" not in spec_empty


def test_gui_state_to_scales_spec_handles_unipolar_scale() -> None:
    """A scale with only Positive dims serialises with empty neg_dims."""
    dim_queries = {"D": ["q"]}
    scale_definitions = {"S": {"Positive": ["D"], "Negative": []}}
    spec = gui_state_to_scales_spec(dim_queries, scale_definitions)
    assert spec["scales"]["S"] == {"pos_dims": ["D"], "neg_dims": []}


def test_gui_state_to_scales_spec_handles_compound_pole() -> None:
    """Multiple dims per pole survive the round trip."""
    dim_queries = {"A": ["a"], "B": ["b"], "C": ["c"], "D": ["d"]}
    scale_definitions = {
        "X-Y": {"Positive": ["A", "B"], "Negative": ["C", "D"]},
    }
    spec = gui_state_to_scales_spec(dim_queries, scale_definitions)
    assert spec["scales"]["X-Y"] == {
        "pos_dims": ["A", "B"],
        "neg_dims": ["C", "D"],
    }


def test_write_scales_spec_writes_pretty_json(tmp_path: Path) -> None:
    """``write_scales_spec`` writes UTF-8, indented JSON, and creates parent dirs."""
    dim_queries = {"Innovation": ["we innovate"]}
    scale_definitions = {
        "Innovation-Tradition": {"Positive": ["Innovation"], "Negative": []},
    }
    spec = gui_state_to_scales_spec(dim_queries, scale_definitions)
    target = tmp_path / "nested" / "subdir" / SCALES_JSON_NAME
    written_path = write_scales_spec(spec, target)
    assert written_path == target
    assert target.exists()
    raw = target.read_text(encoding="utf-8")
    assert "\n" in raw and "  " in raw, "Output should be pretty-printed"
    assert json.loads(raw) == spec


def test_write_scales_spec_handles_unicode(tmp_path: Path) -> None:
    """Non-ASCII seed strings survive the JSON write."""
    dim_queries = {"Sentiment": ["nous sommes heureux", "我们很开心"]}
    spec = gui_state_to_scales_spec(dim_queries, {})
    target = tmp_path / SCALES_JSON_NAME
    write_scales_spec(spec, target)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["dimensions"]["Sentiment"]["queries"] == [
        "nous sommes heureux",
        "我们很开心",
    ]


def test_scales_json_loads_into_score_directly(tmp_path: Path, sample_docs, sample_embeddings) -> None:
    """End-to-end: GUI state -> JSON file -> json.load -> score() runs."""
    from spar_measure import score

    dim_queries = {
        "Innovation": ["we innovate", "we adapt"],
        "Tradition":  ["we follow tradition", "we trust the past"],
    }
    scale_definitions = {
        "Innovation-Tradition": {
            "Positive": ["Innovation"],
            "Negative": ["Tradition"],
        },
    }
    spec = gui_state_to_scales_spec(dim_queries, scale_definitions)
    target = tmp_path / SCALES_JSON_NAME
    write_scales_spec(spec, target)

    # User's notebook flow: load json, pass to score()
    with open(target, encoding="utf-8") as f:
        loaded_spec = json.load(f)
    out = score(
        sample_docs.head(20),
        loaded_spec,
        text_col="text",
        id_col="doc_id",
        precomputed_embeddings=sample_embeddings[:20],
    )

    assert list(out.columns) == ["doc_id", "Innovation-Tradition"]
    assert len(out) == 20
    assert np.isfinite(out["Innovation-Tradition"]).all()


def test_gui_state_to_scales_spec_raises_on_missing_dim() -> None:
    """A scale that references a non-existent dimension fails fast with a
    clear, actionable error message naming the offending scale and dim
    (code review issue 4)."""
    dim_queries = {"Innovation": ["we innovate"]}
    scale_definitions = {
        "Bad-Scale": {"Positive": ["Innovation"], "Negative": ["Tradtion"]},
    }
    with pytest.raises(ValueError, match="Bad-Scale.*Tradtion"):
        gui_state_to_scales_spec(dim_queries, scale_definitions)


def test_gui_state_to_scales_spec_raises_lists_all_missing() -> None:
    """When multiple dims are missing, the error names all of them so the
    user does not have to fix-and-retry repeatedly."""
    dim_queries = {"D1": ["q"]}
    scale_definitions = {
        "S": {"Positive": ["D1", "D2", "D3"], "Negative": ["D4"]},
    }
    with pytest.raises(ValueError) as exc:
        gui_state_to_scales_spec(dim_queries, scale_definitions)
    msg = str(exc.value)
    # All three missing dims are named in the error message
    assert "D2" in msg and "D3" in msg and "D4" in msg
    # The "unknown" list specifically does NOT include D1 (which is defined)
    unknown_chunk = msg.split("unknown dimension(s)")[1].split(".")[0]
    assert "D1" not in unknown_chunk
    for d in ("D2", "D3", "D4"):
        assert d in unknown_chunk


def test_gui_state_to_scales_spec_tolerates_missing_pole_keys() -> None:
    """Scale definition without a Positive (or Negative) key falls back to
    an empty pole rather than raising KeyError. Pins .get() default."""
    dim_queries = {"D": ["q"]}
    spec = gui_state_to_scales_spec(
        dim_queries, scale_definitions={"S": {"Positive": ["D"]}}
    )
    assert spec["scales"]["S"] == {"pos_dims": ["D"], "neg_dims": []}


def test_gui_state_to_scales_spec_handles_unicode_dim_and_scale_names() -> None:
    """Non-ASCII dimension and scale names survive the conversion (extends
    the existing unicode test which only covered query strings)."""
    dim_queries = {"创新": ["we innovate"], "传统": ["we follow tradition"]}
    scale_definitions = {
        "创新-传统": {"Positive": ["创新"], "Negative": ["传统"]},
    }
    spec = gui_state_to_scales_spec(dim_queries, scale_definitions)
    assert "创新" in spec["dimensions"]
    assert "创新-传统" in spec["scales"]
    assert spec["scales"]["创新-传统"]["pos_dims"] == ["创新"]


def test_write_scales_spec_overwrites_existing_file(tmp_path: Path) -> None:
    """Writing to an existing path replaces its contents atomically from
    the user's perspective: the partial-save → full-save sequence is the
    common GUI flow."""
    target = tmp_path / SCALES_JSON_NAME

    partial = gui_state_to_scales_spec({"Innovation": ["q1"]})
    write_scales_spec(partial, target)
    assert "scales" not in json.loads(target.read_text())

    full = gui_state_to_scales_spec(
        {"Innovation": ["q1"], "Tradition": ["q2"]},
        {"Innovation-Tradition": {"Positive": ["Innovation"], "Negative": ["Tradition"]}},
    )
    write_scales_spec(full, target)
    loaded = json.loads(target.read_text())
    assert "scales" in loaded
    assert set(loaded["dimensions"].keys()) == {"Innovation", "Tradition"}


def test_scales_json_name_constant() -> None:
    """Pin the canonical filename so any rename trips a test before
    it silently breaks the README, the docstrings, and Colab examples."""
    assert SCALES_JSON_NAME == "scales.json"


def test_scales_json_matches_inline_spec(tmp_path: Path, sample_docs, sample_embeddings) -> None:
    """Scoring through the JSON path gives bit-exact same results as inline."""
    from spar_measure import score

    inline_spec = {
        "dimensions": {
            "Innovation": {"queries": ["we innovate", "we adapt"]},
            "Tradition":  {"queries": ["we follow tradition", "we trust the past"]},
        },
        "scales": {
            "Innovation-Tradition": {
                "pos_dims": ["Innovation"],
                "neg_dims": ["Tradition"],
            },
        },
    }
    inline_out = score(
        sample_docs.head(50),
        inline_spec,
        text_col="text",
        id_col="doc_id",
        precomputed_embeddings=sample_embeddings[:50],
    )

    # Now go through the GUI-state -> JSON -> load -> score path.
    spec = gui_state_to_scales_spec(
        dim_queries={
            "Innovation": ["we innovate", "we adapt"],
            "Tradition":  ["we follow tradition", "we trust the past"],
        },
        scale_definitions={
            "Innovation-Tradition": {
                "Positive": ["Innovation"],
                "Negative": ["Tradition"],
            },
        },
    )
    target = tmp_path / SCALES_JSON_NAME
    write_scales_spec(spec, target)
    with open(target, encoding="utf-8") as f:
        gui_spec = json.load(f)
    gui_out = score(
        sample_docs.head(50),
        gui_spec,
        text_col="text",
        id_col="doc_id",
        precomputed_embeddings=sample_embeddings[:50],
    )

    np.testing.assert_array_almost_equal(
        inline_out["Innovation-Tradition"].to_numpy(),
        gui_out["Innovation-Tradition"].to_numpy(),
        decimal=10,
    )
