# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared pytest fixtures for the SPAR test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_data_dir() -> Path:
    """Path to the bundled sample_data directory."""
    import spar_measure

    return Path(spar_measure.__file__).parent / "sample_data"


@pytest.fixture(scope="session")
def sample_docs(sample_data_dir: Path) -> pd.DataFrame:
    """The bundled 2000-row Facebook-post corpus."""
    return pd.read_csv(sample_data_dir / "sample_text.csv")


@pytest.fixture(scope="session")
def sample_embeddings(sample_data_dir: Path) -> np.ndarray:
    """Precomputed all-MiniLM-L6-v2 embeddings for :func:`sample_docs`."""
    return np.load(sample_data_dir / "sample_emb.npy")


@pytest.fixture()
def cvf_scales() -> dict:
    """The CVF example scale spec used throughout the code review."""
    return {
        "dimensions": {
            "Create":      {"queries": ["We should adapt and innovate."]},
            "Collaborate": {"queries": ["We should empathize and collaborate."]},
            "Control":     {"queries": ["We should control and stabilize."]},
            "Compete":     {"queries": ["We should respond swiftly and serve customers."]},
        },
        "scales": {
            "External-Internal": {
                "pos_dims": ["Create", "Compete"],
                "neg_dims": ["Control", "Collaborate"],
            },
            "Flexible-Stable": {
                "pos_dims": ["Collaborate", "Create"],
                "neg_dims": ["Control", "Compete"],
            },
        },
    }
