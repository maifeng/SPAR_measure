# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the pure projection / scaling math in :mod:`spar_measure.core`.

These tests construct toy embedding matrices so they run in <1s and do not
load any Hugging Face model.
"""

from __future__ import annotations

import numpy as np
import pytest

from spar_measure.core import combine_scale, project_documents


def test_combine_scale_pos_only() -> None:
    """``combine_scale`` with only positive dims returns their mean."""
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    out = combine_scale([p1, p2], [])
    np.testing.assert_allclose(out, [0.5, 0.5, 0.0])


def test_combine_scale_neg_only() -> None:
    """``combine_scale`` with only negative dims returns ``-mean(neg)``."""
    n1 = np.array([1.0, 0.0, 0.0])
    out = combine_scale([], [n1])
    np.testing.assert_allclose(out, [-1.0, 0.0, 0.0])


def test_combine_scale_pos_minus_neg() -> None:
    """``combine_scale`` with both returns the difference of means."""
    p = np.array([1.0, 0.0, 0.0])
    n = np.array([0.0, 1.0, 0.0])
    out = combine_scale([p], [n])
    np.testing.assert_allclose(out, [1.0, -1.0, 0.0])


def test_combine_scale_empty_raises() -> None:
    """Scales with neither positive nor negative dims are rejected."""
    with pytest.raises(ValueError):
        combine_scale([], [])


def test_project_documents_dot_score_shape() -> None:
    """``project_documents`` returns (n_docs, n_scales) in 'No subspace' mode."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((7, 4)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    scales = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]),
        "B": np.array([0.0, 1.0, 0.0, 0.0]),
        "C": np.array([0.0, 0.0, 1.0, 0.0]),
    }
    out = project_documents(X, scales, single_subspace=False, whiten=False)
    assert out.shape == (7, 3)
    assert list(out.columns) == ["A", "B", "C"]


def test_project_documents_dot_score_is_inner_product() -> None:
    """Confirm that with unit scales, project_documents reproduces ``X @ Sᵀ``."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 3)).astype(np.float64)
    scales = {
        "A": np.array([1.0, 0.0, 0.0]),
        "B": np.array([0.0, 1.0, 0.0]),
    }
    out = project_documents(X, scales, single_subspace=False, whiten=False)
    # S is normalized inside project_documents; here rows are already unit-norm.
    expected = X @ np.stack([scales["A"], scales["B"]]).T
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-10)


def test_project_documents_single_subspace_recovers_identity() -> None:
    """When scales form an orthonormal basis, the subspace projection matches the dot product."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 3))
    scales = {
        "e1": np.array([1.0, 0.0, 0.0]),
        "e2": np.array([0.0, 1.0, 0.0]),
        "e3": np.array([0.0, 0.0, 1.0]),
    }
    indep = project_documents(X, scales, single_subspace=False).to_numpy()
    joint = project_documents(X, scales, single_subspace=True).to_numpy()
    np.testing.assert_allclose(joint, indep, atol=1e-10)


def test_project_documents_single_subspace_handles_collinearity() -> None:
    """Fixes M-05: joint-subspace projection tolerates near-collinear scales.

    The old ``numpy.linalg.inv`` implementation would blow up here; ``pinv``
    returns the minimum-norm solution.
    """
    X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Two nearly-parallel scales.
    scales = {
        "A": np.array([1.0, 1e-10, 0.0]),
        "B": np.array([1.0, 1e-10, 1e-10]),
    }
    out = project_documents(X, scales, single_subspace=True)
    assert out.shape == (2, 2)
    assert np.isfinite(out.to_numpy()).all()


def test_project_documents_whiten_decorrelates() -> None:
    """ZCA whitening removes off-diagonal correlation in the score matrix."""
    rng = np.random.default_rng(3)
    # Highly correlated scores: scale B is a shifted copy of A.
    n = 500
    X = rng.standard_normal((n, 4))
    scales = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]),
        "B": np.array([1.0, 0.1, 0.0, 0.0]),  # nearly collinear with A
    }
    no_whiten = project_documents(X, scales, whiten=False).to_numpy()
    whiten = project_documents(X, scales, whiten=True).to_numpy()
    corr_raw = np.corrcoef(no_whiten.T)[0, 1]
    corr_w = np.corrcoef(whiten.T)[0, 1]
    assert abs(corr_raw) > 0.5  # sanity: truly correlated
    assert abs(corr_w) < 0.1   # whitening knocked it down
