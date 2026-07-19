"""CPU unit tests for the Run 10 channel-axis estimators (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from audio_safety.pipelines.channel_axis import (
    largest_principal_angle,
    mean_anchored_basis,
    paired_differences,
    project,
    reconstruction_ratio,
    refusal_dim_direction,
    select_rank,
    unit,
)


def _orthonormal(basis: np.ndarray, tol: float = 1e-8) -> bool:
    gram = basis @ basis.T
    return np.allclose(gram, np.eye(basis.shape[0]), atol=tol)


def test_unit_normalizes_and_is_zero_safe():
    assert np.isclose(np.linalg.norm(unit([3.0, 4.0])), 1.0)
    # zero vector must not raise or return nan
    assert np.all(np.isfinite(unit(np.zeros(5))))


def test_paired_differences_shape_guard():
    with pytest.raises(ValueError):
        paired_differences(np.zeros((3, 4)), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        paired_differences(np.zeros(4), np.zeros(4))
    diff = paired_differences(np.ones((2, 3)), np.zeros((2, 3)))
    assert np.allclose(diff, 1.0)


def test_mean_anchored_basis_rank1_is_unit_mean_diff():
    rng = np.random.default_rng(0)
    diffs = rng.normal(size=(50, 8)) + np.array([5.0, 0, 0, 0, 0, 0, 0, 0])
    basis = mean_anchored_basis(diffs, rank=1)
    assert basis.shape == (1, 8)
    assert np.allclose(basis[0], unit(diffs.mean(axis=0)))


def test_mean_anchored_basis_is_orthonormal_and_u1_first():
    rng = np.random.default_rng(1)
    diffs = rng.normal(size=(200, 12)) + 3.0 * np.eye(1, 12, 0).ravel()
    basis = mean_anchored_basis(diffs, rank=4)
    assert basis.shape == (4, 12)
    assert _orthonormal(basis)
    # row 0 stays the difference-in-means direction
    assert np.allclose(basis[0], unit(diffs.mean(axis=0)))
    # residual rows are orthogonal to u1 (so P_U is a genuine projection)
    for row in basis[1:]:
        assert abs(float(row @ basis[0])) < 1e-8


def test_mean_anchored_basis_rejects_zero_mean_and_overrank():
    rng = np.random.default_rng(2)
    # symmetric diffs -> ~zero mean -> u1 undefined
    x = rng.normal(size=(4, 6))
    diffs = np.concatenate([x, -x], axis=0)
    with pytest.raises(ValueError):
        mean_anchored_basis(diffs, rank=1)
    with pytest.raises(ValueError):
        mean_anchored_basis(rng.normal(size=(5, 3)) + 1.0, rank=9)


def test_project_is_idempotent_and_fixes_span():
    rng = np.random.default_rng(3)
    diffs = rng.normal(size=(80, 10)) + 2.0
    basis = mean_anchored_basis(diffs, rank=3)
    x = rng.normal(size=(7, 10))
    px = project(x, basis)
    assert np.allclose(project(px, basis), px, atol=1e-8)  # idempotent
    # a vector already in the span is returned unchanged
    in_span = (rng.normal(size=(1, 3)) @ basis)
    assert np.allclose(project(in_span, basis), in_span, atol=1e-8)


def test_projected_transport_reciprocity_numpy():
    """The core Run 10 property: swapping only the U-component makes the patched
    state share the donor's subspace coordinate (restoration and corruption are
    subspace reciprocals). Verified here at the numpy level."""
    rng = np.random.default_rng(4)
    diffs = rng.normal(size=(120, 16)) + 4.0
    basis = mean_anchored_basis(diffs, rank=5)
    clean = rng.normal(size=(9, 16))
    attack = rng.normal(size=(9, 16))

    restore = attack + project(clean - attack, basis)   # pass on attack, donor=clean
    corrupt = clean + project(attack - clean, basis)     # pass on clean, donor=attack

    # patched state carries the donor's subspace coordinate exactly
    assert np.allclose(project(restore, basis), project(clean, basis), atol=1e-8)
    assert np.allclose(project(corrupt, basis), project(attack, basis), atol=1e-8)
    # and leaves the off-subspace component of the host untouched
    def off(z):
        return z - project(z, basis)

    assert np.allclose(off(restore), off(attack), atol=1e-8)
    assert np.allclose(off(corrupt), off(clean), atol=1e-8)


def test_refusal_dim_direction_sign_and_missing_class():
    p2 = np.array([[1.0, 0.0], [1.2, 0.1], [-1.0, 0.0], [-0.8, -0.1]])
    labels = np.array([1, 1, 0, 0])
    direction = refusal_dim_direction(p2, labels)
    assert direction is not None
    # refused mean - complied mean points toward +x here
    assert direction[0] > 0
    assert np.isclose(np.linalg.norm(direction), 1.0)
    # missing a class -> None
    assert refusal_dim_direction(p2, np.array([1, 1, 1, 1])) is None
    # labels other than 0/1 are ignored
    assert refusal_dim_direction(p2, np.array([1, -1, 0, -1])) is not None


def test_select_rank_is_outcome_blind_and_prefers_small_stable_rank():
    rng = np.random.default_rng(5)
    # a strong shared rank-1 signal + small noise -> low rank should be stable
    base = 6.0 * np.eye(1, 20, 0).ravel()
    train = rng.normal(scale=0.1, size=(300, 20)) + base
    dev = rng.normal(scale=0.1, size=(300, 20)) + base
    rank = select_rank(
        train, dev, [1, 2, 3, 5], min_reconstruction=0.8, max_angle_rad=0.3
    )
    assert rank == 1
    # signature takes no labels -> cannot depend on refusal outcomes
    assert "labels" not in select_rank.__code__.co_varnames


def test_select_rank_raises_when_no_admissible_rank():
    rng = np.random.default_rng(7)
    # independent noise in train vs dev -> unstable subspaces -> nothing qualifies
    base = 6.0 * np.eye(1, 20, 0).ravel()
    train = rng.normal(scale=1.0, size=(40, 20)) + base
    dev = rng.normal(scale=1.0, size=(40, 20)) + base
    with pytest.raises(ValueError, match="no candidate rank"):
        select_rank(train, dev, [3, 5], min_reconstruction=0.999, max_angle_rad=0.001)
    # radians guard: a degrees-like value is rejected
    with pytest.raises(ValueError, match="radians"):
        select_rank(train, dev, [1], min_reconstruction=0.1, max_angle_rad=15.0)


def test_mean_anchored_basis_rejects_rank_deficient_diffs():
    # identical non-zero diffs: u1 defined, but residual is ~0 -> no rank-2 direction
    diffs = np.tile(np.arange(1.0, 9.0), (30, 1))
    assert mean_anchored_basis(diffs, rank=1).shape == (1, 8)
    with pytest.raises(ValueError, match="numerical data rank"):
        mean_anchored_basis(diffs, rank=2)
    # non-finite input is rejected
    bad = diffs.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        mean_anchored_basis(bad, rank=1)


def test_reconstruction_and_angle_bounds():
    rng = np.random.default_rng(6)
    diffs = rng.normal(size=(150, 10)) + 3.0
    basis = mean_anchored_basis(diffs, rank=10)
    # full-rank basis reconstructs everything
    assert reconstruction_ratio(diffs, basis) == pytest.approx(1.0, abs=1e-6)
    # identical subspaces -> zero principal angle (arccos amplifies fp error near 1)
    assert largest_principal_angle(basis, basis) == pytest.approx(0.0, abs=1e-5)
