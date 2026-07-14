"""Pure-CPU tests for the COAST-R B/R/U/f linear core."""

import inspect

import numpy as np
import pytest

from audio_safety.evaluation.coast_r import (
    basis_coordinates,
    deterministic_group_folds,
    fit_dim_refusal_basis,
    fit_natural_predictor,
    fit_reachable_basis,
    fit_reduced_rank_transport,
    make_disjoint_role_rotations,
    orthogonal_residual,
    project_onto_basis,
    reconstruct_from_basis,
)


def _two_dimensional_deltas(n_groups: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    groups = []
    for group in range(n_groups):
        angle = 0.31 * group
        for scale in (0.8, 1.3):
            rows.append([scale * np.cos(angle), scale * np.sin(angle), 0.0, 0.0, 0.0])
            groups.append(f"item{group:02d}")
    return np.asarray(rows, dtype=np.float64), np.asarray(groups, dtype=object)


def test_grouped_folds_are_deterministic_disjoint_and_cover_every_row_once():
    _, groups = _two_dimensional_deltas(n_groups=8)
    first = deterministic_group_folds(groups, 4, seed=7)
    second = deterministic_group_folds(groups, 4, seed=7)
    heldout_counts = np.zeros(len(groups), dtype=np.int8)

    for (train, heldout), (train_again, heldout_again) in zip(first, second, strict=True):
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(heldout, heldout_again)
        assert set(groups[train]).isdisjoint(set(groups[heldout]))
        heldout_counts[heldout] += 1

    np.testing.assert_array_equal(heldout_counts, np.ones(len(groups), dtype=np.int8))


def test_four_way_role_rotations_have_no_item_or_row_leakage():
    _, groups = _two_dimensional_deltas(n_groups=8)
    train_indices = np.arange(len(groups), dtype=np.int64)
    original, swapped = make_disjoint_role_rotations(train_indices, groups, seed=3)

    for assignment in (original, swapped):
        index_roles = (
            assignment.b_indices,
            assignment.r_indices,
            assignment.u_indices,
            assignment.f_indices,
        )
        group_roles = (
            set(assignment.b_groups),
            set(assignment.r_groups),
            set(assignment.u_groups),
            set(assignment.f_groups),
        )
        assert set(np.concatenate(index_roles)) == set(train_indices)
        for offset, role in enumerate(index_roles):
            for other in index_roles[offset + 1 :]:
                assert not np.intersect1d(role, other).size
        for offset, role in enumerate(group_roles):
            for other in group_roles[offset + 1 :]:
                assert role.isdisjoint(other)

    assert original.b_groups == swapped.b_groups
    assert original.f_groups == swapped.f_groups
    assert original.r_groups == swapped.u_groups
    assert original.u_groups == swapped.r_groups


def test_reachable_basis_recovers_natural_rank_and_projection_contract():
    deltas, groups = _two_dimensional_deltas()
    reachable = fit_reachable_basis(
        deltas,
        groups,
        rank_cap=4,
        coverage_target=0.999,
        min_rank=2,
        n_folds=3,
        seed=0,
    )

    assert reachable.selected_rank == 2
    np.testing.assert_allclose(
        reachable.basis @ reachable.basis.T,
        np.eye(2),
        atol=1e-10,
    )
    coordinates = basis_coordinates(deltas, reachable.basis)
    reconstructed = reconstruct_from_basis(coordinates, reachable.basis)
    np.testing.assert_allclose(reconstructed, deltas, atol=1e-10)
    np.testing.assert_allclose(project_onto_basis(deltas, reachable.basis), deltas, atol=1e-10)
    np.testing.assert_allclose(orthogonal_residual(deltas, reachable.basis), 0.0, atol=1e-10)
    assert reachable.metrics()["label_free"] is True


def test_reachable_basis_is_uncentered_and_keeps_a_constant_operator_mode():
    groups = np.asarray([f"item{index}" for index in range(6)], dtype=object)
    # Centered PCA would erase this shared nonzero transport completely. COAST-R B
    # is deliberately fitted around the natural zero-delta origin.
    deltas = np.repeat(np.array([[2.0, -1.0, 0.0]]), len(groups), axis=0)
    reachable = fit_reachable_basis(
        deltas,
        groups,
        rank_cap=2,
        rank_candidates=(1, 2),
        coverage_target=0.999,
        min_rank=1,
        n_folds=3,
        seed=0,
    )

    assert reachable.selected_rank == 1
    assert reachable.cv_coverage[1] == pytest.approx(1.0)
    assert reachable.metrics()["centered"] is False
    np.testing.assert_allclose(reachable.project(deltas), deltas, atol=1e-10)


def test_transport_is_nested_inside_reachable_basis_and_rank_two_adds_signal():
    deltas, groups = _two_dimensional_deltas()
    reachable = fit_reachable_basis(
        deltas,
        groups,
        rank_cap=2,
        coverage_target=0.999,
        min_rank=2,
        n_folds=3,
        seed=0,
    )
    natural = reachable.coordinates(deltas)
    endpoints = np.column_stack((2.0 * natural[:, 0], -3.0 * natural[:, 1]))
    transport = fit_reduced_rank_transport(
        deltas,
        endpoints,
        reachable,
        groups=groups,
        max_rank=2,
        ridge_alphas=(1e-8,),
        n_folds=3,
        seed=0,
    )

    rank1 = transport.basis_for_rank(1)
    rank2 = transport.basis_for_rank(2)
    np.testing.assert_allclose(rank1, rank2[:1])
    np.testing.assert_allclose(
        orthogonal_residual(rank2, reachable.basis),
        0.0,
        atol=1e-10,
    )
    rank1_mse = np.mean(np.square(transport.predict_endpoint(deltas, 1) - endpoints))
    rank2_mse = np.mean(np.square(transport.predict_endpoint(deltas, 2) - endpoints))
    assert rank2_mse < rank1_mse * 1e-6
    assert transport.metrics()["inside_reachable_basis"] is True


def test_scalar_endpoint_caps_transport_at_one_identified_dimension():
    deltas, groups = _two_dimensional_deltas()
    reachable = fit_reachable_basis(
        deltas,
        groups,
        rank_cap=2,
        coverage_target=0.999,
        min_rank=2,
        n_folds=3,
        seed=0,
    )
    scalar_endpoint = deltas[:, 0] - 0.25 * deltas[:, 1]
    transport = fit_reduced_rank_transport(
        deltas,
        scalar_endpoint,
        reachable,
        groups=groups,
        max_rank=2,
        ridge_alphas=(1e-8,),
        n_folds=3,
        seed=0,
        endpoint_kind="first_token_baseline",
    )

    assert transport.max_rank == transport.identified_rank == 1
    assert transport.metrics()["scalar_endpoint_is_baseline_only"] is True
    with pytest.raises(ValueError, match="rank must be in"):
        transport.basis_for_rank(2)


def test_refusal_dim_is_unit_rank_one_and_uses_both_classes():
    states = np.array(
        [
            [2.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-2.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ]
    )
    refusal = fit_dim_refusal_basis(states, np.array([0, 0, 1, 1]))

    assert refusal.basis.shape == (1, 3)
    np.testing.assert_allclose(np.linalg.norm(refusal.basis[0]), 1.0)
    # Label 1 (refusal) lies at negative x, so refusal-minus-compliance must keep
    # the negative sign instead of applying the arbitrary SVD sign convention.
    np.testing.assert_allclose(refusal.basis[0], [-1.0, 0.0, 0.0])
    assert refusal.class_zero_count == refusal.class_one_count == 2


def test_natural_predictor_is_label_free_and_predicts_from_neutral_plus_severity():
    deltas_seed, groups = _two_dimensional_deltas()
    reachable = fit_reachable_basis(
        deltas_seed,
        groups,
        rank_cap=2,
        coverage_target=0.999,
        min_rank=2,
        n_folds=3,
        seed=0,
    )
    rng = np.random.default_rng(4)
    neutral_coordinates = rng.uniform(-1.0, 1.0, size=(len(groups), 2))
    neutral = reconstruct_from_basis(neutral_coordinates, reachable.basis)
    severity = np.resize(np.array([[-1.0], [1.0]]), (len(groups), 1))
    delta_coordinates = np.column_stack(
        (
            0.6 * neutral_coordinates[:, 0] + 0.8 * severity[:, 0],
            -0.4 * neutral_coordinates[:, 1] + 0.5 * severity[:, 0],
        )
    )
    deltas = reconstruct_from_basis(delta_coordinates, reachable.basis)

    signature = inspect.signature(fit_natural_predictor)
    assert not {"behavior", "endpoints", "refusal_labels"}.intersection(signature.parameters)
    predictor = fit_natural_predictor(
        neutral,
        severity,
        deltas,
        reachable,
        groups=groups,
        ridge_alphas=(1e-8,),
        n_folds=3,
        seed=0,
        clip_quantiles=(0.0, 1.0),
    )
    predicted_coordinates = predictor.predict_basis_coordinates(neutral, severity)
    predicted_delta = predictor.predict_delta(neutral, severity)

    np.testing.assert_allclose(predicted_coordinates, delta_coordinates, atol=1e-7)
    np.testing.assert_allclose(predicted_delta, deltas, atol=1e-7)
    assert predictor.cv_mse is not None and predictor.baseline_mse is not None
    assert predictor.cv_mse < predictor.baseline_mse
    assert predictor.relative_improvement is not None
    assert predictor.relative_improvement > 0.9
    assert predictor.metrics()["label_free"] is True
    assert predictor.metrics()["uses_transformed_state_at_prediction"] is False
