"""CPU-only linear core for the exploratory COAST-R Stage-A experiment.

The module deliberately separates four fitted objects:

``B``
    An uncentred, outcome-free row basis of naturally observed audio-operator
    deltas.  Its rank is selected only by grouped held-out reconstruction.
``U``
    A nested, behavior-supervised basis learned *inside* ``B`` from a vector
    continuation endpoint.
``R``
    A descriptive difference-in-means refusal direction fit on a disjoint item
    role.  It is not used to construct ``B`` or the natural predictor.
``f``
    A label-free ridge predictor of natural ``B`` coordinates from the neutral
    state projected into ``B`` and declared severity features.  Its public fit
    API cannot receive behavior targets.

All bases use the project convention ``(rank, hidden_dim)``: directions are
rows.  No model or torch dependency is imported here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS = 1e-12


def _matrix(name: str, values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _targets(name: str, values: object, *, n_rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != n_rows or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape ({n_rows}, n_targets)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _group_array(groups: Sequence[object], *, n_rows: int) -> np.ndarray:
    array = np.asarray([str(value) for value in groups], dtype=object)
    if array.ndim != 1 or len(array) != n_rows:
        raise ValueError(f"groups must contain exactly {n_rows} row labels")
    if any(not value for value in array):
        raise ValueError("group labels must be non-empty")
    return array


def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale = np.asarray(values, dtype=np.float64).std(axis=0, ddof=0)
    return np.where(scale > _EPS, scale, 1.0)


def _canonicalize_rows(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=np.float64).copy()
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return result


def _top_right_singular_rows(
    matrix: np.ndarray, rank: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    max_rank = min(matrix.shape)
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")
    if rank == max_rank or max_rank <= 32:
        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        return singular_values[:rank], _canonicalize_rows(vt[:rank])

    from sklearn.utils.extmath import randomized_svd

    _, singular_values, vt = randomized_svd(
        matrix,
        n_components=rank,
        n_iter=5,
        random_state=int(seed),
    )
    return singular_values, _canonicalize_rows(vt)


def deterministic_group_folds(
    groups: Sequence[object], n_splits: int, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic item-grouped ``(train, heldout)`` row indices.

    Python's process-randomized ``hash`` is intentionally not used.  Sorted
    string group IDs are permuted by NumPy's seeded generator, then split once.
    Every group appears in exactly one held-out fold.
    """

    group_array = _group_array(groups, n_rows=len(groups))
    unique = np.asarray(sorted(set(group_array.tolist())), dtype=object)
    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2")
    if len(unique) < int(n_splits):
        raise ValueError(f"n_splits={n_splits} exceeds the number of unique groups={len(unique)}")
    order = np.random.default_rng(int(seed)).permutation(len(unique))
    parts = np.array_split(unique[order], int(n_splits))
    all_indices = np.arange(len(group_array), dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for part in parts:
        heldout_mask = np.isin(group_array, part)
        heldout = all_indices[heldout_mask]
        train = all_indices[~heldout_mask]
        if len(train) == 0 or len(heldout) == 0:
            raise ValueError("group split produced an empty train or held-out fold")
        folds.append((train, heldout))
    return folds


@dataclass(frozen=True)
class RoleAssignment:
    """One four-way-disjoint B/R/U/f assignment inside outer training."""

    rotation: int
    b_indices: np.ndarray
    r_indices: np.ndarray
    u_indices: np.ndarray
    f_indices: np.ndarray
    b_groups: tuple[str, ...]
    r_groups: tuple[str, ...]
    u_groups: tuple[str, ...]
    f_groups: tuple[str, ...]

    def metrics(self) -> dict[str, Any]:
        return {
            "rotation": int(self.rotation),
            "n_b_rows": int(len(self.b_indices)),
            "n_r_rows": int(len(self.r_indices)),
            "n_u_rows": int(len(self.u_indices)),
            "n_f_rows": int(len(self.f_indices)),
            "n_b_groups": int(len(self.b_groups)),
            "n_r_groups": int(len(self.r_groups)),
            "n_u_groups": int(len(self.u_groups)),
            "n_f_groups": int(len(self.f_groups)),
            "b_groups": list(self.b_groups),
            "r_groups": list(self.r_groups),
            "u_groups": list(self.u_groups),
            "f_groups": list(self.f_groups),
        }


def make_disjoint_role_rotations(
    train_indices: Sequence[int],
    groups: Sequence[object],
    seed: int = 0,
) -> tuple[RoleAssignment, RoleAssignment]:
    """Create disjoint B/R/U/f item roles and an R/U-swapped sensitivity fit.

    Rotation 0 assigns shuffled item quarters to ``B,R,U,f``.  Rotation 1 keeps
    the outcome-free B/f calibration roles fixed and swaps R with U.  Thus every
    assignment has four pairwise-disjoint item sets, while the second rotation
    directly tests sensitivity to which half learned refusal versus transport.
    """

    all_groups = _group_array(groups, n_rows=len(groups))
    indices = np.asarray(train_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("train_indices must be a non-empty 1-D sequence")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("train_indices contains duplicates")
    if indices.min() < 0 or indices.max() >= len(all_groups):
        raise IndexError("train_indices contains an out-of-range row")

    unique = np.asarray(sorted(set(all_groups[indices].tolist())), dtype=object)
    if len(unique) < 4:
        raise ValueError("at least four outer-train item groups are required")
    order = np.random.default_rng(int(seed)).permutation(len(unique))
    b_groups, r_groups, u_groups, f_groups = np.array_split(unique[order], 4)

    def rows_for(names: np.ndarray) -> np.ndarray:
        return indices[np.isin(all_groups[indices], names)]

    b = rows_for(b_groups)
    r = rows_for(r_groups)
    u = rows_for(u_groups)
    f = rows_for(f_groups)
    roles = (b, r, u, f)
    if any(len(role) == 0 for role in roles):
        raise RuntimeError("failed to construct four non-empty fit roles")
    if any(
        np.intersect1d(left, right).size
        for offset, left in enumerate(roles)
        for right in roles[offset + 1 :]
    ):
        raise RuntimeError("B/R/U/f roles are not pairwise disjoint")
    if set(np.concatenate(roles).tolist()) != set(indices.tolist()):
        raise RuntimeError("B/R/U/f roles do not cover the outer training fold")

    b_names = tuple(sorted(str(value) for value in b_groups.tolist()))
    r_names = tuple(sorted(str(value) for value in r_groups.tolist()))
    u_names = tuple(sorted(str(value) for value in u_groups.tolist()))
    f_names = tuple(sorted(str(value) for value in f_groups.tolist()))
    return (
        RoleAssignment(0, b, r, u, f, b_names, r_names, u_names, f_names),
        RoleAssignment(1, b, u, r, f, b_names, u_names, r_names, f_names),
    )


def _basis_matrix(basis: object) -> np.ndarray:
    rows = _matrix("basis", basis)
    gram = rows @ rows.T
    if not np.allclose(gram, np.eye(len(rows)), atol=1e-7, rtol=1e-6):
        raise ValueError("basis rows must be orthonormal")
    return rows


def basis_coordinates(vectors: object, basis: object) -> np.ndarray:
    """Return coordinates in an orthonormal row basis."""

    rows = _basis_matrix(basis)
    values = np.asarray(vectors, dtype=np.float64)
    one = values.ndim == 1
    if one:
        values = values[None, :]
    values = _matrix("vectors", values)
    if values.shape[1] != rows.shape[1]:
        raise ValueError("vectors and basis have different ambient dimensions")
    result = values @ rows.T
    return result[0] if one else result


def reconstruct_from_basis(coordinates: object, basis: object) -> np.ndarray:
    """Map row-basis coordinates back to the ambient residual space."""

    rows = _basis_matrix(basis)
    values = np.asarray(coordinates, dtype=np.float64)
    one = values.ndim == 1
    if one:
        values = values[None, :]
    values = _matrix("coordinates", values)
    if values.shape[1] != rows.shape[0]:
        raise ValueError("coordinate count does not match basis rank")
    result = values @ rows
    return result[0] if one else result


def project_onto_basis(vectors: object, basis: object) -> np.ndarray:
    """Orthogonally project ambient vectors onto a row basis."""

    return reconstruct_from_basis(basis_coordinates(vectors, basis), basis)


def orthogonal_residual(vectors: object, basis: object) -> np.ndarray:
    """Return the component orthogonal to a row basis."""

    values = np.asarray(vectors, dtype=np.float64)
    return values - project_onto_basis(values, basis)


def _validate_weights(sample_weight: object | None, n_rows: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (n_rows,):
        raise ValueError(f"sample_weight must have shape ({n_rows},)")
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("sample_weight must be finite, non-negative, and non-zero")
    return weights


@dataclass(frozen=True)
class ReachableBasis:
    """Outcome-free uncentred reachable basis ``B`` (rows are directions)."""

    basis: np.ndarray
    singular_values: np.ndarray
    selected_rank: int
    requested_rank_cap: int
    effective_rank_cap: int
    coverage_target: float
    rank_candidates: tuple[int, ...]
    cv_coverage: dict[int, float]
    coverage_met: bool
    cap_exhausted: bool
    n_samples: int
    ambient_dim: int
    seed: int

    def coordinates(self, deltas: object) -> np.ndarray:
        return basis_coordinates(deltas, self.basis)

    def project(self, deltas: object) -> np.ndarray:
        return project_onto_basis(deltas, self.basis)

    def metrics(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.n_samples),
            "ambient_dim": int(self.ambient_dim),
            "selected_rank": int(self.selected_rank),
            "requested_rank_cap": int(self.requested_rank_cap),
            "effective_rank_cap": int(self.effective_rank_cap),
            "coverage_target": float(self.coverage_target),
            "rank_candidates": [int(rank) for rank in self.rank_candidates],
            "cv_coverage": {
                str(rank): float(value) for rank, value in sorted(self.cv_coverage.items())
            },
            "coverage_met": bool(self.coverage_met),
            "cap_exhausted": bool(self.cap_exhausted),
            "singular_values": self.singular_values.tolist(),
            "seed": int(self.seed),
            "centered": False,
            "label_free": True,
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        ranks = np.asarray(sorted(self.cv_coverage), dtype=np.int64)
        return {
            "basis": np.asarray(self.basis, dtype=np.float64),
            "singular_values": np.asarray(self.singular_values, dtype=np.float64),
            "cv_coverage_ranks": ranks,
            "cv_coverage_values": np.asarray(
                [self.cv_coverage[int(rank)] for rank in ranks], dtype=np.float64
            ),
            "rank_candidates": np.asarray(self.rank_candidates, dtype=np.int64),
            "selected_rank": np.asarray([self.selected_rank], dtype=np.int64),
        }


def fit_reachable_basis(
    deltas: object,
    groups: Sequence[object],
    *,
    rank_cap: int,
    coverage_target: float,
    min_rank: int = 1,
    n_folds: int = 3,
    seed: int = 0,
    sample_weight: object | None = None,
    rank_candidates: Sequence[int] | None = None,
) -> ReachableBasis:
    """Fit label-free uncentred ``B`` and choose rank by grouped reconstruction.

    This function intentionally has no behavior/outcome argument.  Validation
    coverage is captured energy around the natural zero-delta origin, not PCA
    variance around a fitted mean.
    """

    matrix = _matrix("deltas", deltas)
    group_array = _group_array(groups, n_rows=len(matrix))
    weights = _validate_weights(sample_weight, len(matrix))
    if not 0.0 <= float(coverage_target) <= 1.0:
        raise ValueError("coverage_target must be in [0, 1]")
    if int(rank_cap) < 1 or int(min_rank) < 1:
        raise ValueError("rank_cap and min_rank must be positive")

    effective_cap = min(int(rank_cap), matrix.shape[0], matrix.shape[1])
    if int(min_rank) > effective_cap:
        raise ValueError(f"min_rank={min_rank} exceeds feasible rank cap={effective_cap}")
    folds = deterministic_group_folds(group_array, int(n_folds), seed=int(seed))
    captured = np.zeros(effective_cap, dtype=np.float64)
    total = np.zeros(effective_cap, dtype=np.float64)
    counts = np.zeros(effective_cap, dtype=np.int64)
    for fold, (train, heldout) in enumerate(folds):
        fold_cap = min(effective_cap, len(train), matrix.shape[1])
        weighted_train = matrix[train] * np.sqrt(weights[train, None])
        _, fold_basis = _top_right_singular_rows(weighted_train, fold_cap, seed=int(seed) + fold)
        heldout_energy = float(np.sum(weights[heldout, None] * np.square(matrix[heldout])))
        if heldout_energy <= _EPS:
            raise ValueError("a held-out fold has zero delta energy")
        scores = matrix[heldout] @ fold_basis.T
        per_component = np.sum(weights[heldout, None] * np.square(scores), axis=0)
        cumulative = np.cumsum(per_component)
        captured[:fold_cap] += cumulative
        total[:fold_cap] += heldout_energy
        counts[:fold_cap] += 1

    coverage = {
        rank: float(captured[rank - 1] / total[rank - 1])
        for rank in range(1, effective_cap + 1)
        if counts[rank - 1] == len(folds) and total[rank - 1] > _EPS
    }
    if rank_candidates is None:
        declared = tuple(range(int(min_rank), effective_cap + 1))
    else:
        declared = tuple(
            sorted(
                {
                    int(rank)
                    for rank in rank_candidates
                    if int(min_rank) <= int(rank) <= effective_cap
                }
            )
        )
        if not declared:
            raise ValueError("rank_candidates has no feasible rank at or above min_rank")
    eligible = [rank for rank in declared if rank in coverage]
    if not eligible:
        raise ValueError("no candidate reachable rank was estimable in every fold")
    passing = [rank for rank in eligible if coverage[rank] >= float(coverage_target)]
    coverage_met = bool(passing)
    selected = min(passing) if passing else max(eligible)

    weighted_full = matrix * np.sqrt(weights[:, None])
    singular_values, full_basis = _top_right_singular_rows(
        weighted_full, effective_cap, seed=int(seed)
    )
    basis = full_basis[:selected]
    return ReachableBasis(
        basis=basis,
        singular_values=singular_values,
        selected_rank=int(selected),
        requested_rank_cap=int(rank_cap),
        effective_rank_cap=int(effective_cap),
        coverage_target=float(coverage_target),
        rank_candidates=declared,
        cv_coverage=coverage,
        coverage_met=coverage_met,
        cap_exhausted=bool(not coverage_met and selected == max(eligible)),
        n_samples=int(len(matrix)),
        ambient_dim=int(matrix.shape[1]),
        seed=int(seed),
    )


def _grouped_mse(actual: np.ndarray, predicted: np.ndarray, groups: np.ndarray) -> float:
    values = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        values.append(float(np.mean(np.square(actual[mask] - predicted[mask]))))
    return float(np.mean(values))


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=float(alpha), solver="svd", fit_intercept=True)
    model.fit(x, y)
    coefficient = np.asarray(model.coef_, dtype=np.float64)
    if coefficient.ndim == 1:
        coefficient = coefficient[None, :]
    intercept = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
    return coefficient.T, intercept


def _ridge_predict(x: np.ndarray, coefficient: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return x @ coefficient + intercept[None, :]


def _select_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    alphas: tuple[float, ...],
    *,
    n_folds: int,
    seed: int,
) -> tuple[float, dict[float, float]]:
    candidates = tuple(sorted({float(value) for value in alphas}))
    if not candidates or any(value <= 0.0 for value in candidates):
        raise ValueError("ridge_alphas must contain positive values")
    if groups is None or len(set(groups.tolist())) < 2 or int(n_folds) < 2:
        return candidates[0], {}
    split_count = min(int(n_folds), len(set(groups.tolist())))
    folds = deterministic_group_folds(groups, split_count, seed=int(seed))
    losses: dict[float, float] = {}
    for alpha in candidates:
        fold_losses = []
        for train, heldout in folds:
            x_mean = x[train].mean(axis=0)
            x_scale = _safe_scale(x[train])
            y_mean = y[train].mean(axis=0)
            y_scale = _safe_scale(y[train])
            train_x = (x[train] - x_mean) / x_scale
            train_y = (y[train] - y_mean) / y_scale
            coefficient, intercept = _fit_ridge(train_x, train_y, alpha)
            heldout_x = (x[heldout] - x_mean) / x_scale
            predicted = _ridge_predict(heldout_x, coefficient, intercept) * y_scale + y_mean
            fold_losses.append(_grouped_mse(y[heldout], predicted, groups[heldout]))
        losses[alpha] = float(np.mean(fold_losses))
    chosen = min(candidates, key=lambda value: (losses[value], value))
    return chosen, losses


@dataclass(frozen=True)
class TransportSubspace:
    """Nested behavior-relevant ``U`` whose rows remain inside reachable ``B``."""

    basis: np.ndarray
    coordinate_directions: np.ndarray
    singular_values: np.ndarray
    identified_rank: int
    chosen_alpha: float
    cv_mse_by_alpha: dict[float, float]
    endpoint_dim: int
    endpoint_kind: str
    readout_coefficients: tuple[np.ndarray, ...]
    readout_intercepts: tuple[np.ndarray, ...]
    train_mse_by_rank: dict[int, float]

    @property
    def max_rank(self) -> int:
        return int(len(self.basis))

    def basis_for_rank(self, rank: int) -> np.ndarray:
        value = int(rank)
        if not 1 <= value <= self.max_rank:
            raise ValueError(f"rank must be in [1, {self.max_rank}], got {value}")
        return self.basis[:value]

    def coordinates(self, deltas: object, rank: int | None = None) -> np.ndarray:
        value = self.max_rank if rank is None else int(rank)
        return basis_coordinates(deltas, self.basis_for_rank(value))

    def predict_endpoint(self, deltas: object, rank: int) -> np.ndarray:
        value = int(rank)
        scores = np.asarray(self.coordinates(deltas, value), dtype=np.float64)
        one = scores.ndim == 1
        if one:
            scores = scores[None, :]
        predicted = _ridge_predict(
            scores,
            self.readout_coefficients[value - 1],
            self.readout_intercepts[value - 1],
        )
        return predicted[0] if one else predicted

    def metrics(self) -> dict[str, Any]:
        return {
            "max_rank": self.max_rank,
            "identified_rank": int(self.identified_rank),
            "endpoint_dim": int(self.endpoint_dim),
            "endpoint_kind": self.endpoint_kind,
            "chosen_alpha": float(self.chosen_alpha),
            "cv_mse_by_alpha": {
                str(alpha): float(value) for alpha, value in sorted(self.cv_mse_by_alpha.items())
            },
            "singular_values": self.singular_values.tolist(),
            "train_mse_by_rank": {
                str(rank): float(value) for rank, value in sorted(self.train_mse_by_rank.items())
            },
            "nested_prefixes": True,
            "inside_reachable_basis": True,
            "scalar_endpoint_is_baseline_only": bool(self.endpoint_dim == 1),
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        max_rank = self.max_rank
        coefficients = np.zeros((max_rank, max_rank, self.endpoint_dim), dtype=np.float64)
        intercepts = np.zeros((max_rank, self.endpoint_dim), dtype=np.float64)
        for offset, (coefficient, intercept) in enumerate(
            zip(self.readout_coefficients, self.readout_intercepts, strict=True)
        ):
            coefficients[offset, : offset + 1] = coefficient
            intercepts[offset] = intercept
        return {
            "basis": np.asarray(self.basis, dtype=np.float64),
            "coordinate_directions": np.asarray(self.coordinate_directions, dtype=np.float64),
            "singular_values": np.asarray(self.singular_values, dtype=np.float64),
            "identified_rank": np.asarray([self.identified_rank], dtype=np.int64),
            "chosen_alpha": np.asarray([self.chosen_alpha], dtype=np.float64),
            "readout_coefficients": coefficients,
            "readout_intercepts": intercepts,
        }


def fit_reduced_rank_transport(
    deltas: object,
    endpoints: object,
    reachable: ReachableBasis,
    *,
    groups: Sequence[object] | None = None,
    max_rank: int = 4,
    ridge_alphas: tuple[float, ...] = (1.0,),
    n_folds: int = 3,
    seed: int = 0,
    endpoint_kind: str | None = None,
) -> TransportSubspace:
    """Fit a nested reduced-rank ridge transport basis inside frozen ``B``.

    ``endpoints`` may be ``(n,)`` only for the explicitly labelled first-token
    baseline.  The primary Run-7 contract is a vector curve ``(n, token_steps)``.
    """

    matrix = _matrix("deltas", deltas)
    if matrix.shape[1] != reachable.ambient_dim:
        raise ValueError("delta dimension does not match the reachable basis")
    target = _targets("endpoints", endpoints, n_rows=len(matrix))
    group_array = None if groups is None else _group_array(groups, n_rows=len(matrix))
    coordinates = reachable.coordinates(matrix)
    requested_rank = min(int(max_rank), coordinates.shape[1], target.shape[1])
    if requested_rank < 1:
        raise ValueError("max_rank must be positive")

    chosen_alpha, cv_losses = _select_ridge_alpha(
        coordinates,
        target,
        group_array,
        tuple(ridge_alphas),
        n_folds=int(n_folds),
        seed=int(seed),
    )
    x_mean = coordinates.mean(axis=0)
    x_scale = _safe_scale(coordinates)
    y_mean = target.mean(axis=0)
    y_scale = _safe_scale(target)
    standardized_x = (coordinates - x_mean) / x_scale
    standardized_y = (target - y_mean) / y_scale
    coefficient_std, _ = _fit_ridge(standardized_x, standardized_y, chosen_alpha)
    coefficient_raw = coefficient_std / x_scale[:, None]
    left, singular_values, _ = np.linalg.svd(coefficient_raw, full_matrices=False)
    if not singular_values.size or singular_values[0] <= _EPS:
        raise ValueError("ridge fit found no behavior-associated reachable direction")
    identified = int(np.sum(singular_values > max(_EPS, singular_values[0] * 1e-10)))
    feasible_rank = min(requested_rank, identified)
    directions = left[:, :feasible_rank].T
    basis = directions @ reachable.basis
    # Canonicalise U and its corresponding q-space direction together.
    for offset, row in enumerate(basis):
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            basis[offset] *= -1.0
            directions[offset] *= -1.0
    readout_coefficients: list[np.ndarray] = []
    readout_intercepts: list[np.ndarray] = []
    train_mse: dict[int, float] = {}
    for rank in range(1, feasible_rank + 1):
        scores = matrix @ basis[:rank].T
        coefficient, intercept = _fit_ridge(scores, target, chosen_alpha)
        prediction = _ridge_predict(scores, coefficient, intercept)
        readout_coefficients.append(coefficient)
        readout_intercepts.append(intercept)
        train_mse[rank] = float(np.mean(np.square(prediction - target)))

    kind = endpoint_kind or (
        "first_token_baseline" if target.shape[1] == 1 else "continuation_curve"
    )
    return TransportSubspace(
        basis=np.asarray(basis, dtype=np.float64),
        coordinate_directions=np.asarray(directions, dtype=np.float64),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        identified_rank=identified,
        chosen_alpha=float(chosen_alpha),
        cv_mse_by_alpha=cv_losses,
        endpoint_dim=int(target.shape[1]),
        endpoint_kind=str(kind),
        readout_coefficients=tuple(readout_coefficients),
        readout_intercepts=tuple(readout_intercepts),
        train_mse_by_rank=train_mse,
    )


@dataclass(frozen=True)
class DimRefusalBasis:
    """Descriptive harmful-only refusal DIM direction ``R``."""

    basis: np.ndarray
    class_zero_count: int
    class_one_count: int
    mean_gap_norm: float

    def metrics(self) -> dict[str, Any]:
        return {
            "rank": 1,
            "class_zero_count": int(self.class_zero_count),
            "class_one_count": int(self.class_one_count),
            "mean_gap_norm": float(self.mean_gap_norm),
            "descriptive_only": True,
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return {
            "basis": np.asarray(self.basis, dtype=np.float64),
            "mean_gap_norm": np.asarray([self.mean_gap_norm], dtype=np.float64),
        }


def fit_dim_refusal_basis(
    states: object, refusal_labels: object, *, rank: int = 1
) -> DimRefusalBasis:
    """Fit ``mean(refusal)-mean(compliance)``; labels must be binary 0/1."""

    if int(rank) != 1:
        raise ValueError("difference-in-means identifies exactly one descriptive direction")
    matrix = _matrix("states", states)
    labels = np.asarray(refusal_labels)
    if labels.shape != (len(matrix),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("refusal_labels must be a binary vector aligned with states")
    zero = matrix[labels == 0]
    one = matrix[labels == 1]
    if len(zero) == 0 or len(one) == 0:
        raise ValueError("both refusal classes are required for DIM")
    difference = one.mean(axis=0) - zero.mean(axis=0)
    norm = float(np.linalg.norm(difference))
    if norm <= _EPS:
        raise ValueError("refusal class means are identical")
    # Preserve the scientifically meaningful refusal-minus-compliance sign.
    # Unlike PCA/SVD axes, a DIM direction is not sign-ambiguous once labels are
    # declared (1=refusal, 0=compliance).
    basis = (difference / norm)[None, :]
    return DimRefusalBasis(basis, len(zero), len(one), norm)


def _severity_mean_prediction(
    train_severity: np.ndarray,
    train_target: np.ndarray,
    test_severity: np.ndarray,
) -> np.ndarray:
    global_mean = train_target.mean(axis=0)
    means: dict[tuple[float, ...], np.ndarray] = {}
    rounded_train = np.round(train_severity, 12)
    for row in rounded_train:
        key = tuple(float(value) for value in row)
        if key not in means:
            mask = np.all(rounded_train == row, axis=1)
            means[key] = train_target[mask].mean(axis=0)
    return np.stack(
        [
            means.get(tuple(float(value) for value in np.round(row, 12)), global_mean)
            for row in test_severity
        ]
    )


@dataclass(frozen=True)
class NaturalCoordinatePredictor:
    """Label-free predictor ``f(neutral @ B.T, severity) -> delta @ B.T``."""

    reachable_basis: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray
    chosen_alpha: float
    cv_mse: float | None
    baseline_mse: float | None
    relative_improvement: float | None
    cv_mse_by_alpha: dict[float, float]
    neutral_feature_dim: int
    severity_feature_dim: int

    def _features(self, neutral_states: object, severity_features: object) -> np.ndarray:
        neutral = _matrix("neutral_states", neutral_states)
        severity = _matrix("severity_features", severity_features)
        if len(neutral) != len(severity):
            raise ValueError("neutral_states and severity_features must align")
        if neutral.shape[1] != self.reachable_basis.shape[1]:
            raise ValueError("neutral state dimension does not match reachable basis")
        if severity.shape[1] != self.severity_feature_dim:
            raise ValueError(
                f"expected {self.severity_feature_dim} severity features, got {severity.shape[1]}"
            )
        neutral_coordinates = basis_coordinates(neutral, self.reachable_basis)
        return np.column_stack((neutral_coordinates, severity))

    def predict_basis_coordinates(
        self, neutral_states: object, severity_features: object
    ) -> np.ndarray:
        features = self._features(neutral_states, severity_features)
        standardized = (features - self.feature_mean) / self.feature_scale
        prediction = _ridge_predict(standardized, self.coefficient, self.intercept)
        return np.clip(prediction, self.clip_low, self.clip_high)

    def predict_delta(self, neutral_states: object, severity_features: object) -> np.ndarray:
        coordinates = self.predict_basis_coordinates(neutral_states, severity_features)
        return reconstruct_from_basis(coordinates, self.reachable_basis)

    def predict_transport_delta(
        self,
        neutral_states: object,
        severity_features: object,
        transport: TransportSubspace,
        rank: int,
    ) -> np.ndarray:
        predicted = self.predict_delta(neutral_states, severity_features)
        return project_onto_basis(predicted, transport.basis_for_rank(int(rank)))

    def metrics(self) -> dict[str, Any]:
        return {
            "chosen_alpha": float(self.chosen_alpha),
            "cv_mse": None if self.cv_mse is None else float(self.cv_mse),
            "baseline_mse": (None if self.baseline_mse is None else float(self.baseline_mse)),
            "relative_improvement": (
                None if self.relative_improvement is None else float(self.relative_improvement)
            ),
            "cv_relative_improvement": (
                None if self.relative_improvement is None else float(self.relative_improvement)
            ),
            "cv_mse_by_alpha": {
                str(alpha): float(value) for alpha, value in sorted(self.cv_mse_by_alpha.items())
            },
            "neutral_feature_dim": int(self.neutral_feature_dim),
            "severity_feature_dim": int(self.severity_feature_dim),
            "target_dim": int(self.reachable_basis.shape[0]),
            "label_free": True,
            "uses_transformed_state_at_prediction": False,
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return {
            "reachable_basis": np.asarray(self.reachable_basis, dtype=np.float64),
            "feature_mean": np.asarray(self.feature_mean, dtype=np.float64),
            "feature_scale": np.asarray(self.feature_scale, dtype=np.float64),
            "coefficient": np.asarray(self.coefficient, dtype=np.float64),
            "intercept": np.asarray(self.intercept, dtype=np.float64),
            "clip_low": np.asarray(self.clip_low, dtype=np.float64),
            "clip_high": np.asarray(self.clip_high, dtype=np.float64),
            "chosen_alpha": np.asarray([self.chosen_alpha], dtype=np.float64),
        }


def fit_natural_predictor(
    neutral_states: object,
    severity_features: object,
    deltas: object,
    reachable: ReachableBasis,
    *,
    groups: Sequence[object] | None = None,
    ridge_alphas: tuple[float, ...] = (1.0,),
    n_folds: int = 3,
    seed: int = 0,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> NaturalCoordinatePredictor:
    """Fit the label-free natural-coordinate predictor ``f``.

    The only targets are observed activation deltas projected into frozen ``B``.
    The signature intentionally has no behavior labels, endpoints, refusal
    scores, transformed test states, or gradients.
    """

    neutral = _matrix("neutral_states", neutral_states)
    severity = _matrix("severity_features", severity_features)
    delta = _matrix("deltas", deltas)
    if len(neutral) != len(severity) or len(neutral) != len(delta):
        raise ValueError("neutral_states, severity_features, and deltas must align")
    if neutral.shape[1] != reachable.ambient_dim or delta.shape[1] != reachable.ambient_dim:
        raise ValueError("state/delta dimension does not match reachable basis")
    group_array = None if groups is None else _group_array(groups, n_rows=len(delta))
    neutral_coordinates = reachable.coordinates(neutral)
    target_coordinates = reachable.coordinates(delta)
    features = np.column_stack((neutral_coordinates, severity))
    low_q, high_q = (float(clip_quantiles[0]), float(clip_quantiles[1]))
    if not 0.0 <= low_q < high_q <= 1.0:
        raise ValueError("clip_quantiles must satisfy 0 <= low < high <= 1")

    chosen_alpha, cv_losses = _select_ridge_alpha(
        features,
        target_coordinates,
        group_array,
        tuple(ridge_alphas),
        n_folds=int(n_folds),
        seed=int(seed),
    )

    cv_mse: float | None = None
    baseline_mse: float | None = None
    relative: float | None = None
    if group_array is not None and len(set(group_array.tolist())) >= 2 and int(n_folds) >= 2:
        folds = deterministic_group_folds(
            group_array,
            min(int(n_folds), len(set(group_array.tolist()))),
            seed=int(seed),
        )
        predictions = np.full_like(target_coordinates, np.nan)
        baselines = np.full_like(target_coordinates, np.nan)
        for train, heldout in folds:
            mean = features[train].mean(axis=0)
            scale = _safe_scale(features[train])
            coefficient, intercept = _fit_ridge(
                (features[train] - mean) / scale,
                target_coordinates[train],
                chosen_alpha,
            )
            predictions[heldout] = _ridge_predict(
                (features[heldout] - mean) / scale, coefficient, intercept
            )
            baselines[heldout] = _severity_mean_prediction(
                severity[train], target_coordinates[train], severity[heldout]
            )
        cv_mse = _grouped_mse(target_coordinates, predictions, group_array)
        baseline_mse = _grouped_mse(target_coordinates, baselines, group_array)
        relative = float((baseline_mse - cv_mse) / baseline_mse) if baseline_mse > _EPS else None

    feature_mean = features.mean(axis=0)
    feature_scale = _safe_scale(features)
    coefficient, intercept = _fit_ridge(
        (features - feature_mean) / feature_scale,
        target_coordinates,
        chosen_alpha,
    )
    clip_low = np.quantile(target_coordinates, low_q, axis=0)
    clip_high = np.quantile(target_coordinates, high_q, axis=0)
    return NaturalCoordinatePredictor(
        reachable_basis=np.asarray(reachable.basis, dtype=np.float64),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficient=coefficient,
        intercept=intercept,
        clip_low=clip_low,
        clip_high=clip_high,
        chosen_alpha=float(chosen_alpha),
        cv_mse=cv_mse,
        baseline_mse=baseline_mse,
        relative_improvement=relative,
        cv_mse_by_alpha=cv_losses,
        neutral_feature_dim=int(neutral_coordinates.shape[1]),
        severity_feature_dim=int(severity.shape[1]),
    )


def fit_coast_r_stage_a(
    arrays: dict[str, np.ndarray],
    cells: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    config: Any,
) -> dict[str, Any]:
    """Compatibility entry point for the project Stage-A orchestration.

    The pipeline owns artifact-to-row assembly; all statistical fits it composes
    are the CPU-only primitives above.  Keeping this thin adapter also lets tests
    invoke the exact public contract requested by the CLI without duplicating
    waveform/archive parsing in the evaluation layer.
    """

    from audio_safety.pipelines.coast_r import _run_crossfit_fit

    return _run_crossfit_fit(arrays, cells, score_rows, config)


__all__ = [
    "DimRefusalBasis",
    "NaturalCoordinatePredictor",
    "ReachableBasis",
    "RoleAssignment",
    "TransportSubspace",
    "basis_coordinates",
    "deterministic_group_folds",
    "fit_coast_r_stage_a",
    "fit_dim_refusal_basis",
    "fit_natural_predictor",
    "fit_reachable_basis",
    "fit_reduced_rank_transport",
    "make_disjoint_role_rotations",
    "orthogonal_residual",
    "project_onto_basis",
    "reconstruct_from_basis",
]
