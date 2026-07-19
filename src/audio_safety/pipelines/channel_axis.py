"""CPU-only estimators for the Run 10 channel-invariance causal audit.

Torch-free so the statistics run and unit-test on a CPU-only environment (AGENTS.md).
The GPU driver captures paired clean/attack activations, calls these estimators on the
train/dev folds to freeze the channel subspace ``U`` and the refusal-DiM control, and
then applies frozen ``U`` at test time via
:class:`audio_safety.models.hooks.ProjectedTransportIntervention`.

Design references (frozen in the Run 10 direction doc):
- The channel axis is **manipulation-defined and outcome-blind**: a mean-anchored SVD of
  paired clean-attack differences (difference-in-means is ``u1``; residual SVD adds the
  remaining rows). Rank is frozen by train/dev subspace stability + held-out
  reconstruction, NEVER by refusal outcomes.
- The refusal-DiM control is a **refused-vs-complied** difference-in-means, RECOMPUTED on
  the current cohort (do NOT reuse the Run 5 pitch-cohort axis; see the direction doc).
"""

from __future__ import annotations

import numpy as np


def unit(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the L2-normalized vector (safe against zero norm)."""
    vector = np.asarray(vector, dtype=np.float64)
    return vector / (np.linalg.norm(vector) + eps)


def paired_differences(clean: np.ndarray, attack: np.ndarray) -> np.ndarray:
    """``clean - attack`` for row-aligned paired activation matrices ``(n, d)``."""
    clean = np.asarray(clean, dtype=np.float64)
    attack = np.asarray(attack, dtype=np.float64)
    if clean.shape != attack.shape:
        raise ValueError(f"clean/attack shape mismatch: {clean.shape} vs {attack.shape}")
    if clean.ndim != 2:
        raise ValueError(f"expected (n, d) paired matrices, got ndim={clean.ndim}")
    return clean - attack


def mean_anchored_basis(diffs: np.ndarray, rank: int) -> np.ndarray:
    """Orthonormal ``(rank, d)`` channel basis from paired differences.

    Row 0 = ``unit(mean(diffs))`` (the difference-in-means, i.e. the rank-1 nested
    baseline). Rows 1..rank-1 = top right-singular vectors of the mean-removed diffs,
    Gram-Schmidt-orthonormalized against the earlier rows. Purely manipulation-defined:
    no refusal outcome enters.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.ndim != 2:
        raise ValueError(f"diffs must be (n, d), got ndim={diffs.ndim}")
    if diffs.shape[0] < 1:
        raise ValueError("need at least one paired difference")
    if not np.all(np.isfinite(diffs)):
        raise ValueError("diffs contain non-finite values")
    d = diffs.shape[1]
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if rank > d:
        raise ValueError(f"rank {rank} cannot exceed d={d}")

    # u1 = the difference-in-means direction. Reject a degenerate (~zero) mean rather
    # than silently emitting an all-zero row (which would make B@B.T not a projector).
    mean_diff = diffs.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_diff))
    data_scale = float(np.sqrt((diffs**2).mean())) + 1e-12
    if not np.isfinite(mean_norm) or mean_norm <= 1e-8 * data_scale:
        raise ValueError("difference-in-means direction is undefined (mean paired difference ~0)")
    u1 = mean_diff / mean_norm
    basis = [u1]
    if rank > 1:
        residual = diffs - mean_diff[None, :]
        # Remove the u1 component so the SVD contributes genuinely new directions.
        residual = residual - (residual @ u1)[:, None] * u1[None, :]
        _, singular_values, vt = np.linalg.svd(residual, full_matrices=False)
        # Only admit numerically nonzero singular directions; SVD returns arbitrary
        # nullspace rows once the data rank is exhausted (e.g. identical diffs).
        sv_tol = (
            np.finfo(residual.dtype).eps
            * max(residual.shape)
            * (float(singular_values[0]) if singular_values.size else 0.0)
        )
        for singular_value, row in zip(singular_values, vt, strict=False):
            if len(basis) >= rank:
                break
            if singular_value <= sv_tol:
                continue
            candidate = row.astype(np.float64, copy=True)
            for existing in basis:
                candidate = candidate - (candidate @ existing) * existing
            norm = np.linalg.norm(candidate)
            if norm > 1e-8:
                basis.append(candidate / norm)
        if len(basis) != rank:
            raise ValueError(
                f"requested rank {rank} exceeds the numerical data rank of the paired "
                f"differences ({len(basis)} admissible directions)"
            )
    basis = np.stack(basis)
    if basis.shape != (rank, d):
        raise ValueError("basis has incorrect effective rank")
    if not np.allclose(basis @ basis.T, np.eye(rank), atol=1e-8, rtol=1e-8):
        raise ValueError("basis rows are not orthonormal")
    return basis


def project(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project rows of ``x`` onto the row space of orthonormal ``basis`` ``(k, d)``."""
    x = np.asarray(x, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if x.shape[-1] != basis.shape[-1]:
        raise ValueError(f"x dim {x.shape[-1]} != basis dim {basis.shape[-1]}")
    return (x @ basis.T) @ basis


def refusal_dim_direction(p2: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    """Refused-vs-complied difference-in-means (unit) at the readout position.

    ``labels``: 1 = refused, 0 = complied, anything else = ignored. Returns ``None`` if a
    class is missing. Mirrors ``scripts/phase_causal_patch.py::dim_dir`` but is intended
    to be recomputed on the CURRENT cohort — the Run 5 pitch-cohort axis must not be
    reused (Run 10 direction doc, RESOLVED note).
    """
    p2 = np.asarray(p2, dtype=np.float64)
    labels = np.asarray(labels)
    mask = np.isin(labels, [0, 1])
    if len(np.unique(labels[mask])) < 2:
        return None
    diff = p2[mask][labels[mask] == 1].mean(axis=0) - p2[mask][labels[mask] == 0].mean(axis=0)
    return unit(diff)


def largest_principal_angle(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Largest principal angle (radians) between two subspaces; 0 = identical span.

    Assumes orthonormal rows. Used to freeze rank by train/dev subspace STABILITY
    (outcome-blind), never by refusal outcomes.
    """
    basis_a = np.asarray(basis_a, dtype=np.float64)
    basis_b = np.asarray(basis_b, dtype=np.float64)
    singular = np.linalg.svd(basis_a @ basis_b.T, compute_uv=False)
    singular = np.clip(singular, -1.0, 1.0)
    return float(np.arccos(singular.min()))


def reconstruction_ratio(diffs: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of paired-difference energy captured by the subspace (use held-out diffs)."""
    diffs = np.asarray(diffs, dtype=np.float64)
    projected = project(diffs, basis)
    numerator = float((projected**2).sum())
    denominator = float((diffs**2).sum()) + 1e-12
    return numerator / denominator


def select_rank(
    train_diffs: np.ndarray,
    dev_diffs: np.ndarray,
    candidate_ranks: list[int],
    *,
    min_reconstruction: float,
    max_angle_rad: float,
) -> int:
    """Smallest rank whose dev reconstruction >= ``min_reconstruction`` AND whose
    train/dev subspaces agree within ``max_angle_rad`` RADIANS. Outcome-blind (no labels).

    Raises if no candidate qualifies — the run is then invalid/ambiguous and must NOT be
    silently forced through at the least-supported rank (Codex review, 2026-07-19).
    """
    if not candidate_ranks:
        raise ValueError("candidate_ranks must be non-empty")
    if not (0.0 <= max_angle_rad <= np.pi / 2 + 1e-9):
        raise ValueError("max_angle_rad must be in [0, pi/2] radians (arccos returns radians)")
    for rank in sorted(candidate_ranks):
        basis_train = mean_anchored_basis(train_diffs, rank)
        basis_dev = mean_anchored_basis(dev_diffs, rank)
        recon = reconstruction_ratio(dev_diffs, basis_train)
        angle = largest_principal_angle(basis_train, basis_dev)
        if recon >= min_reconstruction and angle <= max_angle_rad:
            return rank
    raise ValueError(
        "no candidate rank satisfies the reconstruction and stability thresholds; "
        "the channel subspace is not stable — treat the run as ambiguous"
    )
