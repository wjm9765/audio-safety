"""Refusal-cone construction (design.md §4): diff-in-means + PCA cross-check.

Pure linear algebra on pre-extracted activations — no model access here, so this
module stays torch-free and unit-testable. Causal ablation validation (§4.4) needs
model forwards and lives with the extraction pipeline.
"""

from collections.abc import Mapping

import numpy as np


def gram_schmidt(vectors: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Orthonormalize row vectors in order (first vector's direction is preserved,
    so b1 keeps its diff-in-means interpretation). Near-dependent vectors are
    dropped rather than kept as noise axes.

    vectors: (k, d) -> returns (k' <= k, d) orthonormal rows.
    """
    basis: list[np.ndarray] = []
    for v in vectors.astype(np.float64):
        w = v - sum((v @ b) * b for b in basis)
        norm = np.linalg.norm(w)
        if norm > eps:
            basis.append(w / norm)
    if not basis:
        raise ValueError("all vectors were linearly dependent")
    return np.stack(basis)


def diff_in_means_directions(
    harmful_by_category: Mapping[str, np.ndarray],
    benign: np.ndarray,
) -> np.ndarray:
    """Method A (design.md §4.3): per-category mean(H_cat) - mean(B), rows ordered by
    decreasing norm so b1 = strongest separation axis, then Gram-Schmidt."""
    benign_mean = benign.mean(axis=0)
    diffs = np.stack([h.mean(axis=0) - benign_mean for h in harmful_by_category.values()])
    order = np.argsort(-np.linalg.norm(diffs, axis=1))
    return gram_schmidt(diffs[order])


def pca_directions(harmful: np.ndarray, benign: np.ndarray, k: int) -> np.ndarray:
    """Method B (design.md §4.3): PCA over per-sample difference vectors
    {h_i^harm - mean(B)} -> top-k components as rows (unit norm)."""
    diffs = harmful - benign.mean(axis=0)
    centered = diffs - diffs.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:k]


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two subspaces given as
    orthonormal row bases — the Method A vs B agreement check (design.md §4.3)."""
    qa = np.linalg.qr(basis_a.T)[0]
    qb = np.linalg.qr(basis_b.T)[0]
    singular_values = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(singular_values, -1.0, 1.0))


def project_onto_cone(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Cone coordinates p = [<v, b1>, ..., <v, bk>] for row vector(s).

    vectors: (d,) or (n, d); basis: (k, d) orthonormal rows -> (k,) or (n, k).
    """
    return vectors @ basis.T


def off_subspace_residual_norm(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Norm of the component outside the cone (design.md §5.3 'off-subspace drift')."""
    coords = project_onto_cone(vectors, basis)
    reconstructed = coords @ basis
    return np.linalg.norm(vectors - reconstructed, axis=-1)
