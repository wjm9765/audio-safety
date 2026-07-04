"""Decision statistics for the drift probe (design.md §5).

Conventions:
- ``samples`` is a float array of shape (n_contents, n_families, k): the cone-projection
  profile of each paired drift vector d_f(c). Rows (contents) are the paired unit —
  permutation and bootstrap both resample/shuffle at the content level, preserving
  the repeated-measures structure.
- ``profiles`` maps family name -> mean profile vector P_f (length k).

All functions are pure numpy so they run (and are unit-tested) without a GPU.
"""

from collections.abc import Mapping, Sequence

import numpy as np

Profiles = Mapping[str, np.ndarray]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("cosine undefined for zero-norm profile")
    return float(a @ b) / denom


def family_profiles(samples: np.ndarray, families: Sequence[str]) -> dict[str, np.ndarray]:
    """Mean profile P_f = mean_c p_f(c) per family (design.md §5.1)."""
    if samples.ndim != 3 or samples.shape[1] != len(families):
        raise ValueError(f"expected (n_contents, {len(families)}, k), got {samples.shape}")
    return {f: samples[:, i, :].mean(axis=0) for i, f in enumerate(families)}


def mean_pairwise_cosine(profiles: Profiles) -> float:
    """Mean cosine over all unordered family pairs — the primary H0/H1 metric."""
    names = list(profiles)
    if len(names) < 2:
        raise ValueError("need at least two families")
    cosines = [
        _cosine(profiles[a], profiles[b]) for i, a in enumerate(names) for b in names[i + 1 :]
    ]
    return float(np.mean(cosines))


def pairwise_cosine_matrix(profiles: Profiles) -> tuple[list[str], np.ndarray]:
    """Full family x family cosine matrix (Fig 2 heatmap)."""
    names = list(profiles)
    mat = np.eye(len(names))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            mat[i, j] = mat[j, i] = _cosine(profiles[a], profiles[names[j]])
    return names, mat


def dominant_axes(profiles: Profiles) -> dict[str, int]:
    """argmax_j |P_f[j]| per family (design.md §5.2.2)."""
    return {f: int(np.argmax(np.abs(p))) for f, p in profiles.items()}


def dominant_axis_disagreement(profiles: Profiles) -> bool:
    """True if at least two families peak on different cone axes."""
    return len(set(dominant_axes(profiles).values())) > 1


def permutation_test(
    samples: np.ndarray,
    families: Sequence[str],
    n_permutations: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Permutation test for H1 (low mean pairwise cosine).

    Null model: family labels are exchangeable *within each content* (the paired
    design's exchangeability unit). Each permutation independently shuffles the
    family axis per content row, then recomputes the mean pairwise cosine of the
    family mean profiles.

    Returns (observed_mpc, p_value) with p = P(null <= observed): the probability
    of seeing this much family separation (low cosine) by chance. Uses the
    add-one (phipson-smyth) estimator so p is never exactly 0.

    CAVEAT (interpretation): the permutation null is FULL exchangeability of family
    labels. The design's H0 ("same direction, different magnitudes") is weaker —
    magnitude heterogeneity alone can yield a small p even when directions collapse.
    A small p therefore does NOT by itself refute H0; this is exactly why the
    pre-registered GO rule (design.md §0) requires BOTH mpc < threshold AND p <
    threshold. Never interpret p without the mpc effect size.
    """
    n_contents, n_families, _ = samples.shape
    if n_families != len(families):
        raise ValueError("families length must match samples.shape[1]")

    observed = mean_pairwise_cosine(family_profiles(samples, families))

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for t in range(n_permutations):
        permuted = np.empty_like(samples)
        for c in range(n_contents):
            permuted[c] = samples[c, rng.permutation(n_families), :]
        null[t] = mean_pairwise_cosine(family_profiles(permuted, families))

    p_value = (1.0 + float(np.sum(null <= observed))) / (n_permutations + 1.0)
    return observed, p_value


def bootstrap_cosine_ci(
    samples: np.ndarray,
    families: Sequence[str],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean pairwise cosine, resampling contents
    (rows) with replacement — respects the paired design."""
    n_contents = samples.shape[0]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_bootstrap)
    for t in range(n_bootstrap):
        idx = rng.integers(0, n_contents, size=n_contents)
        stats[t] = mean_pairwise_cosine(family_profiles(samples[idx], families))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
