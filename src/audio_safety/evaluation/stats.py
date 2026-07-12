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
from math import comb

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


# --- Paired binary inference (Run 4 Stage A / T0 behavioral gate) -------------
#
# These are pure numpy/math so they run and are unit-tested without a GPU. The
# sampling unit is the semantic item (a matched harmful/benign pair rendered to
# both text and audio); paired tests condition on the item so per-item audio-vs-
# text flips drive the inference, not marginal rates.


def _binom_sf_at_least(k: int, n: int, p: float = 0.5) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, p). n is small (discordant pairs)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(sum(comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k, n + 1)))


def mcnemar_exact(audio_only: int, text_only: int) -> dict[str, float]:
    """Exact McNemar test on paired binary outcomes.

    ``audio_only`` = number of items where the audio arm succeeded (e.g. attack
    success / harmful compliance) and the text arm did not; ``text_only`` = the
    reverse. Only discordant pairs are informative. Under H0 each discordant pair
    is an independent fair coin, so:

    - ``p_two_sided`` tests audio != text (concordant-invariant),
    - ``p_one_sided_audio_gt_text`` = P(>= audio_only audio-favoring flips) tests
      the pre-registered direction audio > text.

    Returns a dict (also carries the raw discordant counts for auditability).
    """
    if audio_only < 0 or text_only < 0:
        raise ValueError("discordant counts must be non-negative")
    n_disc = audio_only + text_only
    if n_disc == 0:
        return {
            "n_discordant": 0,
            "audio_only": 0,
            "text_only": 0,
            "p_two_sided": 1.0,
            "p_one_sided_audio_gt_text": 1.0,
        }
    p_one = _binom_sf_at_least(audio_only, n_disc, 0.5)
    # Exact two-sided McNemar: sum of binomial masses <= the observed tail mass.
    k = min(audio_only, text_only)
    p_two = min(1.0, 2.0 * _binom_sf_at_least(max(audio_only, text_only), n_disc, 0.5))
    if audio_only == text_only:
        p_two = 1.0
    return {
        "n_discordant": int(n_disc),
        "audio_only": int(audio_only),
        "text_only": int(text_only),
        "p_two_sided": float(p_two),
        "p_one_sided_audio_gt_text": float(p_one),
        "concordant_note_k_min": int(k),
    }


def paired_risk_difference_ci(
    audio_outcomes: Sequence[int],
    text_outcomes: Sequence[int],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired risk difference RD = rate(audio) - rate(text) with a percentile
    bootstrap CI that resamples ITEMS (rows) with replacement, preserving the
    paired structure. Inputs are 0/1 arrays of equal length, aligned by item.
    """
    audio = np.asarray(audio_outcomes, dtype=float)
    text = np.asarray(text_outcomes, dtype=float)
    if audio.shape != text.shape or audio.ndim != 1:
        raise ValueError("audio and text outcomes must be equal-length 1-D arrays")
    n = audio.shape[0]
    if n == 0:
        raise ValueError("need at least one paired item for a risk difference")
    rd = float(audio.mean() - text.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for t in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[t] = audio[idx].mean() - text[idx].mean()
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "rd": rd,
        "rd_pp": 100.0 * rd,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_low_pp": 100.0 * float(lo),
        "ci_high_pp": 100.0 * float(hi),
        "n_items": int(n),
        "audio_rate": float(audio.mean()),
        "text_rate": float(text.mean()),
    }


def cohens_kappa(labels_a: Sequence[object], labels_b: Sequence[object]) -> float | None:
    """Cohen's kappa for two raters over categorical labels (judge agreement).

    Returns None when chance agreement ``pe`` is 1 (both raters constant on the
    same category): kappa is undefined there, so callers should fall back to the
    raw agreement rate rather than read a spurious 1.0.
    """
    a = list(labels_a)
    b = list(labels_b)
    if len(a) != len(b):
        raise ValueError("rater label lists must be equal length")
    n = len(a)
    if n == 0:
        raise ValueError("need at least one item for kappa")
    categories = sorted({*a, *b}, key=str)
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    conf = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        conf[index[x], index[y]] += 1
    po = float(np.trace(conf)) / n
    row = conf.sum(axis=1) / n
    col = conf.sum(axis=0) / n
    pe = float(row @ col)
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)
