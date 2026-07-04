"""Statistical machinery tests on synthetic data with known H0/H1 ground truth."""

import numpy as np
import pytest

from audio_safety.config.schema import DecisionConfig
from audio_safety.evaluation import (
    bootstrap_cosine_ci,
    decide,
    dominant_axis_disagreement,
    family_profiles,
    mean_pairwise_cosine,
    permutation_test,
)
from audio_safety.pipelines.cone import (
    gram_schmidt,
    principal_angles,
    project_onto_cone,
)
from audio_safety.pipelines.drift import drift_vectors, project_drifts

FAMILIES = ["plain", "nonspeech", "style", "perturbed"]
K = 6


def _synthetic(mode: str, n: int = 60, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """(n, 4, K) samples.
    h0            : all families identical direction AND magnitude (exchangeable)
    h0_scaled     : same direction, per-family magnitudes (design.md's H0 collapse)
    h1            : each family peaks on its own axis
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((n, len(FAMILIES), K))
    for i in range(len(FAMILIES)):
        direction = np.zeros(K)
        if mode == "h0":
            direction[0] = 1.5
        elif mode == "h0_scaled":
            direction[0] = 1.0 + 0.5 * i  # same axis, different magnitude
        else:
            direction[i] = 1.0  # distinct dominant axes
        samples[:, i, :] = direction + noise * rng.standard_normal((n, K))
    return samples


# --- profiles & cosine ---


def test_h0_high_cosine():
    profiles = family_profiles(_synthetic("h0"), FAMILIES)
    assert mean_pairwise_cosine(profiles) > 0.95
    assert not dominant_axis_disagreement(profiles)


def test_h1_low_cosine_and_axis_disagreement():
    profiles = family_profiles(_synthetic("h1"), FAMILIES)
    assert mean_pairwise_cosine(profiles) < 0.3
    assert dominant_axis_disagreement(profiles)


# --- permutation test ---


def test_permutation_significant_under_h1():
    observed, p = permutation_test(_synthetic("h1"), FAMILIES, n_permutations=500, seed=0)
    assert observed < 0.3
    assert p < 0.05


def test_permutation_not_significant_under_exchangeable_h0():
    _, p = permutation_test(_synthetic("h0"), FAMILIES, n_permutations=500, seed=0)
    assert p > 0.05


def test_scaled_collapse_still_nogo_despite_small_p():
    """Documented caveat (stats.permutation_test): under 'same direction, different
    magnitude' collapse, full-exchangeability permutation can reject (small p).
    The pre-registered decision rule must still return NO-GO via the mpc threshold —
    p alone must never drive the verdict."""
    s = _synthetic("h0_scaled")
    profiles = family_profiles(s, FAMILIES)
    mpc = mean_pairwise_cosine(profiles)
    _, p = permutation_test(s, FAMILIES, n_permutations=500, seed=0)
    assert mpc > 0.95  # directions collapsed...
    assert p < 0.05  # ...yet exchangeability is rejected (magnitude heterogeneity)
    result = decide(
        mpc, p, dominant_axis_disagreement(profiles), DecisionConfig(), causal_axes_validated=True
    )
    assert result.status == "NO-GO"


def test_permutation_deterministic():
    s = _synthetic("h1")
    assert permutation_test(s, FAMILIES, 200, seed=3) == permutation_test(s, FAMILIES, 200, seed=3)


# --- bootstrap ---


def test_bootstrap_ci_brackets_observed():
    s = _synthetic("h1")
    observed = mean_pairwise_cosine(family_profiles(s, FAMILIES))
    lo, hi = bootstrap_cosine_ci(s, FAMILIES, n_bootstrap=300, seed=0)
    assert lo <= observed <= hi
    assert hi - lo < 0.3  # tight for clean synthetic data


# --- decision rule (design.md §0) ---


def test_decision_go():
    result = decide(0.3, 0.001, True, DecisionConfig(), causal_axes_validated=True)
    assert result.status == "GO"


def test_decision_nogo_on_collapse():
    result = decide(0.9, 0.001, True, DecisionConfig(), causal_axes_validated=True)
    assert result.status == "NO-GO"


def test_decision_ambiguous_midrange():
    result = decide(0.7, 0.001, True, DecisionConfig(), causal_axes_validated=True)
    assert result.status == "AMBIGUOUS"


def test_no_go_without_causal_validation():
    """GO must never be granted if ablation hasn't been run (design.md §0 table)."""
    result = decide(0.3, 0.001, True, DecisionConfig(), causal_axes_validated=None)
    assert result.status == "AMBIGUOUS"
    assert any("causal" in r for r in result.reasons)


# --- cone linear algebra ---


def test_gram_schmidt_orthonormal():
    rng = np.random.default_rng(0)
    basis = gram_schmidt(rng.standard_normal((5, 32)))
    np.testing.assert_allclose(basis @ basis.T, np.eye(5), atol=1e-8)


def test_gram_schmidt_drops_dependent_rows():
    v = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    basis = gram_schmidt(v)
    assert basis.shape == (2, 2)


def test_principal_angles_identical_subspace():
    rng = np.random.default_rng(1)
    basis = gram_schmidt(rng.standard_normal((3, 16)))
    np.testing.assert_allclose(principal_angles(basis, basis), 0.0, atol=1e-6)


def test_projection_recovers_coordinates():
    rng = np.random.default_rng(2)
    basis = gram_schmidt(rng.standard_normal((4, 24)))
    coords = rng.standard_normal(4)
    vector = coords @ basis
    np.testing.assert_allclose(project_onto_cone(vector, basis), coords, atol=1e-8)


# --- paired drift ---


def test_drift_pairing_mismatch_raises():
    with pytest.raises(ValueError, match="pairing mismatch"):
        drift_vectors(np.zeros((10, 8)), np.zeros((9, 8)))


def test_project_drifts_shape():
    rng = np.random.default_rng(4)
    d, n = 24, 12
    basis = gram_schmidt(rng.standard_normal((K, d)))
    h_text = rng.standard_normal((n, d))
    h_by_family = {f: rng.standard_normal((n, d)) for f in FAMILIES}
    out = project_drifts(h_by_family, h_text, basis, FAMILIES)
    assert out.shape == (n, len(FAMILIES), K)
