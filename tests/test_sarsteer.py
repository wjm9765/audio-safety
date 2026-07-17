"""Unit tests for the faithful SARSteer defense core.

Pure-numpy tests run in the CPU-only base env. The steering-hook test is gated on
torch (gpu group) via importorskip.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_safety.pipelines.sarsteer import (
    build_sarsteer_vectors,
    load_sarsteer_vectors,
    orthogonal_complement,
    safe_subspace,
    save_sarsteer_vectors,
)


def test_orthogonal_complement_removes_span_and_keeps_orthogonal():
    d = 6
    # Basis spans the first two coordinate axes.
    basis = np.zeros((d, 2), dtype=np.float32)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0

    # A vector living entirely in the span collapses to ~0.
    in_span = np.array([3.0, -4.0, 0, 0, 0, 0], dtype=np.float32)
    assert np.linalg.norm(orthogonal_complement(in_span, basis)) < 1e-5

    # A vector orthogonal to the span is preserved (raw magnitude, no renorm).
    orthogonal = np.array([0, 0, 2.0, 5.0, -1.0, 0], dtype=np.float32)
    out = orthogonal_complement(orthogonal, basis)
    np.testing.assert_allclose(out, orthogonal, atol=1e-5)

    # A mixed vector keeps only the out-of-span coordinates.
    mixed = np.array([7.0, 9.0, 2.0, 0, 0, 0], dtype=np.float32)
    out = orthogonal_complement(mixed, basis)
    np.testing.assert_allclose(out, [0, 0, 2.0, 0, 0, 0], atol=1e-5)


def test_orthogonal_complement_is_orthogonal_to_basis():
    rng = np.random.default_rng(0)
    d, k = 12, 4
    raw = rng.standard_normal((d, k))
    basis, _ = np.linalg.qr(raw)  # orthonormal columns
    basis = basis[:, :k].astype(np.float32)
    v = rng.standard_normal(d).astype(np.float32)
    v_perp = orthogonal_complement(v, basis)
    # Residual has no component along any basis direction.
    np.testing.assert_allclose(basis.T @ v_perp, np.zeros(k), atol=1e-4)


def test_safe_subspace_recovers_dominant_direction_and_is_orthonormal():
    rng = np.random.default_rng(1)
    d = 8
    direction = np.zeros(d)
    direction[3] = 1.0
    # Samples vary strongly along axis 3, weakly elsewhere.
    scores = rng.standard_normal(200) * 10.0
    data = np.outer(scores, direction) + rng.standard_normal((200, d)) * 0.01
    basis = safe_subspace(data.astype(np.float32), n_pcs=1)
    assert basis.shape == (d, 1)
    # Top PC aligns with axis 3 (up to sign).
    assert abs(abs(float(basis[3, 0])) - 1.0) < 1e-2
    # Orthonormal columns.
    np.testing.assert_allclose(basis.T @ basis, np.eye(1), atol=1e-4)


def test_safe_subspace_caps_components_at_rank():
    # 3 centered samples span rank <= 2; asking for 10 PCs must not overreach.
    data = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]], dtype=np.float32)
    basis = safe_subspace(data, n_pcs=10)
    assert basis.shape[1] <= 2
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-4)


def test_build_sarsteer_vectors_ablates_benign_variance():
    d = 5
    # Benign speech varies along axis 0; refusal vector has an axis-0 component
    # that must be removed and an axis-2 component that must survive.
    benign = np.zeros((50, d), dtype=np.float32)
    benign[:, 0] = np.linspace(-5, 5, 50)
    refusal = {7: np.array([4.0, 0, 3.0, 0, 0], dtype=np.float32)}
    out = build_sarsteer_vectors(refusal, {7: benign}, n_pcs=1)
    assert set(out) == {7}
    # axis-0 (benign) component removed, axis-2 kept.
    assert abs(out[7][0]) < 1e-3
    assert abs(out[7][2] - 3.0) < 1e-3


def test_build_sarsteer_vectors_requires_shared_layer():
    with pytest.raises(ValueError):
        build_sarsteer_vectors(
            {1: np.ones(3, dtype=np.float32)},
            {2: np.ones((4, 3), dtype=np.float32)},
        )


def test_save_load_roundtrip(tmp_path):
    vectors = {
        3: np.array([1.0, 2.0, 3.0], dtype=np.float32),
        16: np.array([-1.0, 0.5, 0.0], dtype=np.float32),
    }
    path = tmp_path / "sarsteer_vectors.npz"
    save_sarsteer_vectors(path, vectors, {"alpha": 0.1, "n_pcs": 10})
    loaded = load_sarsteer_vectors(path)
    assert set(loaded) == set(vectors)
    for ell in vectors:
        np.testing.assert_allclose(loaded[ell], vectors[ell])


def test_multilayer_steering_adds_raw_vector_at_all_positions():
    torch = pytest.importorskip("torch")
    import types

    from audio_safety.models.hooks import MultiLayerAdditiveSteering

    d = 4

    class Identity(torch.nn.Module):
        def forward(self, x):  # noqa: D401 - trivial passthrough
            return x

    layers = torch.nn.ModuleList([Identity(), Identity(), Identity()])
    model = types.SimpleNamespace(language_model=types.SimpleNamespace(layers=layers))

    # A vector with norm != 1: no-normalize means the RAW vector is scaled by alpha.
    v1 = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)  # norm 2
    vectors = {1: v1}
    alpha = 0.1

    hidden = torch.zeros(1, 5, d)  # (batch, time=5, d)
    with MultiLayerAdditiveSteering(model, vectors=vectors, alpha=alpha, normalize=False) as steer:
        out0 = layers[0](hidden)  # unsteered layer -> unchanged
        out1 = layers[1](hidden)  # steered layer
        # simulate 3 KV-cache decode steps at the steered layer (length-1 slices)
        for _ in range(3):
            layers[1](torch.zeros(1, 1, d))

    assert torch.allclose(out0, hidden)
    expected = hidden + alpha * torch.tensor(v1)  # raw, not unit-normalized
    assert torch.allclose(out1, expected)
    # Applied at prefill + each decode step for the steered layer only.
    assert steer.applied_counts == {1: 4}


def test_multilayer_steering_normalize_flag_unit_scales():
    torch = pytest.importorskip("torch")
    import types

    from audio_safety.models.hooks import MultiLayerAdditiveSteering

    d = 4

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    layers = torch.nn.ModuleList([Identity()])
    model = types.SimpleNamespace(language_model=types.SimpleNamespace(layers=layers))
    v = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)  # norm 5
    hidden = torch.zeros(1, 2, d)
    with MultiLayerAdditiveSteering(model, vectors={0: v}, alpha=1.0, normalize=True):
        out = layers[0](hidden)
    unit = torch.tensor(v) / torch.linalg.vector_norm(torch.tensor(v))
    assert torch.allclose(out, hidden + unit, atol=1e-6)
