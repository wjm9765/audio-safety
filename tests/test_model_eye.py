import numpy as np
import pytest

from audio_safety.evaluation.model_eye import (
    circular_time_shift,
    controlled_span_delta,
    model_eye_component,
    norm_match,
    resample_first_axis,
    temporal_modulation_split,
)


def test_temporal_split_reconstructs_delta_to_float_precision():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal((1, 4, 100)).astype(np.float32)
    slow, fast = temporal_modulation_split(delta, frame_rate_hz=100.0, cutoff_hz=8.0)
    np.testing.assert_allclose(slow + fast, delta, rtol=2e-6, atol=2e-7)


def test_time_shift_preserves_values_and_norm():
    delta = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    shifted = circular_time_shift(delta, 1)
    assert np.linalg.norm(shifted) == pytest.approx(np.linalg.norm(delta))
    np.testing.assert_array_equal(np.sort(shifted.ravel()), np.sort(delta.ravel()))


def test_resample_and_norm_match():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    resized = resample_first_axis(values, 7)
    assert resized.shape == (7, 4)
    matched = norm_match(np.ones_like(resized), resized)
    assert np.linalg.norm(matched) == pytest.approx(np.linalg.norm(resized), rel=1e-6)


def test_feature_wrong_item_is_resampled_and_norm_matched():
    delta = np.arange(40, dtype=np.float32).reshape(1, 4, 10) + 1
    wrong = np.ones((1, 4, 6), dtype=np.float32)
    component = model_eye_component(
        delta,
        "wrong_item",
        frame_rate_hz=100.0,
        cutoff_hz=8.0,
        timeshift_frames=2,
        wrong_item_delta=wrong,
    )
    assert component.shape == delta.shape
    assert np.linalg.norm(component) == pytest.approx(np.linalg.norm(delta), rel=1e-6)


def _component(delta, name, wrong=None, valid_length=7):
    return model_eye_component(
        delta,
        name,
        frame_rate_hz=100.0,
        cutoff_hz=8.0,
        timeshift_frames=2,
        wrong_item_delta=wrong,
        valid_length=valid_length,
        wrong_item_valid_length=None if wrong is None else 5,
    )


def _padded_delta():
    """A delta with a non-zero padded tail, as Whisper's global floor clamp gives."""

    delta = np.zeros((1, 3, 12), dtype=np.float32)
    delta[..., :7] = np.arange(21, dtype=np.float32).reshape(1, 3, 7) + 1
    delta[..., 7:] = -0.25  # global log-Mel floor shift outside the valid frames
    return delta


def test_temporal_and_structural_components_never_move_whisper_padding():
    delta = _padded_delta()
    wrong = np.zeros((1, 3, 12), dtype=np.float32)
    wrong[..., :5] = 2.0
    for name in ("temporal_fast", "temporal_slow", "time_shift", "wrong_item"):
        np.testing.assert_array_equal(_component(delta, name, wrong)[..., 7:], 0.0)


def test_all_component_is_the_exact_delta_so_dose_endpoints_reproduce_arms():
    delta = _padded_delta()
    # Zeroing the padded tail here would make source + 1.0 * component differ from
    # the physical target arm, which is exactly what the M1 endpoint gate checks.
    np.testing.assert_array_equal(_component(delta, "all"), delta)


def test_valid_only_identifies_the_speech_x_pad_interaction():
    """`all` splits into exactly two orthogonal supports, so the 2x2 is identified.

    Without a valid-frames-only arm there is no y(1,0) cell and the interaction
    I = y(1,1) - y(1,0) - y(0,1) + y(0,0) cannot be estimated, which is what
    licenses attributing an `all` effect to the speech region rather than to the
    Whisper global-floor pathway.
    """

    delta = _padded_delta()
    valid = _component(delta, "valid_only")
    pad = _component(delta, "pad_floor")
    np.testing.assert_array_equal(valid[..., 7:], 0.0)
    np.testing.assert_array_equal(valid[..., :7], delta[..., :7])
    np.testing.assert_array_equal(valid + pad, _component(delta, "all"))
    # The temporal split is a refinement of valid_only, not of all.
    slow = _component(delta, "temporal_slow")
    fast = _component(delta, "temporal_fast")
    np.testing.assert_allclose(slow + fast, valid, rtol=2e-6, atol=2e-7)


def test_pad_floor_completes_the_decomposition():
    delta = _padded_delta()
    slow = _component(delta, "temporal_slow")
    fast = _component(delta, "temporal_fast")
    pad = _component(delta, "pad_floor")
    np.testing.assert_array_equal(pad[..., :7], 0.0)
    np.testing.assert_array_equal(pad[..., 7:], delta[..., 7:])
    np.testing.assert_allclose(slow + fast + pad, delta, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("condition", ["wrong_item", "random_direction", "position_sham"])
def test_span_controls_are_norm_matched(condition):
    real = np.arange(20, dtype=np.float32).reshape(5, 4) + 1
    wrong = np.arange(12, dtype=np.float32).reshape(3, 4) + 2
    out = controlled_span_delta(real, condition, wrong_item_delta=wrong, seed=7)
    assert out.shape == real.shape
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(real), rel=1e-6)


def test_identity_span_control_is_zero():
    real = np.ones((3, 2), dtype=np.float32)
    np.testing.assert_array_equal(
        controlled_span_delta(real, "identity", wrong_item_delta=None, seed=0),
        np.zeros_like(real),
    )
