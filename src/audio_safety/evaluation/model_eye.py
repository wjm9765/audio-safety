"""Torch-free feature/component and norm-matched control helpers for Exp3."""

from __future__ import annotations

from typing import Literal

import numpy as np

ModelEyeComponent = Literal[
    "all",
    "temporal_fast",
    "temporal_slow",
    "time_shift",
    "wrong_item",
    "pad_floor",
    "valid_only",
]
PatchControl = Literal["real", "identity", "wrong_item", "random_direction", "position_sham"]


def temporal_modulation_split(
    delta: np.ndarray,
    *,
    frame_rate_hz: float,
    cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(slow, fast)`` whose sum exactly reconstructs ``delta``.

    The final axis is the Whisper feature-frame axis. ``slow`` keeps temporal
    modulation frequencies at or below ``cutoff_hz`` and ``fast`` is defined as
    the floating-point residual, rather than a second inverse FFT.
    """

    values = np.asarray(delta, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("delta must have a time axis with at least two frames")
    if frame_rate_hz <= 0.0 or not 0.0 < cutoff_hz < frame_rate_hz / 2.0:
        raise ValueError("cutoff_hz must lie strictly between 0 and Nyquist")
    spectrum = np.fft.rfft(values.astype(np.float64), axis=-1)
    frequencies = np.fft.rfftfreq(values.shape[-1], d=1.0 / frame_rate_hz)
    spectrum[..., frequencies > cutoff_hz] = 0.0
    slow = np.fft.irfft(spectrum, n=values.shape[-1], axis=-1).astype(np.float32)
    fast = values - slow
    return slow, fast


def circular_time_shift(delta: np.ndarray, frames: int) -> np.ndarray:
    """Destroy alignment while preserving the exact value multiset and norm."""

    values = np.asarray(delta, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("delta must have a time axis with at least two frames")
    if not isinstance(frames, int) or isinstance(frames, bool) or frames == 0:
        raise ValueError("frames must be a non-zero integer")
    return np.roll(values, shift=frames, axis=-1)


def resample_first_axis(values: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a ``(positions, dim...)`` displacement along positions."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("values must have a non-empty first axis")
    if target_length < 1:
        raise ValueError("target_length must be positive")
    if array.shape[0] == target_length:
        return array.copy()
    old_grid = np.linspace(0.0, 1.0, array.shape[0])
    new_grid = np.linspace(0.0, 1.0, target_length)
    flat = array.reshape(array.shape[0], -1)
    out = np.empty((target_length, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        out[:, column] = np.interp(new_grid, old_grid, flat[:, column])
    return out.reshape((target_length, *array.shape[1:]))


def resample_last_axis(values: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a feature displacement along its final time axis."""

    array = np.asarray(values, dtype=np.float32)
    moved = np.moveaxis(array, -1, 0)
    return np.moveaxis(resample_first_axis(moved, target_length), 0, -1)


def norm_match(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale ``candidate`` to the Frobenius/L2 norm of ``reference``."""

    candidate = np.asarray(candidate, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} != {reference.shape}")
    target_norm = float(np.linalg.norm(reference.reshape(-1)))
    candidate_norm = float(np.linalg.norm(candidate.reshape(-1)))
    if target_norm == 0.0:
        return np.zeros_like(candidate)
    if candidate_norm <= 1e-12:
        raise ValueError("cannot norm-match a zero candidate to a non-zero reference")
    return candidate * np.float32(target_norm / candidate_norm)


def model_eye_component(
    delta: np.ndarray,
    component: ModelEyeComponent,
    *,
    frame_rate_hz: float,
    cutoff_hz: float,
    timeshift_frames: int,
    wrong_item_delta: np.ndarray | None = None,
    valid_length: int | None = None,
    wrong_item_valid_length: int | None = None,
) -> np.ndarray:
    """Select a component of ``delta``; temporal ops touch only valid frames.

    ``all`` is the exact unmodified ``delta`` so that dose 0 and dose 1 reproduce
    the two physical arms bit-for-bit.  Whisper's log-Mel floor is
    ``max(log_spec, log_spec.max() - 8.0)``, a *global* clamp, so any edit that
    moves the peak level also shifts the zero-padded tail by that same amount;
    ``delta`` is therefore non-zero outside the valid frames.  ``pad_floor``
    isolates exactly that padded-tail offset, which keeps the decomposition
    complete: ``temporal_slow + temporal_fast + pad_floor == all``.
    """

    delta = np.asarray(delta, dtype=np.float32)
    if valid_length is None:
        valid_length = delta.shape[-1]
    if not 2 <= valid_length <= delta.shape[-1]:
        raise ValueError("valid_length must cover at least two frames within delta")
    core = delta[..., :valid_length]
    if component == "all":
        return delta.copy()
    if component == "pad_floor":
        output = np.zeros_like(delta)
        output[..., valid_length:] = delta[..., valid_length:]
        return output
    if component == "valid_only":
        # The speech-region arm of the 2x2 (speech x pad) decomposition:
        # all == valid_only + pad_floor, so the interaction
        # I = y(1,1) - y(1,0) - y(0,1) + y(0,0) is identified.
        output = np.zeros_like(delta)
        output[..., :valid_length] = core
        return output
    if component in {"temporal_slow", "temporal_fast"}:
        slow, fast = temporal_modulation_split(
            core, frame_rate_hz=frame_rate_hz, cutoff_hz=cutoff_hz
        )
        selected = slow if component == "temporal_slow" else fast
    elif component == "time_shift":
        selected = circular_time_shift(core, timeshift_frames)
    elif component == "wrong_item":
        if wrong_item_delta is None:
            raise ValueError("wrong_item component requires wrong_item_delta")
        wrong_item_delta = np.asarray(wrong_item_delta, dtype=np.float32)
        if wrong_item_valid_length is None:
            wrong_item_valid_length = wrong_item_delta.shape[-1]
        if not 2 <= wrong_item_valid_length <= wrong_item_delta.shape[-1]:
            raise ValueError("wrong_item_valid_length is outside the wrong-item delta")
        wrong_core = wrong_item_delta[..., :wrong_item_valid_length]
        selected = resample_last_axis(wrong_core, valid_length)
        if selected.shape != core.shape:
            raise ValueError(
                f"wrong-item feature shape {selected.shape} cannot match current core {core.shape}"
            )
        selected = norm_match(selected, core)
    else:
        raise ValueError(f"unknown model-eye component {component!r}")
    if selected.shape != core.shape:
        raise ValueError(
            f"model-eye component shape {selected.shape} cannot match valid core {core.shape}"
        )
    output = np.zeros_like(delta)
    output[..., :valid_length] = selected
    return output


def controlled_span_delta(
    real_delta: np.ndarray,
    condition: PatchControl,
    *,
    wrong_item_delta: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """Create a control displacement with the real displacement's exact norm."""

    real = np.asarray(real_delta, dtype=np.float32)
    if real.ndim != 2 or real.shape[0] < 1:
        raise ValueError("real_delta must have shape (positions, hidden_dim)")
    if condition == "real":
        return real.copy()
    if condition == "identity":
        return np.zeros_like(real)
    if condition == "position_sham":
        rng = np.random.default_rng(seed)
        return real[rng.permutation(real.shape[0])]
    if condition == "random_direction":
        rng = np.random.default_rng(seed)
        return norm_match(rng.standard_normal(real.shape).astype(np.float32), real)
    if condition != "wrong_item":
        raise ValueError(f"unknown patch condition {condition!r}")
    if wrong_item_delta is None:
        raise ValueError("wrong_item condition requires wrong_item_delta")
    wrong = resample_first_axis(wrong_item_delta, real.shape[0])
    if wrong.shape != real.shape:
        raise ValueError(f"wrong-item displacement shape {wrong.shape} != {real.shape}")
    return norm_match(wrong, real)
