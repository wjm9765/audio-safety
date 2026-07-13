"""Deterministic waveform-level acoustic manipulations for attack families.

These are the signal/acoustic building blocks that the text-jailbreak wrapper
path (``attacks.py``) does not cover: additive interference overlays
(AIA-style), Gaussian/ambient noise at a target SNR, universal additive
perturbations (AdvWave-style, applied from a frozen perturbation vector), and
the content-preserving nuisance transforms (pitch / time / gain) used both as
weak "acoustic" variants and as the benign nuisance controls for whitening.

Design rules (AGENTS.md):
- Pure functions over ``np.ndarray`` mono float32 waveforms; no global state.
- Heavy audio deps (``librosa``, ``soundfile``, ``scipy.signal``) are imported
  lazily inside functions so the base env stays CPU-import-clean.
- Every stochastic op takes an explicit ``numpy.random.Generator`` (or integer
  seed) so a rendered cohort is reproducible and auditable by hashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "load_wav",
    "save_wav",
    "rms",
    "apply_gain",
    "apply_gain_db",
    "scale_to_snr",
    "add_gaussian_noise",
    "mix_overlay",
    "pitch_shift",
    "time_stretch",
    "bandlimit",
    "apply_perturbation",
    "as_generator",
]

_EPS = 1e-12


def as_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Coerce a seed/Generator/None into a numpy Generator (None -> seed 0)."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(0)
    return np.random.default_rng(int(rng))


def load_wav(path: str | Path, sr: int = 16_000) -> np.ndarray:
    """Load a mono float32 waveform at ``sr`` (librosa resamples if needed)."""
    import librosa

    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(wav, dtype=np.float32)


def save_wav(path: str | Path, wav: np.ndarray, sr: int = 16_000) -> None:
    """Write a mono float32 waveform, clipped to [-1, 1]."""
    import soundfile as sf

    out = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), out, sr)


def rms(wav: np.ndarray) -> float:
    """Root-mean-square amplitude (energy proxy)."""
    w = np.asarray(wav, dtype=np.float64)
    return float(np.sqrt(np.mean(w * w) + _EPS))


def apply_gain(wav: np.ndarray, factor: float) -> np.ndarray:
    """Linear gain (amplitude multiplier)."""
    return np.asarray(wav, dtype=np.float32) * np.float32(factor)


def apply_gain_db(wav: np.ndarray, db: float) -> np.ndarray:
    """Gain specified in decibels."""
    return apply_gain(wav, float(10.0 ** (db / 20.0)))


def scale_to_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``noise`` so that ``signal`` sits at ``snr_db`` dB above it.

    Returns the rescaled noise (same length as given). SNR is computed on RMS:
    ``snr_db = 20 log10(rms_signal / rms_noise_scaled)``.
    """
    rs, rn = rms(signal), rms(noise)
    target_rn = rs / (10.0 ** (snr_db / 20.0))
    return np.asarray(noise, dtype=np.float32) * np.float32(target_rn / (rn + _EPS))


def _match_length(clip: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Tile/crop ``clip`` to exactly ``n`` samples (random offset if longer)."""
    clip = np.asarray(clip, dtype=np.float32)
    if len(clip) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(clip) < n:
        reps = int(np.ceil(n / len(clip)))
        clip = np.tile(clip, reps)
    if len(clip) > n:
        off = int(rng.integers(0, len(clip) - n + 1))
        clip = clip[off : off + n]
    return clip.astype(np.float32)


def add_gaussian_noise(
    wav: np.ndarray, snr_db: float, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Add white Gaussian noise at a target SNR (dB)."""
    gen = as_generator(rng)
    wav = np.asarray(wav, dtype=np.float32)
    noise = gen.standard_normal(len(wav)).astype(np.float32)
    noise = scale_to_snr(wav, noise, snr_db)
    return (wav + noise).astype(np.float32)


def mix_overlay(
    wav: np.ndarray,
    interference: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Additively mix an interference clip under ``wav`` at a target SNR (dB).

    The AIA-style building block: the malicious speech stays the dominant
    signal; a benign-content interference clip is layered underneath at
    ``snr_db``. ``interference`` is tiled/cropped to match ``wav`` length.
    """
    gen = as_generator(rng)
    wav = np.asarray(wav, dtype=np.float32)
    inter = _match_length(interference, len(wav), gen)
    inter = scale_to_snr(wav, inter, snr_db)
    return (wav + inter).astype(np.float32)


def pitch_shift(wav: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Content-preserving pitch shift by ``n_steps`` semitones."""
    import librosa

    out = librosa.effects.pitch_shift(
        np.asarray(wav, dtype=np.float32), sr=sr, n_steps=float(n_steps)
    )
    return np.asarray(out, dtype=np.float32)


def time_stretch(wav: np.ndarray, rate: float) -> np.ndarray:
    """Content-preserving time stretch by ``rate`` (>1 = faster/shorter)."""
    import librosa

    out = librosa.effects.time_stretch(np.asarray(wav, dtype=np.float32), rate=float(rate))
    return np.asarray(out, dtype=np.float32)


def bandlimit(
    wav: np.ndarray, sr: int, low_hz: float | None = None, high_hz: float | None = None
) -> np.ndarray:
    """Butterworth band/low/high-pass (telephony-style channel simulation)."""
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * sr
    if low_hz and high_hz:
        sos = butter(4, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    elif high_hz:
        sos = butter(4, high_hz / nyq, btype="lowpass", output="sos")
    elif low_hz:
        sos = butter(4, low_hz / nyq, btype="highpass", output="sos")
    else:
        return np.asarray(wav, dtype=np.float32)
    return np.asarray(sosfiltfilt(sos, np.asarray(wav, dtype=np.float64)), dtype=np.float32)


def apply_perturbation(
    wav: np.ndarray, perturbation: np.ndarray, l_inf: float | None = None
) -> np.ndarray:
    """Add a frozen additive perturbation (AdvWave/universal-trigger style).

    ``perturbation`` is tiled/cropped to ``wav`` length. If ``l_inf`` is given,
    the perturbation is clamped to that per-sample budget before adding.
    """
    wav = np.asarray(wav, dtype=np.float32)
    pert = _match_length(perturbation, len(wav), as_generator(0))
    if l_inf is not None:
        pert = np.clip(pert, -abs(l_inf), abs(l_inf))
    return (wav + pert).astype(np.float32)
