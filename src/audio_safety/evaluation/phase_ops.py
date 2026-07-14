"""Phase-isolating acoustic operators for the LALM refusal DSP-confound audit.

Goal: separate a nominal "pitch shift" into the factors it actually manipulates.
librosa's ``effects.pitch_shift`` uses an STFT phase vocoder with independent-bin
phase propagation, whose loss of vertical/horizontal phase coherence is the classic
source of "phasiness". This module reproduces that path exactly (``pv_standard``) and
provides a *phase-repaired* variant that holds the magnitude interpolation, timing,
resampling, gain and output length identical but restores phase coherence via
identity phase-locking around spectral peaks (``pv_locked``). ``pv_lambda`` circularly
interpolates between them. ``phase_transplant`` injects the *measured* librosa phase
error onto the neutral magnitude (F0/formants unchanged) to test phase sufficiency.

Everything that touches librosa/pyworld/torch is imported lazily so the base
(CPU/torch-free) test environment can import the package.

Design cross-checked with Codex gpt-5.6-sol (2026-07-14):
- necessity  = pv_standard vs pv_locked (same pipeline, phase only) -> flips vanish
- sufficiency = neutral-magnitude phase-residual transplant -> flips return dose-wise
- the model's-eye input is a Whisper log-mel (magnitude only), so phase reaches the
  model ONLY through realized-magnitude distortion after iSTFT/overlap-add: report the
  realized log-mel deviation as the input-level analyzable cause.
"""

from __future__ import annotations

import numpy as np

# librosa pitch_shift internal defaults (STFT for the phase vocoder path)
PV_N_FFT = 2048
PV_HOP = 512


# --------------------------------------------------------------------------- #
# custom phase vocoder: standard (librosa-identical) + identity phase locking  #
# --------------------------------------------------------------------------- #
def _phase_vocoder(D: np.ndarray, rate: float, hop_length: int, n_fft: int,
                   mode: str = "standard", lam: float = 1.0,
                   peak_rel_thresh: float = 1e-3) -> np.ndarray:
    """Phase vocoder reproducing ``librosa.phase_vocoder`` for ``mode='standard'``.

    ``mode='locked'`` applies Laroche-Dolson identity phase-locking: peaks propagate
    with the standard per-bin accumulation, every other bin's synthesis phase is
    re-locked each frame to its region-of-influence peak plus the analysis relative
    phase, restoring vertical coherence while leaving the magnitude untouched.
    ``mode='lambda'`` returns phase_locked + lam*wrap(phase_std - phase_locked).
    """
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)

    # expected phase advance per bin per hop (librosa: linspace(0, pi*hop, n_bins))
    n_bins = D.shape[-2]
    phi_advance = np.linspace(0, np.pi * hop_length, n_bins)

    phase_acc = np.angle(D[..., 0])
    D = np.pad(D, [(0, 0)] * (D.ndim - 1) + [(0, 2)], mode="constant")

    for t, step in enumerate(time_steps):
        columns = D[..., int(step): int(step) + 2]
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])

        if mode == "standard":
            synth_phase = phase_acc
        else:
            synth_phase = _lock_phase(phase_acc, np.angle(columns[..., 0]), mag,
                                      mode, lam, peak_rel_thresh)

        d_stretch[..., t] = mag * np.exp(1j * synth_phase)

        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc = phase_acc + phi_advance + dphase

    return d_stretch


def _lock_phase(phase_std: np.ndarray, ana_phase: np.ndarray, mag: np.ndarray,
                mode: str, lam: float, peak_rel_thresh: float) -> np.ndarray:
    """Identity phase-locking for one frame (1-D over frequency bins)."""
    n = mag.shape[-1]
    thr = peak_rel_thresh * (mag.max() + 1e-12)
    # local maxima above threshold are peaks
    is_peak = np.zeros(n, dtype=bool)
    interior = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]) & (mag[1:-1] > thr)
    is_peak[1:-1] = interior
    peaks = np.flatnonzero(is_peak)
    if peaks.size == 0:
        return phase_std  # nothing to lock (silent frame)
    # nearest-peak region of influence
    nearest = peaks[np.argmin(np.abs(np.arange(n)[:, None] - peaks[None, :]), axis=1)]
    phase_locked = phase_std[nearest] + (ana_phase - ana_phase[nearest])
    if mode == "locked":
        return phase_locked
    # lambda blend: locked + lam * wrap(std - locked)
    d = phase_std - phase_locked
    d = d - 2.0 * np.pi * np.round(d / (2.0 * np.pi))
    return phase_locked + lam * d


def pitch_shift_custom(y: np.ndarray, sr: int, n_steps: float, *,
                       mode: str = "standard", lam: float = 1.0,
                       n_fft: int = PV_N_FFT, hop_length: int = PV_HOP,
                       res_type: str = "soxr_hq") -> np.ndarray:
    """Pitch-shift reproducing ``librosa.effects.pitch_shift`` for mode='standard'.

    mode='locked' repairs phase coherence with every other stage held identical.
    """
    import librosa

    rate = 2.0 ** (-float(n_steps) / 12.0)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D_stretch = _phase_vocoder(D, rate=rate, hop_length=hop_length, n_fft=n_fft,
                               mode=mode, lam=lam)
    len_stretch = int(round(y.shape[-1] / rate))
    y_stretch = librosa.istft(D_stretch, hop_length=hop_length, n_fft=n_fft,
                              dtype=y.dtype, length=len_stretch)
    y_shift = librosa.resample(y_stretch, orig_sr=float(sr) / rate, target_sr=sr,
                               res_type=res_type)
    return librosa.util.fix_length(y_shift, size=y.shape[-1])


def phase_transplant(y_neutral: np.ndarray, sr: int, n_steps: float, *,
                     dose: float = 1.0, n_fft: int = PV_N_FFT,
                     hop_length: int = PV_HOP) -> np.ndarray:
    """Inject the *measured* librosa phase-vocoder incoherence onto neutral magnitude.

    Holds F0 and the spectral envelope at their neutral values (magnitude = neutral)
    and adds dose * (angle(pv_standard) - angle(pv_locked)) to the neutral phase, so a
    resulting flip cannot be attributed to any intended F0/formant change.
    """
    import librosa

    x_std = pitch_shift_custom(y_neutral, sr, n_steps, mode="standard",
                               n_fft=n_fft, hop_length=hop_length)
    x_lock = pitch_shift_custom(y_neutral, sr, n_steps, mode="locked",
                                n_fft=n_fft, hop_length=hop_length)
    S_std = librosa.stft(x_std, n_fft=n_fft, hop_length=hop_length)
    S_lock = librosa.stft(x_lock, n_fft=n_fft, hop_length=hop_length)
    E = np.angle(S_std) - np.angle(S_lock)
    E = E - 2.0 * np.pi * np.round(E / (2.0 * np.pi))  # wrap to (-pi, pi]

    D_neu = librosa.stft(y_neutral, n_fft=n_fft, hop_length=hop_length)
    T = min(D_neu.shape[-1], E.shape[-1])
    D_out = np.abs(D_neu[..., :T]) * np.exp(1j * (np.angle(D_neu[..., :T]) + dose * E[..., :T]))
    y_out = librosa.istft(D_out, n_fft=n_fft, hop_length=hop_length,
                          dtype=y_neutral.dtype, length=y_neutral.shape[-1])
    return y_out


# --------------------------------------------------------------------------- #
# WORLD high-fidelity external controls (source-model differs; use as external) #
# --------------------------------------------------------------------------- #
def world_render(y: np.ndarray, sr: int, semitones: float, mode: str) -> np.ndarray:
    """WORLD analysis->resynthesis. mode in {neutral,f0,formant,compound}."""
    import pyworld as pw

    x = np.ascontiguousarray(y.astype(np.float64))
    f0, sp, ap = pw.wav2world(x, sr)
    fac = 2.0 ** (semitones / 12.0)
    if mode in ("f0", "compound"):
        f0 = f0 * fac
    if mode in ("formant", "compound"):
        n_bins = sp.shape[1]
        src = np.arange(n_bins)
        query = np.clip(src / fac, 0, n_bins - 1)
        sp = np.maximum(np.stack([np.interp(query, src, row) for row in sp]), 1e-16)
    return pw.synthesize(f0, sp, ap, sr).astype(np.float32)


# --------------------------------------------------------------------------- #
# matched-magnitude negative control: smooth zero-phase EQ on pv_locked         #
# --------------------------------------------------------------------------- #
def _smooth_gain_curve(n_bins: int, n_curves: int = 4) -> np.ndarray:
    """Deterministic smooth unit log-gain curve over frequency (fixed for all cells)."""
    f = np.linspace(0.0, 1.0, n_bins)
    curve = np.zeros(n_bins)
    for k in range(1, n_curves + 1):
        curve += np.cos(np.pi * k * f + 0.7 * k) / k
    return curve / (np.abs(curve).max() + 1e-9)


def linear_phase_eq(y: np.ndarray, strength: float, *, n_fft: int = PV_N_FFT,
                    hop_length: int = PV_HOP) -> np.ndarray:
    """Apply a smooth ZERO-PHASE (linear-phase) EQ: real positive gain per bin, so
    magnitude moves smoothly while STFT phase is untouched (phase-coherent by design)."""
    import librosa

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    gain = np.exp(strength * _smooth_gain_curve(D.shape[0]))[:, None]
    y2 = librosa.istft(D * gain, n_fft=n_fft, hop_length=hop_length,
                       dtype=y.dtype, length=y.shape[-1])
    return y2


def mel_matched_control(y_locked: np.ndarray, sr: int, target_rms: float,
                        feature_extractor, *, tol: float = 0.05, max_iter: int = 14):
    """Bisect EQ strength so the control's processor-space log-mel RMS deviation from
    ``pv_locked`` equals ``target_rms`` (= D_pair). Returns (waveform, realized_rms, strength)."""
    ref = model_logmel(y_locked, sr, feature_extractor)
    lo, hi = 0.0, 8.0
    y2 = y_locked
    rms = 0.0
    strength = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        y2 = linear_phase_eq(y_locked, mid)
        m = model_logmel(y2, sr, feature_extractor)
        T = min(m.shape[1], ref.shape[1])
        rms = float(np.sqrt(((m[:, :T] - ref[:, :T]) ** 2).mean()))
        strength = mid
        if abs(rms - target_rms) <= tol * max(target_rms, 1e-6):
            break
        if rms < target_rms:
            lo = mid
        else:
            hi = mid
    return y2, rms, strength


# --------------------------------------------------------------------------- #
# model's-eye log-mel + acoustic validity metrics                              #
# --------------------------------------------------------------------------- #
def model_logmel(y: np.ndarray, sr: int, feature_extractor) -> np.ndarray:
    """The exact Whisper log-mel Qwen2-Audio consumes, trimmed to valid frames."""
    feats = feature_extractor(y, sampling_rate=sr, return_tensors="np")
    mel = np.asarray(feats["input_features"])[0]  # (n_mels, n_frames_padded)
    n_valid = int(np.ceil(len(y) / feature_extractor.hop_length))
    return mel[:, :max(1, min(mel.shape[1], n_valid))]


def logmel_deviation(y: np.ndarray, y_ref: np.ndarray, sr: int, feature_extractor) -> dict:
    """L1/L2 deviation of the model's-eye log-mel vs a reference render."""
    m = model_logmel(y, sr, feature_extractor)
    m0 = model_logmel(y_ref, sr, feature_extractor)
    T = min(m.shape[1], m0.shape[1])
    d = m[:, :T] - m0[:, :T]
    return {"logmel_l1": float(np.abs(d).mean()), "logmel_l2": float(np.sqrt((d ** 2).mean()))}


def f0_envelope_metrics(y: np.ndarray, y_ref: np.ndarray, sr: int) -> dict:
    """pyworld F0 RMSE (cents, voiced) and log spectral-envelope L1 vs reference."""
    import pyworld as pw

    def analyze(x):
        f0, sp, ap = pw.wav2world(np.ascontiguousarray(x.astype(np.float64)), sr)
        return f0, sp

    f0, sp = analyze(y)
    f0r, spr = analyze(y_ref)
    T = min(len(f0), len(f0r))
    voiced = (f0[:T] > 0) & (f0r[:T] > 0)
    if voiced.sum() >= 3:
        cents = 1200.0 * np.log2(f0[:T][voiced] / f0r[:T][voiced])
        f0_rmse_cents = float(np.sqrt(np.mean(cents ** 2)))
    else:
        f0_rmse_cents = float("nan")
    Tf = min(sp.shape[0], spr.shape[0])
    env_l1 = float(np.abs(np.log(sp[:Tf] + 1e-16) - np.log(spr[:Tf] + 1e-16)).mean())
    return {"f0_rmse_cents": f0_rmse_cents, "logenv_l1": env_l1}


def phase_incoherence_score(y: np.ndarray, sr: int, *, n_fft: int = PV_N_FFT,
                            hop_length: int = PV_HOP, mag_frac: float = 0.1) -> float:
    """Horizontal phase-incoherence proxy on high-energy bins.

    Mean absolute second time-difference of the unwrapped phase at bins whose
    magnitude is in the top ``mag_frac`` of the clip, minus the expected linear
    advance. Coherent (tonal) tracks give small values; phase-vocoder 'phasiness'
    inflates it. Reported as a descriptive separation metric, not a certificate.
    """
    import librosa

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    phase = np.unwrap(np.angle(S), axis=1)
    d1 = np.diff(phase, axis=1)
    d2 = np.diff(d1, axis=1)                       # curvature of phase over time
    w = mag[:, 2:]
    thr = np.quantile(mag, 1.0 - mag_frac)
    m = w > thr
    if m.sum() < 10:
        return float("nan")
    return float(np.average(np.abs(d2)[m], weights=w[m]))
