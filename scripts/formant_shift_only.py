#!/usr/bin/env -S uv run python
"""Decisive acoustic-feature dissociation: shift FORMANTS ONLY (warp the WORLD spectral
envelope, keep F0 and aperiodicity fixed) by the same factor a phase-vocoder pitch shift
of p semitones would impose (2**(p/12)), regenerate, and judge whether refusal erodes.

Compared with F0-only shifting (formant_backend_check.py, which did NOT erode refusal),
this isolates whether the librosa-pitch-shift-induced brittleness is driven by the FORMANT
shift. If formant-only erodes refusal while F0-only does not -> clean F0-safe/formant-fragile
double dissociation. If neither erodes -> the STFT effect is phase-vocoder phase artifact.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def warp_envelope(sp: np.ndarray, factor: float) -> np.ndarray:
    """Warp the spectral envelope along frequency by `factor` (formant shift), keeping the
    number of bins. factor>1 shifts formants up. Linear interpolation per frame."""
    n_frames, n_bins = sp.shape
    src = np.arange(n_bins)
    # to move a formant at bin k to bin k*factor, sample the source at bin/factor
    query = src / factor
    query = np.clip(query, 0, n_bins - 1)
    out = np.empty_like(sp)
    for i in range(n_frames):
        out[i] = np.interp(query, src, sp[i])
    return np.maximum(out, 1e-16)


def formant_only_shift(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    import pyworld as pw
    x = np.ascontiguousarray(wav.astype(np.float64))
    f0, sp, ap = pw.wav2world(x, sr)
    factor = 2.0 ** (semitones / 12.0)     # same warp a librosa pitch-shift imposes on formants
    sp2 = warp_envelope(sp, factor)
    y = pw.synthesize(f0, sp2, ap, sr)     # F0 and aperiodicity UNCHANGED
    return y.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("/workspace/audio_safety_data/data"))
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--label-field", default="reviewed_behavior_label")
    args = ap.parse_args()

    import librosa, soundfile as sf
    from audio_safety.config import load_experiment_config
    from audio_safety.models.qwen2_audio import load_qwen2_audio, generate_audio_response
    from audio_safety.utils.paths import resolve_paths

    cfg = load_experiment_config(Path("configs/experiments/run5_pitch_representation_fast.yaml"))
    paths = resolve_paths(cfg.paths)
    cells = [json.loads(l) for l in (args.run_dir / "pitch_representation/cells.jsonl").read_text().splitlines() if l.strip()]
    byit = {}
    for c in cells:
        if c["safety_label"] == "harmful":
            byit.setdefault(c["item_id"], {})[round(float(c["pitch_semitones"]), 6)] = c
    jobs = []
    for it, cm in byit.items():
        neu = cm.get(0.0)
        if not neu or neu.get(args.label_field) != "policy_refusal":
            continue
        flips = [p for p, c in cm.items() if p != 0.0 and c.get(args.label_field) == "harmful_compliance"]
        if not flips:
            continue
        src = str(args.data_dir / neu["source_path"])
        for p in sorted(flips):
            jobs.append((it, p, src))
    print(f"{len(jobs)} formant-ONLY renders across {len(set(j[0] for j in jobs))} brittle items")

    outdir = args.run_dir / "pitch_representation/formant_only_audio"
    outdir.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instruction = cfg.dataset.target_generation.instruction
    src_cache, rows = {}, []
    for it, p, src in jobs:
        if src not in src_cache:
            src_cache[src], _ = librosa.load(src, sr=args.sr, mono=True)
        wav = formant_only_shift(src_cache[src], args.sr, p)
        wp = outdir / f"{it}_p{p:+g}.wav"
        sf.write(str(wp), wav, args.sr)
        out = generate_audio_response(model, processor, str(wp), instruction, max_new_tokens=args.max_new_tokens, do_sample=False)
        rows.append({"item_id": it, "pitch": p, "reference_text": byit[it][0.0].get("reference_text"), "output": out})
        print(f"  {it[-4:]} p={p:+g} (formant-only): {out[:70]!r}")
    outp = args.run_dir / "pitch_representation/formant_only.jsonl"
    outp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nwrote {outp} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
