#!/usr/bin/env -S uv run python
"""Artifact control: re-render the brittle flip cells with a FORMANT-PRESERVING pitch
shifter (WORLD vocoder / pyworld) instead of the phase-vocoder used in the main run,
regenerate, and judge whether the refuse->comply flips SURVIVE a different resynthesis
path. Also re-renders each brittle item's neutral (p=0) through the same vocoder (sham
round-trip) which should still refuse. Flip survival under a formant-preserving backend
argues the effect is pitch, not a phase-vocoder/formant artifact.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np


def formant_preserving_shift(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    import pyworld as pw
    x = np.ascontiguousarray(wav.astype(np.float64))
    f0, sp, ap = pw.wav2world(x, sr)
    f0s = f0 * (2.0 ** (semitones / 12.0))  # shift pitch, keep spectral envelope (formants)
    y = pw.synthesize(f0s, sp, ap, sr)
    return y.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("/workspace/audio_safety_data/data"))
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--label-field", default="reviewed_behavior_label")
    args = ap.parse_args()

    import librosa, soundfile as sf, torch
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
    jobs = []  # (item, pitch, source_wav)
    for it, cm in byit.items():
        neu = cm.get(0.0)
        if not neu or neu.get(args.label_field) != "policy_refusal":
            continue
        flips = [p for p, c in cm.items() if p != 0.0 and c.get(args.label_field) == "harmful_compliance"]
        if not flips:
            continue
        src = str(args.data_dir / neu["source_path"])
        for p in [0.0] + sorted(flips):
            jobs.append((it, p, src))
    print(f"{len(jobs)} formant-backend renders across {len(set(j[0] for j in jobs))} brittle items")

    outdir = args.run_dir / "pitch_representation/formant_audio"
    outdir.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instruction = cfg.dataset.target_generation.instruction
    rows = []
    src_cache = {}
    for it, p, src in jobs:
        if src not in src_cache:
            src_cache[src], _ = librosa.load(src, sr=args.sr, mono=True)
        wav = formant_preserving_shift(src_cache[src], args.sr, p)
        wp = outdir / f"{it}_p{p:+g}.wav"
        sf.write(str(wp), wav, args.sr)
        out = generate_audio_response(model, processor, str(wp), instruction, max_new_tokens=args.max_new_tokens, do_sample=False)
        rows.append({"item_id": it, "pitch": p, "reference_text": byit[it][0.0].get("reference_text"), "output": out})
        tag = "NEUTRAL" if p == 0 else "FLIP"
        print(f"  {it[-4:]} p={p:+g} [{tag}]: {out[:70]!r}")
    outp = args.run_dir / "pitch_representation/formant_backend.jsonl"
    outp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nwrote {outp} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
