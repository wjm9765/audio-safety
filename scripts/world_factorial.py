#!/usr/bin/env -S uv run python
"""WORLD F0 x spectral-envelope factorial on the verified librosa flip cells: for each
flip cell render 4 high-fidelity conditions from the source neutral wav and regenerate:
  neutral  : WORLD analysis->resynthesis, no shift (sham; should refuse)
  f0        : scale F0 by 2**(p/12), keep envelope + aperiodicity
  formant   : warp spectral envelope by 2**(p/12), keep F0 + aperiodicity
  compound  : BOTH F0 and envelope by 2**(p/12) (approximates the librosa joint shift, clean phase)
Adjudicates whether the librosa flip is F0, formant, their interaction (compound), or a
phase-vocoder phase artifact (none of the above WORLD conditions reproduce it).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def warp_envelope(sp, factor):
    n_frames, n_bins = sp.shape
    src = np.arange(n_bins)
    query = np.clip(src / factor, 0, n_bins - 1)
    out = np.empty_like(sp)
    for i in range(n_frames):
        out[i] = np.interp(query, src, sp[i])
    return np.maximum(out, 1e-16)


def render(wav, sr, semitones, mode):
    import pyworld as pw
    x = np.ascontiguousarray(wav.astype(np.float64))
    f0, sp, ap = pw.wav2world(x, sr)
    fac = 2.0 ** (semitones / 12.0)
    if mode == "neutral":
        pass
    elif mode == "f0":
        f0 = f0 * fac
    elif mode == "formant":
        sp = warp_envelope(sp, fac)
    elif mode == "compound":
        f0 = f0 * fac
        sp = warp_envelope(sp, fac)
    else:
        raise ValueError(mode)
    return pw.synthesize(f0, sp, ap, sr).astype(np.float32)


def main():
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
        for p in sorted(flips):
            jobs.append((it, p, str(args.data_dir / neu["source_path"])))
    print(f"{len(jobs)} flip cells x 4 WORLD conditions")
    outdir = args.run_dir / "pitch_representation/world_factorial_audio"
    outdir.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instr = cfg.dataset.target_generation.instruction
    src_cache, rows = {}, []
    for it, p, src in jobs:
        if src not in src_cache:
            src_cache[src], _ = librosa.load(src, sr=args.sr, mono=True)
        row = {"item_id": it, "pitch": p, "reference_text": byit[it][0.0].get("reference_text")}
        for mode in ("neutral", "f0", "formant", "compound"):
            wav = render(src_cache[src], args.sr, p, mode)
            wp = outdir / f"{it}_p{p:+g}_{mode}.wav"
            sf.write(str(wp), wav, args.sr)
            row[mode] = generate_audio_response(model, processor, str(wp), instr, max_new_tokens=args.max_new_tokens, do_sample=False)
        rows.append(row)
        print(f"  {it[-4:]} p={p:+g} | f0:{row['f0'][:22]!r} formant:{row['formant'][:22]!r} compound:{row['compound'][:22]!r}")
    outp = args.run_dir / "pitch_representation/world_factorial.jsonl"
    outp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nwrote {outp} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
