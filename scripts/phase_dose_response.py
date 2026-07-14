#!/usr/bin/env -S uv run python
"""Run 7 dose-response (Codex-recommended decisive experiment): on the SAME pitched
magnitude, interpolate STFT phase from locked (alpha=0) to standard/incoherent (alpha=1):
  phi_alpha = phi_locked + alpha * wrap(phi_standard - phi_locked)
Tests the phase-by-pitched-spectrum interaction the failed neutral-magnitude transplant
could not. If margin erosion, L18 refusal-axis displacement, flip rate, and decoding
failure rise MONOTONICALLY with alpha, the causal chain closes:
  phase incoherence -> processor log-mel delta -> frozen-axis displacement -> margin
  erosion -> refuse->comply.

Renders pv_lambda at alpha in {0,.25,.5,.75,1} for the 91 refusers x {-3,+3}, captures
first-token margin + L18 P2 (one forward) and a greedy behavior response. Writes
<run>/pitch_dose/{cells.jsonl, activations.npz}. Judge with judge_pitch_cells then
phase_dose_analyze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SRC_RUN = "run5_20260714_0308_pitch_n150"
LAYERS = [16, 18, 20]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("/workspace/audio_safety_data/data"))
    ap.add_argument("--src-run", default=SRC_RUN)
    ap.add_argument("--signs", type=float, nargs="+", default=[-3.0, 3.0])
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--limit", type=int, default=91)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    from audio_safety.config import load_experiment_config
    from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio
    from audio_safety.pipelines.pitch_representation import _capture_cell, _first_token_ids
    from audio_safety.utils.paths import resolve_paths
    from audio_safety.evaluation.phase_ops import pitch_shift_custom

    cfg = load_experiment_config(Path("configs/experiments/run5_pitch_representation_fast.yaml"))
    paths = resolve_paths(cfg.paths)
    gate = cfg.pitch_representation.model_copy(update={"llm_layers": LAYERS})
    src_cells = [json.loads(l) for l in
                 (paths.output_dir / args.src_run / "pitch_representation/cells.jsonl").read_text().splitlines()
                 if l.strip()]
    refusers = [c for c in src_cells if c["safety_label"] == "harmful"
                and round(float(c["pitch_semitones"]), 6) == 0.0
                and c.get("reviewed_behavior_label") == "policy_refusal"][:args.limit]
    print(f"{len(refusers)} refusers x {len(args.signs)} signs x {len(args.alphas)} alphas")

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instr = cfg.dataset.target_generation.instruction
    refusal_ids = _first_token_ids(processor.tokenizer, gate.refusal_prefixes)
    compliance_ids = _first_token_ids(processor.tokenizer, gate.compliance_prefixes)
    outdir = args.run_dir / "pitch_dose"
    (outdir / "audio").mkdir(parents=True, exist_ok=True)
    rows, p2_stack = [], []

    for n, item in enumerate(refusers):
        it = item["item_id"]
        y_neu, _ = librosa.load(str(args.data_dir / item["source_path"]), sr=args.sr, mono=True)
        for sign in args.signs:
            for a in args.alphas:
                y = pitch_shift_custom(y_neu, args.sr, sign, mode="lambda", lam=float(a))
                wp = outdir / "audio" / f"{it}_p{sign:+g}_a{a:.2f}.wav"
                sf.write(str(wp), y, args.sr)
                states, trace = _capture_cell(model, processor, wp, cfg, gate, refusal_ids, compliance_ids)
                llm_layers = [int(v) for v in trace["llm_layers"]]
                p2 = states["llm_p2"]
                out = generate_audio_response(model, processor, wp, instr,
                                              max_new_tokens=args.max_new_tokens, do_sample=False)
                rows.append({
                    "activation_index": len(p2_stack), "item_id": it, "sign": sign, "alpha": a,
                    "condition": f"pv_lambda_{a:.2f}", "safety_label": "harmful",
                    "reference_text": item["reference_text"],
                    "refusal_margin": float(trace["refusal_margin"]), "output": out,
                    "variant_path": str(wp.relative_to(args.run_dir)),
                })
                p2_stack.append(np.stack([p2[llm_layers.index(L)] for L in LAYERS]))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(refusers)} items ({len(rows)} cells)")

    (outdir / "cells.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    np.savez_compressed(outdir / "activations.npz", llm_p2=np.stack(p2_stack), llm_layers=np.asarray(LAYERS))
    print(f"wrote {outdir/'cells.jsonl'} ({len(rows)} cells) + activations.npz {np.stack(p2_stack).shape}")


if __name__ == "__main__":
    main()
