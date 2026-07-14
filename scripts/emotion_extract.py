#!/usr/bin/env -S uv run python
"""Run 8 emotion-probe extraction: for each CosyVoice2 render (item × {harmful,benign}
× {neutral,sad,fearful,angry}), capture the L18 residual at first_generation_prelogit +
first-token refusal margin (same endpoint as run7) and a greedy behavior response.
Writes <run>/emotion/{cells.jsonl, activations.npz}. Reuses the run7 endpoint so the
frozen refusal subspace/axis apply directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = [16, 18, 20]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--jobs", default="render_jobs.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--transcribe-n", type=int, default=30, help="WER content-check on first N harmful items/style")
    args = ap.parse_args()

    from audio_safety.config import load_experiment_config
    from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio
    from audio_safety.pipelines.pitch_representation import _capture_cell, _first_token_ids
    from audio_safety.utils.paths import resolve_paths

    cfg = load_experiment_config(Path("configs/experiments/run5_pitch_representation_fast.yaml"))
    paths = resolve_paths(cfg.paths)
    gate = cfg.pitch_representation.model_copy(update={"llm_layers": LAYERS})
    jobs = [json.loads(l) for l in (args.run_dir / args.jobs).read_text().splitlines() if l.strip()]
    jobs = [j for j in jobs if Path(j["output_path"]).exists()]
    print(f"{len(jobs)} rendered clips to extract")

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instr = cfg.dataset.target_generation.instruction
    tr_instr = gate.transcribe_instruction
    refusal_ids = _first_token_ids(processor.tokenizer, gate.refusal_prefixes)
    compliance_ids = _first_token_ids(processor.tokenizer, gate.compliance_prefixes)

    outdir = args.run_dir / "emotion"
    outdir.mkdir(parents=True, exist_ok=True)
    rows, p2_stack = [], []
    tr_count = {}
    for n, j in enumerate(jobs):
        wav = j["output_path"]
        states, trace = _capture_cell(model, processor, Path(wav), cfg, gate, refusal_ids, compliance_ids)
        llm = [int(v) for v in trace["llm_layers"]]
        p2 = states["llm_p2"]
        out = generate_audio_response(model, processor, wav, instr,
                                      max_new_tokens=args.max_new_tokens, do_sample=False)
        key = (j["safety_label"], j["style"])
        wer = None
        if j["safety_label"] == "harmful" and tr_count.get(key, 0) < args.transcribe_n:
            tr_count[key] = tr_count.get(key, 0) + 1
            hyp = generate_audio_response(model, processor, wav, tr_instr,
                                          max_new_tokens=gate.transcribe_max_new_tokens, do_sample=False)
            wer = _wer(j["text"], hyp)
        rows.append({"activation_index": len(p2_stack), "item_id": j["item_id"],
                     "safety_label": j["safety_label"], "style": j["style"],
                     "reference_text": j["text"], "refusal_margin": float(trace["refusal_margin"]),
                     "output": out, "wer": wer})
        p2_stack.append(np.stack([p2[llm.index(L)] for L in LAYERS]))
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(jobs)} clips ({len(rows)} cells)")

    (outdir / "cells.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    np.savez_compressed(outdir / "activations.npz", llm_p2=np.stack(p2_stack), llm_layers=np.asarray(LAYERS))
    print(f"wrote {outdir/'cells.jsonl'} ({len(rows)}) + activations.npz {np.stack(p2_stack).shape}")


def _wer(ref: str, hyp: str) -> float:
    r, h = (ref or "").lower().split(), (hyp or "").lower().split()
    if not r:
        return float("nan")
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (r[i-1] != h[j-1]))
    return d[len(r)][len(h)] / len(r)


if __name__ == "__main__":
    main()
