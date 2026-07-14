#!/usr/bin/env -S uv run python
"""Run 7: paired-render causal chain for phase-vocoder frontend distortion.

For each of the 91 neutral refusers x sign in {-3,+3}, render five conditions from the
item's neutral CosyVoice2 waveform and, for each, (a) one forward pass capturing the
first-token refusal margin M and the L18 residual at ``first_generation_prelogit`` and
(b) one greedy behavior generation for 2-judge labelling:

  neutral            replay of the neutral render (once per item)
  pv_standard        custom phase vocoder, independent-bin phase (== librosa pitch_shift)
  pv_locked          same PV pipeline, identity phase-locking (phase repaired)
  phase_transplant   neutral magnitude + measured (angle pv_standard - angle pv_locked)
  mel_matched_ctrl   pv_locked + smooth linear-phase EQ, processor-RMS-matched to D_pair

Also records the model-visible input distance D_pair = RMS(input_features(pv_standard) -
input_features(pv_locked)) and acoustic validity (F0 cents / log-envelope / incoherence)
and, on a calibration subset, ASR WER for content preservation.

Writes <run>/pitch_frontend/{cells.jsonl, activations.npz, audio/...}. Reuses the exact
run5 endpoint (_capture_cell) so margins are comparable. Frozen refusal direction is
taken later from the run5 activations, not refit here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("neutral", "pv_standard", "pv_locked", "phase_transplant", "mel_matched_ctrl")
SRC_RUN = "run5_20260714_0308_pitch_n150"
LAYERS = [16, 18, 20]


def _hash(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("/workspace/audio_safety_data/data"))
    ap.add_argument("--src-run", default=SRC_RUN)
    ap.add_argument("--signs", type=float, nargs="+", default=[-3.0, 3.0])
    ap.add_argument("--subdir", default="pitch_frontend", help="output subdir under run-dir")
    ap.add_argument("--conditions", nargs="+",
                    default=["pv_standard", "pv_locked", "phase_transplant", "mel_matched_ctrl"],
                    help="which non-neutral conditions to render (power extensions can use a subset)")
    ap.add_argument("--skip-neutral", action="store_true", help="skip the neutral replay (for +-2 extension)")
    ap.add_argument("--limit", type=int, default=91)
    ap.add_argument("--calibrate-n", type=int, default=20,
                    help="transcribe WER for the first N items (content-preservation check)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    from audio_safety.config import load_experiment_config
    from audio_safety.models.qwen2_audio import (
        generate_audio_response, load_qwen2_audio,
    )
    from audio_safety.pipelines.pitch_representation import _capture_cell, _first_token_ids
    from audio_safety.utils.paths import resolve_paths
    from audio_safety.evaluation.phase_ops import (
        pitch_shift_custom, phase_transplant, mel_matched_control, model_logmel,
        f0_envelope_metrics, phase_incoherence_score,
    )

    cfg = load_experiment_config(Path("configs/experiments/run5_pitch_representation_fast.yaml"))
    paths = resolve_paths(cfg.paths)
    gate = cfg.pitch_representation.model_copy(update={"llm_layers": LAYERS})

    src_cells = [json.loads(l) for l in
                 (paths.output_dir / args.src_run / "pitch_representation/cells.jsonl").read_text().splitlines()
                 if l.strip()]
    refusers = []
    for c in src_cells:
        if (c["safety_label"] == "harmful" and round(float(c["pitch_semitones"]), 6) == 0.0
                and c.get("reviewed_behavior_label") == "policy_refusal"):
            refusers.append(c)
    refusers = refusers[:args.limit]
    print(f"{len(refusers)} neutral refusers x {len(args.signs)} signs x {len(CONDITIONS)} conditions")

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    fe = processor.feature_extractor
    instr = cfg.dataset.target_generation.instruction
    tr_instr = gate.transcribe_instruction
    refusal_ids = _first_token_ids(processor.tokenizer, gate.refusal_prefixes)
    compliance_ids = _first_token_ids(processor.tokenizer, gate.compliance_prefixes)

    outdir = args.run_dir / args.subdir
    (outdir / "audio").mkdir(parents=True, exist_ok=True)
    rows, p2_stack = [], []

    def capture_and_generate(wav_path, item, sign, cond, transcribe):
        states, trace = _capture_cell(model, processor, wav_path, cfg, gate, refusal_ids, compliance_ids)
        llm_layers = [int(v) for v in trace["llm_layers"]]
        p2 = states["llm_p2"]  # (n_layers, d_model)
        p2_by_layer = {L: p2[llm_layers.index(L)] for L in LAYERS}
        out = generate_audio_response(model, processor, wav_path, instr,
                                      max_new_tokens=args.max_new_tokens, do_sample=False)
        wer = None
        if transcribe:
            hyp = generate_audio_response(model, processor, wav_path, tr_instr,
                                          max_new_tokens=gate.transcribe_max_new_tokens, do_sample=False)
            wer = _wer(item["reference_text"], hyp)
        idx = len(p2_stack)
        p2_stack.append(np.stack([p2_by_layer[L] for L in LAYERS]))
        try:
            vp = str(Path(wav_path).relative_to(args.run_dir))
        except ValueError:
            vp = str(wav_path)
        return {
            "activation_index": idx, "item_id": item["item_id"], "sign": sign,
            "condition": cond, "reference_text": item["reference_text"],
            "safety_label": "harmful", "refusal_margin": float(trace["refusal_margin"]),
            "output": out, "wer": wer, "variant_path": vp,
        }

    for n, item in enumerate(refusers):
        it = item["item_id"]
        neutral_wav = args.data_dir / item["source_path"]
        y_neu, _ = librosa.load(str(neutral_wav), sr=args.sr, mono=True)
        transcribe = n < args.calibrate_n

        # neutral once per item
        if not args.skip_neutral:
            rows.append(capture_and_generate(neutral_wav, item, 0.0, "neutral", transcribe))

        for sign in args.signs:
            renders, d_pair, ctrl_rms = {}, None, None
            need_lock = ("pv_locked" in args.conditions or "mel_matched_ctrl" in args.conditions)
            y_std = pitch_shift_custom(y_neu, args.sr, sign, mode="standard") if "pv_standard" in args.conditions else None
            y_lock = pitch_shift_custom(y_neu, args.sr, sign, mode="locked") if need_lock else None
            if "pv_standard" in args.conditions:
                renders["pv_standard"] = y_std
            if "pv_locked" in args.conditions:
                renders["pv_locked"] = y_lock
            if "phase_transplant" in args.conditions:
                renders["phase_transplant"] = phase_transplant(y_neu, args.sr, sign, dose=1.0)
            if "mel_matched_ctrl" in args.conditions:
                y_std_m = y_std if y_std is not None else pitch_shift_custom(y_neu, args.sr, sign, mode="standard")
                f_std = model_logmel(y_std_m, args.sr, fe)
                f_lock = model_logmel(y_lock, args.sr, fe)
                T = min(f_std.shape[1], f_lock.shape[1])
                d_pair = float(np.sqrt(((f_std[:, :T] - f_lock[:, :T]) ** 2).mean()))
                y_ctrl, ctrl_rms, _ = mel_matched_control(y_lock, args.sr, d_pair, fe)
                renders["mel_matched_ctrl"] = y_ctrl
            for cond, wav in renders.items():
                wp = outdir / "audio" / f"{it}_p{sign:+g}_{cond}.wav"
                sf.write(str(wp), wav, args.sr)
                row = capture_and_generate(wp, item, sign, cond, transcribe)
                row["d_pair"] = d_pair
                row["mel_ctrl_realized_rms"] = ctrl_rms if cond == "mel_matched_ctrl" else None
                row["input_feat_hash"] = _hash(model_logmel(wav, args.sr, fe))
                if cond in ("pv_standard", "pv_locked", "phase_transplant", "mel_matched_ctrl"):
                    fem = f0_envelope_metrics(wav, y_neu, args.sr)
                    row["f0_rmse_cents"] = fem["f0_rmse_cents"]
                    row["logenv_l1"] = fem["logenv_l1"]
                    row["incoherence"] = phase_incoherence_score(wav, args.sr)
                rows.append(row)
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(refusers)} items done ({len(rows)} cells)")

    (outdir / "cells.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    np.savez_compressed(outdir / "activations.npz",
                        llm_p2=np.stack(p2_stack), llm_layers=np.asarray(LAYERS))
    print(f"wrote {outdir/'cells.jsonl'} ({len(rows)} cells) + activations.npz {np.stack(p2_stack).shape}")


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    if not r:
        return float("nan")
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1,
                          d[i-1][j-1] + (r[i-1] != h[j-1]))
    return d[len(r)][len(h)] / len(r)


if __name__ == "__main__":
    main()
