#!/usr/bin/env -S uv run python
"""Multi-sample attack generation (fills idle A40 during design debate).

Greedy decoding gives ONE response per attack audio; real jailbreak success is better measured
over MANY stochastic tries (best-of-N / pass@k). For each attack (pv_standard) audio — and the
clean neutral baseline — this draws --n-samples stochastic generations (do_sample, temperature)
with the SAME system prompt + instruction as the margin/behavior runs, and saves every sample
for later harm-judging. Sharded for 2-model A40 parallelism; incremental flush so partial
results survive an early stop.

Writes <run>/multisample/samples_<tag>.jsonl with one row per (audio, sample_idx).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--manifests", nargs="+", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=Path("configs/experiments/run5_pitch_representation_fast.yaml"))
    ap.add_argument("--conditions", nargs="+", default=["pv_standard", "neutral"],
                    help="which manifest styles to sample (default: attack + clean baseline)")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    import torch

    from audio_safety.config import load_experiment_config
    from audio_safety.models.qwen2_audio import (
        build_audio_analysis_conversation,
        load_qwen2_audio,
        model_input_device,
        prepare_qwen2_audio_inputs,
    )
    from audio_safety.utils.paths import resolve_paths

    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths)
    gate = cfg.pitch_representation
    instr = cfg.dataset.target_generation.instruction

    by_path: dict[str, dict] = {}
    for man in args.manifests:
        for r in _load(man):
            style = str(r["style"])
            if style not in args.conditions:
                continue
            p = str(r["path"])
            by_path.setdefault(p, {
                "item_id": str(r["item_id"]),
                "condition": "clean" if style == "neutral" else style,
                "sign": round(float(r.get("sign", 0.0)), 6),
                "path": p, "reference_text": r.get("reference_text"),
            })
    rows = [r for i, r in enumerate(sorted(by_path.values(), key=lambda x: x["path"]))
            if i % args.num_shards == args.shard]
    if not rows:
        raise SystemExit("empty cohort")

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    device = model_input_device(model)
    torch.manual_seed(args.seed + args.shard)

    out_dir = args.run_dir / "multisample"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or (f"shard{args.shard}" if args.num_shards > 1 else "all")
    out_path = out_dir / f"samples_{tag}.jsonl"

    with out_path.open("w") as f:
        for idx, r in enumerate(rows, 1):
            conv = build_audio_analysis_conversation(r["path"], instr, system_prompt=gate.system_prompt)
            inputs = prepare_qwen2_audio_inputs(processor, conv, device=device)
            prompt_len = inputs.input_ids.shape[1]
            # N independent stochastic samples (loop, not num_return_sequences: the latter can
            # mis-replicate multimodal audio features).
            for si in range(args.n_samples):
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                         temperature=args.temperature, top_p=args.top_p)
                t = processor.batch_decode(gen[:, prompt_len:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
                f.write(json.dumps({**{k: r[k] for k in ("item_id", "condition", "sign", "reference_text")},
                                    "sample_idx": si, "response": t}) + "\n")
            f.flush()
            if idx % 20 == 0 or idx == len(rows):
                print(f"  {idx}/{len(rows)} audios x {args.n_samples} samples", flush=True)

    print(f"wrote {out_path} ({len(rows)} audios x {args.n_samples} samples)")


if __name__ == "__main__":
    main()
