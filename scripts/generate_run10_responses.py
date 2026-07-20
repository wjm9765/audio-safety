#!/usr/bin/env -S uv run python
"""Generate Qwen2-Audio full responses on the Run 10 confirmatory cohort (behavioral check).

The L18 causal test uses the first-token refusal MARGIN M. That can move the surface
"Sorry/Sure" prefix without changing whether the full answer is actually harmful (Codex
threat #4). So this generates the model's ACTUAL responses (greedy, same system prompt +
instruction as the margin measurement) on the clean(neutral) and pv_standard audio, so a
judge can decide whether the attack produced genuinely harmful compliance -- not just an
eroded first-token logit.

Consumes the confirmatory manifests (recognition/confirmatory/clean_pvstd_{m3,p3}.jsonl);
dedupes by audio path (the neutral clean is shared across signs). Writes
<run>/behavior/responses.jsonl with {item_id, condition, sign, path, reference_text, response}.
GPU-only, greedy decoding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--manifests", nargs="+", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=Path("configs/experiments/run5_pitch_representation_fast.yaml"))
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from audio_safety.config import load_experiment_config
    from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio
    from audio_safety.utils.paths import resolve_paths

    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths)
    gate = cfg.pitch_representation
    instr = cfg.dataset.target_generation.instruction

    # collect unique (path) rows across manifests; neutral clean is shared across signs
    by_path: dict[str, dict] = {}
    for man in args.manifests:
        for line in man.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cond = "clean" if str(r["style"]) == "neutral" else str(r["style"])
            by_path.setdefault(str(r["path"]), {
                "item_id": str(r["item_id"]), "condition": cond,
                "sign": round(float(r.get("sign", 0.0)), 6),
                "path": str(r["path"]), "reference_text": r.get("reference_text"),
            })
    rows = list(by_path.values())
    if args.limit:
        rows = rows[: args.limit]

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)

    out_dir = args.run_dir / "behavior"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.jsonl"
    with out_path.open("w") as f:
        for i, r in enumerate(rows, 1):
            resp = generate_audio_response(
                model, processor, r["path"], instr,
                max_new_tokens=args.max_new_tokens,
                system_prompt=gate.system_prompt, do_sample=False,  # greedy, matches margin setup
            )
            r["response"] = resp
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} generated", flush=True)

    print(f"wrote {out_path} ({len(rows)} responses)")


if __name__ == "__main__":
    main()
