#!/usr/bin/env -S uv run python
"""Run 4 causal-attribution: judge patched trace outputs (blind, by trace_id).

Each causal-trace record is judged INDEPENDENTLY with the same blinded OpenRouter
judge(s) used for the §8 flips, keyed by its unique ``trace_id`` (never by
item/style/modality, which collide across patch conditions). The judge sees only
the recipient item's blind request text and the patched response.

Example:
    ./scripts/judge_traces.py --config configs/experiments/run4_attack_flip.yaml \
        --run-name run4_causal_trace
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.evaluation.judge import judge_records
from audio_safety.utils.io import load_jsonl, save_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-name", type=str, default="run4_causal_trace")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    ct = cfg.causal_trace
    if ct is None or not ct.enabled:
        raise SystemExit("config has no enabled causal_trace block")
    if cfg.conversion_gap is None:
        raise SystemExit("config has no conversion_gap.judge block to reuse")
    paths = resolve_paths(cfg.paths, output_dir=args.output_dir)
    run_dir = run_output_dir(paths.output_dir, args.run_name)

    judge_cfg = cfg.conversion_gap.judge
    if ct.judge_models is not None:
        judge_cfg = judge_cfg.model_copy(update={"models": list(ct.judge_models)})

    records = load_jsonl(run_dir / ct.records_file)
    judged = list(
        judge_records(
            records,
            judge_cfg,
            request_field="judge_request_text",
            response_field="output",
        )
    )
    save_jsonl(judged, run_dir / ct.judged_file)
    print(f"[judge-traces] judged {len(judged)} traces with {judge_cfg.models} "
          f"-> {run_dir / ct.judged_file}")
    print(f"[judge-traces] next: ./scripts/analyze_causal_trace.py --config {args.config} "
          f"--run-name {args.run_name}")


if __name__ == "__main__":
    main()
