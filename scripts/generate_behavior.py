#!/usr/bin/env -S uv run python
"""Generate short Qwen2-Audio outputs and heuristic behavior labels.

Example:
    ./scripts/generate_behavior.py \
        --config configs/experiments/exp1_refusal_cone_drift.yaml
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.evaluation import label_behavior_records
from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio
from audio_safety.utils.io import load_jsonl, save_jsonl
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data root")
    parser.add_argument("--cache-dir", type=Path, default=None, help="override model cache root")
    parser.add_argument("--limit", type=int, default=None, help="max rows to generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir, cache_dir=args.cache_dir)
    source_path = paths.data_dir / cfg.dataset.asr.scored_manifest_file
    output_path = paths.data_dir / cfg.dataset.target_generation.outputs_file

    rows = [
        row for row in load_jsonl(source_path) if bool(row.get("transcript_control_passed"))
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    outputs = []
    for row in rows:
        audio_path = paths.data_dir / str(row["path"])
        response = generate_audio_response(
            model,
            processor,
            audio_path,
            cfg.dataset.target_generation.instruction,
            max_new_tokens=cfg.dataset.target_generation.max_new_tokens,
        )
        updated = dict(row)
        updated["output"] = response
        outputs.append(updated)

    labeled = label_behavior_records(outputs)
    save_jsonl(labeled, output_path)
    review_n = sum(bool(row.get("needs_manual_review")) for row in labeled)
    print(f"[behavior] wrote {len(labeled)} outputs -> {output_path}")
    print(f"[behavior] needs manual review: {review_n}/{len(labeled)}")


if __name__ == "__main__":
    main()
