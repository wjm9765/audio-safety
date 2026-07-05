#!/usr/bin/env -S uv run python
"""Generate short Qwen2-Audio outputs and heuristic behavior labels.

Example:
    ./scripts/generate_behavior.py \
        --config configs/experiments/exp1_refusal_cone_drift.yaml
"""

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from audio_safety.config import load_experiment_config
from audio_safety.evaluation import label_output
from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio
from audio_safety.utils.io import load_jsonl, save_jsonl
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data root")
    parser.add_argument("--cache-dir", type=Path, default=None, help="override model cache root")
    parser.add_argument("--limit", type=int, default=None, help="max rows to generate")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="discard existing behavior outputs instead of resuming by row key",
    )
    return parser.parse_args()


def row_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("item_id")),
        str(row.get("safety_label")),
        str(row.get("style")),
    )


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

    outputs: list[dict[str, object]] = []
    completed: set[tuple[str, str, str]] = set()
    if output_path.exists() and not args.overwrite:
        outputs = load_jsonl(output_path)
        completed = {row_key(row) for row in outputs if str(row.get("output") or "").strip()}
        print(
            f"[behavior] resuming from {len(completed)}/{len(rows)} existing outputs in {output_path}",
            flush=True,
        )
    elif output_path.exists() and args.overwrite:
        print(f"[behavior] overwriting existing outputs at {output_path}", flush=True)

    pending = [row for row in rows if row_key(row) not in completed]
    print(f"[behavior] pending rows: {len(pending)}/{len(rows)}", flush=True)

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    progress = tqdm(pending, desc="Qwen2-Audio behavior", unit="row")
    for row in progress:
        key = row_key(row)
        progress.set_postfix_str("/".join(key))
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
        label, refusal_mode, needs_review = label_output(response, safety_label=str(row["safety_label"]))
        updated["behavior_label"] = label
        updated["refusal_mode"] = refusal_mode
        updated["needs_manual_review"] = needs_review
        outputs.append(updated)
        save_jsonl(outputs, output_path)

    review_n = sum(bool(row.get("needs_manual_review")) for row in outputs)
    print(f"[behavior] wrote {len(outputs)} outputs -> {output_path}")
    print(f"[behavior] needs manual review: {review_n}/{len(outputs)}")


if __name__ == "__main__":
    main()
