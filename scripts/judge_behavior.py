#!/usr/bin/env -S uv run python
"""Run the blinded LLM judges over the Run 4 Stage A arms.

Loads the audio, text, and own-transcript behavior manifests, normalizes them to
a common shape (``modality`` + blind ``judge_request_text`` = ``reference_text``),
then runs every configured judge (micro-batched over OpenRouter) and writes one
judged manifest. Resumable by (item_id, safety_label, style, modality).

Example:
    OPENROUTER_API_KEY=... ./scripts/judge_behavior.py \
        --config configs/experiments/run4_conversion_gap.yaml
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.evaluation.judge import judge_records
from audio_safety.utils.io import load_jsonl, save_jsonl
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="max rows to judge")
    parser.add_argument("--save-every", type=int, default=20, help="flush cadence in rows")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def judged_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("item_id")),
        str(row.get("safety_label")),
        str(row.get("style")),
        str(row.get("modality")),
    )


def _normalize_audio(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        updated = dict(row)
        updated["modality"] = "audio"
        updated.setdefault("judge_request_text", str(row.get("reference_text") or ""))
        out.append(updated)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.conversion_gap is None:
        raise SystemExit("this config has no `conversion_gap` block (Run 4 Stage A)")
    judge_cfg = cfg.conversion_gap.judge
    text_cfg = cfg.conversion_gap.text_arm
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir)

    audio_rows = _normalize_audio(
        load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
    )
    text_path = paths.data_dir / text_cfg.text_outputs_file
    tr_path = paths.data_dir / text_cfg.transcript_outputs_file
    text_rows = load_jsonl(text_path) if text_path.exists() else []
    tr_rows = load_jsonl(tr_path) if tr_path.exists() else []
    all_rows = audio_rows + text_rows + tr_rows
    if args.limit is not None:
        all_rows = all_rows[: args.limit]

    output_path = paths.data_dir / judge_cfg.outputs_file
    judged: list[dict] = []
    done: set[tuple[str, str, str, str]] = set()
    if output_path.exists() and not args.overwrite:
        judged = load_jsonl(output_path)
        done = {judged_key(r) for r in judged if r.get("judge_verdicts")}
        print(f"[judge] resuming from {len(done)} judged rows in {output_path}", flush=True)

    pending = [row for row in all_rows if judged_key(row) not in done]
    print(
        f"[judge] arms: audio={len(audio_rows)} text={len(text_rows)} "
        f"transcript={len(tr_rows)}; pending={len(pending)}; "
        f"judges={judge_cfg.models}; batch_size={judge_cfg.batch_size}",
        flush=True,
    )

    since_save = 0
    for judged_row in judge_records(
        pending,
        judge_cfg,
        request_field="judge_request_text",
        response_field="output",
    ):
        judged.append(judged_row)
        since_save += 1
        if since_save >= args.save_every:
            save_jsonl(judged, output_path)
            since_save = 0
    save_jsonl(judged, output_path)
    print(f"[judge] wrote {len(judged)} judged rows -> {output_path}")


if __name__ == "__main__":
    main()
