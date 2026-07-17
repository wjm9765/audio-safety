#!/usr/bin/env -S uv run python
"""Split one ALMGuard arm's generations into batches for local agent adjudication.

Each generation is judged INDEPENDENTLY (blind to the other defense arm), so this
consumes a single arm JSONL (undefended OR defended) and emits self-describing
``batch_*.json`` files. A labeller reads a batch and writes ``labels_*.json`` with,
per record_id, ``reviewed_behavior_label`` (from that row's ``allowed_labels``) and
an optional ``reason``. No OpenRouter / external judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_safety.evaluation.agent_judge_single import build_arm_batches
from audio_safety.utils.io import load_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True, help="one arm JSONL (record_id, output)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-field", default="output")
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_jsonl(args.arm)
    batches = build_arm_batches(
        rows, output_field=args.output_field, batch_size=args.batch_size
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for existing in args.out_dir.glob("batch_*.json"):
        existing.unlink()
    for index, batch in enumerate(batches):
        path = args.out_dir / f"batch_{index:03d}.json"
        path.write_text(json.dumps(batch, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(
        f"[judge] {len(rows)} arm rows -> {len(batches)} batches "
        f"(batch_size={args.batch_size}) in {args.out_dir}"
    )


if __name__ == "__main__":
    main()
