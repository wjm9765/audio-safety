#!/usr/bin/env -S uv run python
"""Merge one arm's local agent label batches into a validated single-label sidecar."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audio_safety.evaluation.agent_judge_single import merge_arm_labels
from audio_safety.utils.io import load_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True, help="the arm JSONL that was batched")
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-field", default="output")
    parser.add_argument("--resolution", default="claude_agent_local")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_jsonl(args.arm)

    label_rows: list[dict[str, object]] = []
    files = sorted(args.labels_dir.glob("labels_*.json"))
    if not files:
        raise SystemExit(f"no labels_*.json under {args.labels_dir}")
    for path in files:
        payload = json.loads(path.read_text("utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"{path} must contain a JSON array of label objects")
        label_rows.extend(payload)

    merged = merge_arm_labels(
        rows, label_rows, output_field=args.output_field, resolution=args.resolution
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), "utf-8"
    )
    counts = Counter(
        (r["safety_label"], r["condition"], r["reviewed_behavior_label"]) for r in merged
    )
    print(f"[merge] {len(merged)} labelled rows from {len(files)} batches -> {args.out}")
    for key, count in sorted(counts.items(), key=str):
        print(f"[merge]   {key[0]:8s} {key[1]:22s} {key[2]:20s} {count}")


if __name__ == "__main__":
    main()
