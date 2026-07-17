#!/usr/bin/env -S uv run python
"""Merge local agent label batches into a validated gate label sidecar."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audio_safety.evaluation.agent_judge_io import merge_label_batches
from audio_safety.utils.io import load_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=str, default="claude_agent_local")
    parser.add_argument("--filter-alpha", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_jsonl(args.paired)
    if args.filter_alpha is not None:
        rows = [r for r in rows if float(r.get("sweep_alpha", -1)) == args.filter_alpha]

    label_rows: list[dict[str, object]] = []
    files = sorted(args.labels_dir.glob("labels_*.json"))
    if not files:
        raise SystemExit(f"no labels_*.json under {args.labels_dir}")
    for path in files:
        payload = json.loads(path.read_text("utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"{path} must contain a JSON array of label objects")
        label_rows.extend(payload)

    merged = merge_label_batches(rows, label_rows, resolution=args.resolution)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), "utf-8")
    counts = Counter(
        (r["gate_role"], arm, r[f"{arm}_reviewed_behavior_label"])
        for r in merged
        for arm in ("undefended", "defended")
    )
    print(f"[merge] {len(merged)} labelled rows from {len(files)} batches -> {args.out}")
    for key, count in sorted(counts.items(), key=str):
        print(f"[merge]   {key[0]:24s} {key[1]:11s} {key[2]:20s} {count}")


if __name__ == "__main__":
    main()
